#  Copyright (c) 2019 zfit
"""Recurrent polynomials."""
import abc
from typing import List, Dict, Optional, Mapping

import tensorflow as tf
from zfit import ztf
from ..util import ztyping
from ..util.container import convert_to_container
from ..settings import ztypes
from ..core.limits import Space
from ..core.basepdf import BasePDF


def rescale_minus_plus_one(x: tf.Tensor, limits: "zfit.Space") -> tf.Tensor:
    """Rescale and shift *x* as *limits* were rescaled and shifted to be in (-1, 1). Useful for orthogonal polynomials.

    Args:
        x: Array like data
        limits: 1-D limits

    Returns:
        tf.Tensor: the rescaled tensor.
    """
    lim_low, lim_high = limits.limit1d
    x = (2 * x - lim_low - lim_high) / (lim_high - lim_low)
    return x


class RecursivePolynomial(BasePDF):
    """1D polynomial generated via three-term recurrence.


    """

    def __init__(self, obs, coeffs: list,
                 apply_scaling: bool = True, coeff0: Optional[tf.Tensor] = None,
                 name: str = "Polynomial"):  # noqa
        """Base class to create 1 dimensional recursive polynomials that can be rescaled. Overwrite _poly_func.

        Args:
            coeffs (list): Coefficients for each polynomial. Used to calculate the degree.
            apply_scaling (bool): Rescale the data so that the actual limits represent (-1, 1).

                .. math::
                   x_{n+1} = recurrence(x_{n}, x_{n-1}, n)

        """
        # 0th coefficient set to 1 by default
        coeff0 = ztf.constant(1.) if coeff0 is None else tf.cast(coeff0, dtype=ztypes.float)
        coeffs = convert_to_container(coeffs).copy()
        coeffs.insert(0, coeff0)
        params = {f"c_{i}": coeff for i, coeff in enumerate(coeffs)}
        self._degree = len(coeffs) - 1  # 1 coeff -> 0th degree
        self._do_scale = apply_scaling
        if apply_scaling and not (isinstance(obs, Space) and obs.n_limits == 1):
            raise ValueError("obs need to be a Space with exactly one limit if rescaling is requested.")
        super().__init__(obs=obs, name=name, params=params)

    def _polynomials_rescale(self, x):
        if self._do_scale:
            x = rescale_minus_plus_one(x, limits=self.space)
        return x

    @property
    def degree(self):
        """int: degree of the polynomial, starting from 0."""
        return self._degree

    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        x = self._polynomials_rescale(x)
        return self._poly_func(x=x)

    @abc.abstractmethod
    def _poly_func(self, x):
        raise NotImplementedError


def create_poly(x, polys, coeffs, recurrence):
    degree = len(coeffs) - 1
    polys = do_recurrence(x, polys=polys, degree=degree, recurrence=recurrence)
    sum_polys = tf.reduce_sum([coeff * poly for coeff, poly in zip(coeffs, polys)], axis=0)
    return sum_polys


def do_recurrence(x, polys, degree, recurrence):
    polys = [polys[0](x), polys[1](x)]
    for i_deg in range(1, degree):  # recurrence returns for the n+1th degree
        polys.append(recurrence(polys[-1], polys[-2], i_deg, x))
    return polys


legendre_polys = [lambda x: tf.ones_like(x), lambda x: x]


def legendre_recurrence(p1, p2, n, x):
    """Recurrence relation for Legendre polynomials.

    .. math::
         (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)

    """
    return ((2 * n + 1) * tf.multiply(x, p1) - n * p2) / (n + 1)


def legendre_shape(x, coeffs):
    return create_poly(x=x, polys=legendre_polys, coeffs=coeffs, recurrence=legendre_recurrence)


def legendre_integral(limits: ztyping.SpaceType, norm_range: ztyping.SpaceType,
                      params: List["zfit.Parameter"], model: RecursivePolynomial):
    """Recursive integral of Legendre polynomials"""
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)
    # if np.allclose((lower_rescaled, upper_rescaled), (-1, 1)):
    #     return ztf.constant(2.)  #

    lower = ztf.convert_to_tensor(lower_rescaled)
    upper = ztf.convert_to_tensor(upper_rescaled)

    integral_0 = model.params[f"c_0"] * (upper - lower)  # if polynomial 0 is 1
    if model.degree == 0:
        integral = integral_0
    else:

        def indefinite_integral(limits):
            max_degree = model.degree + 1  # needed +1 for integral, max poly in term for n is n+1
            polys = do_recurrence(x=limits, polys=legendre_polys, degree=max_degree,
                                  recurrence=legendre_recurrence)
            one_limit_integrals = []
            for degree in range(1, max_degree):
                coeff = model.params[f"c_{degree}"]
                one_limit_integrals.append(coeff * (polys[degree + 1] - polys[degree - 1]) /
                                           (2. * (ztf.convert_to_tensor(degree)) + 1))
            return ztf.reduce_sum(one_limit_integrals, axis=0)

        integral = indefinite_integral(upper) - indefinite_integral(lower) + integral_0
        integral = tf.reshape(integral, shape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width

    return integral


class Legendre(RecursivePolynomial):

    def __init__(self, obs: ztyping.ObsTypeInput, coeffs: List[ztyping.ParamTypeInput],
                 apply_scaling: bool = True, coeff0: Optional[ztyping.ParamTypeInput] = None,
                 name: str = "Legendre"):  # noqa
        """Linear combination of Legendre polynomials of order len(coeffs), the coeffs are overall scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of _a single order_ of the polynomial is

        .. math::
            (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)

            with
            P_0 = 1
            P_1 = x


        Args:
            obs: The default space the PDF is defined in.
            coeffs (list[params]): A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling (bool): Rescale the data so that the actual limits represent (-1, 1).
            coeff0 (param): The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            name (str): Name of the polynomial
        """
        super().__init__(obs=obs, name=name,
                         coeffs=coeffs, apply_scaling=apply_scaling, coeff0=coeff0)

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return legendre_shape(x=x, coeffs=coeffs)


legendre_limits = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Legendre.register_analytic_integral(func=legendre_integral, limits=legendre_limits)

chebyshev_polys = [lambda x: tf.ones_like(x), lambda x: x]


def chebyshev_recurrence(p1, p2, _, x):
    """Recurrence relation for Chebyshev polynomials.

    T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

    """
    return 2 * tf.multiply(x, p1) - p2


def chebyshev_shape(x, coeffs):
    return create_poly(x=x, polys=chebyshev_polys, coeffs=coeffs, recurrence=chebyshev_recurrence)


class Chebyshev(RecursivePolynomial):

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, coeff0: Optional[ztyping.ParamTypeInput] = None,
                 name: str = "Chebyshev"):  # noqa
        """Linear combination of Chebyshev (first kind) polynomials of order len(coeffs), coeffs are scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of _a single order_ of the polynomial is

        .. math::
            T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

            with
            T_{0} = 1
            T_{1} = x

        Notice that :math:`T_1` is x as opposed to 2x in Chebyshev polynomials of the second kind.

        Args:
            obs: The default space the PDF is defined in.
            coeffs (list[params]): A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling (bool): Rescale the data so that the actual limits represent (-1, 1).
            coeff0 (param): The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            name (str): Name of the polynomial
        """
        super().__init__(obs=obs, name=name,
                         coeffs=coeffs, coeff0=coeff0,
                         apply_scaling=apply_scaling)

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return chebyshev_shape(x=x, coeffs=coeffs)


def func_integral_chebyshev1(limits, norm_range, params, model):
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = ztf.convert_to_tensor(lower_rescaled)
    upper = ztf.convert_to_tensor(upper_rescaled)

    integral = model.params[f"c_0"] * (upper - lower)  # if polynomial 0 is defined as T_0 = 1
    if model.degree >= 1:
        integral += model.params[f"c_1"] * 0.5 * (upper ** 2 - lower ** 2)  # if polynomial 0 is defined as T_0 = 1
    if model.degree >= 2:

        def indefinite_integral(limits):
            max_degree = model.degree + 1
            polys = do_recurrence(x=limits, polys=chebyshev_polys, degree=max_degree,
                                  recurrence=chebyshev_recurrence)
            one_limit_integrals = []
            for degree in range(2, max_degree):
                coeff = model.params[f"c_{degree}"]
                n_float = ztf.convert_to_tensor(degree)
                integral = (n_float * polys[degree + 1] / (ztf.square(n_float) - 1)
                            - limits * polys[degree] / (n_float - 1))
                one_limit_integrals.append(coeff * integral)
            return ztf.reduce_sum(one_limit_integrals, axis=0)

        integral += indefinite_integral(upper) - indefinite_integral(lower)
        integral = tf.reshape(integral, shape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width
    return integral


chebyshev1_limits_integral = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev.register_analytic_integral(func=func_integral_chebyshev1, limits=chebyshev1_limits_integral)

chebyshev2_polys = [lambda x: tf.ones_like(x), lambda x: x * 2]


def chebyshev2_shape(x, coeffs):
    return create_poly(x=x, polys=chebyshev2_polys, coeffs=coeffs, recurrence=chebyshev_recurrence)


class Chebyshev2(RecursivePolynomial):

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, coeff0: Optional[ztyping.ParamTypeInput] = None,
                 name: str = "Chebyshev2"):  # noqa
        """Linear combination of Chebyshev (second kind) polynomials of order len(coeffs), coeffs are scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of _a single order_ of the polynomial is

        .. math::
            T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

            with
            T_{0} = 1
            T_{1} = 2x

        Notice that :math:`T_1` is 2x as opposed to x in Chebyshev polynomials of the first kind.



        Args:
            obs: The default space the PDF is defined in.
            coeffs (list[params]): A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling (bool): Rescale the data so that the actual limits represent (-1, 1).
            coeff0 (param): The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            name (str): Name of the polynomial
        """
        super().__init__(obs=obs, name=name,
                         coeffs=coeffs, coeff0=coeff0, apply_scaling=apply_scaling)

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return chebyshev2_shape(x=x, coeffs=coeffs)


def func_integral_chebyshev2(limits, norm_range, params, model):
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = ztf.convert_to_tensor(lower_rescaled)
    upper = ztf.convert_to_tensor(upper_rescaled)

    # the integral of cheby2_ni is a cheby1_ni+1/(n+1). We add the (n+1) to the coeffs. The cheby1 shape makes
    # the sum for us.
    coeffs_cheby1 = {'c_0': ztf.constant(0., dtype=model.dtype)}

    for name, coeff in params.items():
        n_plus1 = int(name.split("_", 1)[-1]) + 1
        coeffs_cheby1[f'c_{n_plus1}'] = coeff / ztf.convert_to_tensor(n_plus1, dtype=model.dtype)
    coeffs_cheby1 = convert_coeffs_dict_to_list(coeffs_cheby1)

    def indefinite_integral(limits):
        return chebyshev_shape(x=limits, coeffs=coeffs_cheby1)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = tf.reshape(integral, shape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width

    return integral


chebyshev2_limits_integral = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev2.register_analytic_integral(func=func_integral_chebyshev2, limits=chebyshev2_limits_integral)


def generalized_laguerre_polys_factory(alpha=0.):
    return [lambda x: tf.ones_like(x), lambda x: 1 + alpha - x]


laguerre_polys = generalized_laguerre_polys_factory(alpha=0.)


def generalized_laguerre_recurrence_factory(alpha=0.):
    def generalized_laguerre_recurrence(p1, p2, n, x):
        """Recurrence relation for Laguerre polynomials.

        :math:`(n+1) L_{n+1}(x) = (2n + 1 + \alpha - x) L_{n}(x) - (n + \alpha) L_{n-1}(x)`

        """
        return (tf.multiply(2 * n + 1 + alpha - x, p1) - (n + alpha) * p2) / (n + 1)

    return generalized_laguerre_recurrence


laguerre_recurrence = generalized_laguerre_recurrence_factory(alpha=0.)


def generalized_laguerre_shape_factory(alpha=0.):
    recurrence = generalized_laguerre_recurrence_factory(alpha=alpha)
    polys = generalized_laguerre_polys_factory(alpha=alpha)

    def general_laguerre_shape(x, coeffs):
        return create_poly(x=x, polys=polys, coeffs=coeffs, recurrence=recurrence)

    return general_laguerre_shape


laguerre_shape = generalized_laguerre_shape_factory(alpha=0.)
laguerre_shape_alpha_minusone = generalized_laguerre_shape_factory(alpha=-1.)  # for integral


class Laguerre(RecursivePolynomial):

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, coeff0: Optional[ztyping.ParamTypeInput] = None,
                 name: str = "Laguerre"):  # noqa
        """Linear combination of Laguerre polynomials of order len(coeffs), the coeffs are overall scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of _a single order_ of the polynomial is

        .. math::

            (n+1) L_{n+1}(x) = (2n + 1 + \alpha - x) L_{n}(x) - (n + \alpha) L_{n-1}(x)

        with
        P_0 = 1
        P_1 = 1 - x


        Args:
            obs: The default space the PDF is defined in.
            coeffs (list[params]): A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling (bool): Rescale the data so that the actual limits represent (-1, 1).
            coeff0 (param): The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            name (str): Name of the polynomial
        """
        super().__init__(obs=obs, name=name, coeffs=coeffs, coeff0=coeff0,
                         apply_scaling=apply_scaling)

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return laguerre_shape(x=x, coeffs=coeffs)


def func_integral_laguerre(limits, norm_range, params: Dict, model):
    """The integral of the simple laguerre polynomials.

    Defined as :math:`\int L_{n} = (-1) L_{n+1}^{(-1)}` with :math:`L^{(\alpha)}` the generalized Laguerre polynom.

    Args:
        limits:
        norm_range:
        params:
        model:

    Returns:

    """
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = ztf.convert_to_tensor(lower_rescaled)
    upper = ztf.convert_to_tensor(upper_rescaled)

    # The laguerre shape makes the sum for us. setting the 0th coeff to 0, since no -1 term exists.
    coeffs_laguerre_nup = {f'c_{int(n.split("_", 1)[-1]) + 1}': c
                           for i, (n, c) in enumerate(params.items())}  # increase n -> n+1 of naming
    coeffs_laguerre_nup['c_0'] = tf.constant(0., dtype=model.dtype)
    coeffs_laguerre_nup = convert_coeffs_dict_to_list(coeffs_laguerre_nup)

    def indefinite_integral(limits):
        return -1 * laguerre_shape_alpha_minusone(x=limits, coeffs=coeffs_laguerre_nup)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = tf.reshape(integral, shape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width
    return integral


laguerre_limits_integral = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Laguerre.register_analytic_integral(func=func_integral_laguerre, limits=laguerre_limits_integral)

hermite_polys = [lambda x: tf.ones_like(x), lambda x: 2 * x]


def hermite_recurrence(p1, p2, n, x):
    """Recurrence relation for Hermite polynomials (physics).

    :math:`H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)`

    """
    return 2 * (tf.multiply(x, p1) - n * p2)


def hermite_shape(x, coeffs):
    return create_poly(x=x, polys=hermite_polys, coeffs=coeffs, recurrence=hermite_recurrence)


class Hermite(RecursivePolynomial):

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, coeff0: Optional[ztyping.ParamTypeInput] = None,
                 name: str = "Hermite"):  # noqa
        """Linear combination of Hermite polynomials (for physics) of order len(coeffs), with coeffs as scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of _a single order_ of the polynomial is

        .. math::

            H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)

        with
        P_0 = 1
        P_1 = 2x

        Args:
            obs: The default space the PDF is defined in.
            coeffs (list[params]): A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling (bool): Rescale the data so that the actual limits represent (-1, 1).
            coeff0 (param): The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            name (str): Name of the polynomial
        """
        super().__init__(obs=obs, name=name, coeffs=coeffs, coeff0=coeff0,
                         apply_scaling=apply_scaling)

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return hermite_shape(x=x, coeffs=coeffs)


def func_integral_hermite(limits, norm_range, params, model):
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = ztf.convert_to_tensor(lower_rescaled)
    upper = ztf.convert_to_tensor(upper_rescaled)

    # the integral of hermite is a hermite_ni. We add the ni to the coeffs.
    coeffs = {'c_0': ztf.constant(0., dtype=model.dtype)}

    for name, coeff in params.items():
        ip1_coeff = int(name.split("_", 1)[-1]) + 1
        coeffs[f'c_{ip1_coeff}'] = coeff / ztf.convert_to_tensor(ip1_coeff * 2., dtype=model.dtype)
    coeffs = convert_coeffs_dict_to_list(coeffs)

    def indefinite_integral(limits):
        return hermite_shape(x=limits, coeffs=coeffs)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = tf.reshape(integral, shape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width

    return integral


hermite_limits_integral = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Hermite.register_analytic_integral(func=func_integral_hermite, limits=hermite_limits_integral)


def convert_coeffs_dict_to_list(coeffs: Mapping) -> List:
    return [coeffs[f"c_{i}"] for i in range(len(coeffs))]

# EOF
