#  Copyright (c) 2023 zfit
"""Recurrent polynomials."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pydantic

from typing import Literal

from ..core.serialmixin import SerializableMixin
from ..serialization import SpaceRepr, Serializer
from ..serialization.pdfrepr import BasePDFRepr
from ..util.ztyping import ExtendedInputType, NormInputType

if TYPE_CHECKING:
    import zfit

from typing import Mapping
import abc

import tensorflow as tf

import zfit.z.numpy as znp
from zfit import z

from ..core.basepdf import BasePDF
from ..core.space import Space
from ..settings import ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import SpecificFunctionNotImplemented


def rescale_minus_plus_one(x: tf.Tensor, limits: zfit.Space) -> tf.Tensor:
    """Rescale and shift *x* as *limits* were rescaled and shifted to be in (-1, 1). Useful for orthogonal polynomials.

    Args:
        x: Array like data
        limits: 1-D limits

    Returns:
        The rescaled tensor.
    """
    lim_low, lim_high = limits.limit1d
    x = (2 * x - lim_low - lim_high) / (lim_high - lim_low)
    return x


class RecursivePolynomial(BasePDF):
    """1D polynomial generated via three-term recurrence."""

    def __init__(
        self,
        obs,
        coeffs: list,
        apply_scaling: bool = True,
        coeff0: tf.Tensor | None = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Polynomial",
    ):  # noqa
        """Base class to create 1 dimensional recursive polynomials that can be rescaled. Overwrite _poly_func.

        Args:
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            coeffs: |@doc:pdf.polynomial.init.coeffs| Coefficients of the sum of the polynomial.
               The coefficients of the polynomial, starting with the first order
               term. To set the constant term, use ``coeff0``. |@docend:pdf.polynomial.init.coeffs|
            apply_scaling: |@doc:pdf.polynomial.init.apply_scaling| Rescale the data so that the actual limits represent (-1, 1).
               This is usually wanted as the polynomial is defined in this range.
               Default is ``True``. |@docend:pdf.polynomial.init.apply_scaling|

                .. math::
                   x_{n+1} = recurrence(x_{n}, x_{n-1}, n)

            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
        """
        # 0th coefficient set to 1 by default
        coeff0 = (
            z.constant(1.0) if coeff0 is None else tf.cast(coeff0, dtype=ztypes.float)
        )
        coeffs = convert_to_container(coeffs).copy()
        coeffs.insert(0, coeff0)
        params = {f"c_{i}": coeff for i, coeff in enumerate(coeffs)}
        self._degree = len(coeffs) - 1  # 1 coeff -> 0th degree
        self._apply_scale = apply_scaling
        if apply_scaling and not (isinstance(obs, Space) and obs.n_limits == 1):
            raise ValueError(
                "obs need to be a Space with exactly one limit if rescaling is requested."
            )
        super().__init__(
            obs=obs, name=name, params=params, extended=extended, norm=norm
        )

    def _polynomials_rescale(self, x):
        if self._apply_scale:
            x = rescale_minus_plus_one(x, limits=self.space)
        return x

    @property
    def apply_scaling(self):
        return self._apply_scale

    @property
    def degree(self):
        """Int: degree of the polynomial, starting from 0."""
        return self._degree

    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        x = self._polynomials_rescale(x)
        return self._poly_func(x=x)

    @abc.abstractmethod
    def _poly_func(self, x):
        raise SpecificFunctionNotImplemented


class BaseRecursivePolynomialRepr(BasePDFRepr):
    x: SpaceRepr
    params: Mapping[str, Serializer.types.ParamTypeDiscriminated] = pydantic.Field(
        alias="coeffs"
    )
    apply_scaling: Optional[bool]

    @pydantic.root_validator(pre=True)
    def convert_params(cls, values):  # does not propagate `params` into the fields
        if cls.orm_mode(values):
            values = dict(values)
            values["x"] = values.pop("space")
        return values

    def _to_orm(self, init):
        init["coeff0"], *init["coeffs"] = init.pop("params").values()
        return super()._to_orm(init)


def create_poly(x, polys, coeffs, recurrence):
    degree = len(coeffs) - 1
    polys = do_recurrence(x, polys=polys, degree=degree, recurrence=recurrence)
    sum_polys = znp.sum([coeff * poly for coeff, poly in zip(coeffs, polys)], axis=0)
    return sum_polys


def do_recurrence(x, polys, degree, recurrence):
    polys = [polys[0](x), polys[1](x)]
    for i_deg in range(1, degree):  # recurrence returns for the n+1th degree
        polys.append(recurrence(polys[-1], polys[-2], i_deg, x))
    return polys


legendre_polys = [lambda x: tf.ones_like(x), lambda x: x]


@z.function(wraps="zfit_tensor", stateless_args=False)
def legendre_recurrence(p1, p2, n, x):
    """Recurrence relation for Legendre polynomials.

    .. math::
         (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)
    """
    return ((2 * n + 1) * znp.multiply(x, p1) - n * p2) / (n + 1)


def legendre_shape(x, coeffs):
    return create_poly(
        x=x, polys=legendre_polys, coeffs=coeffs, recurrence=legendre_recurrence
    )


def legendre_integral(
    limits: ztyping.SpaceType,
    norm: ztyping.SpaceType,
    params: list[zfit.Parameter],
    model: RecursivePolynomial,
):
    """Recursive integral of Legendre polynomials."""
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)
    # if np.allclose((lower_rescaled, upper_rescaled), (-1, 1)):
    #     return z.constant(2.)  #

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    integral_0 = model.params["c_0"] * (upper - lower)  # if polynomial 0 is 1
    if model.degree == 0:
        integral = integral_0
    else:

        def indefinite_integral(limits):
            max_degree = (
                model.degree + 1
            )  # needed +1 for integral, max poly in term for n is n+1
            polys = do_recurrence(
                x=limits,
                polys=legendre_polys,
                degree=max_degree,
                recurrence=legendre_recurrence,
            )
            one_limit_integrals = []
            for degree in range(1, max_degree):
                coeff = model.params[f"c_{degree}"]
                one_limit_integrals.append(
                    coeff
                    * (polys[degree + 1] - polys[degree - 1])
                    / (2.0 * (z.convert_to_tensor(degree)) + 1)
                )
            return z.reduce_sum(one_limit_integrals, axis=0)

        integral = indefinite_integral(upper) - indefinite_integral(lower) + integral_0
        integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width

    return integral


class Legendre(RecursivePolynomial, SerializableMixin):
    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        coeffs: list[ztyping.ParamTypeInput],
        apply_scaling: bool = True,
        coeff0: ztyping.ParamTypeInput | None = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Legendre",
    ):  # noqa
        """Linear combination of Legendre polynomials of order len(coeffs), the coeffs are overall scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of **a single order** of the polynomial is

        .. math::
            (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)

            with
            P_0 = 1
            P_1 = x


        Args:
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            coeffs: A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling: Rescale the data so that the actual limits represent (-1, 1).
            coeff0: The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.name|
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            apply_scaling=apply_scaling,
            coeff0=coeff0,
            extended=extended,
            norm=norm,
        )

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return legendre_shape(x=x, coeffs=coeffs)


class LegendreRepr(BaseRecursivePolynomialRepr):
    _implementation = Legendre
    hs3_type: Literal["Legendre"] = pydantic.Field("Legendre", alias="type")


legendre_limits = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Legendre.register_analytic_integral(func=legendre_integral, limits=legendre_limits)

chebyshev_polys = [lambda x: tf.ones_like(x), lambda x: x]


@z.function(wraps="zfit_tensor", stateless_args=False)
def chebyshev_recurrence(p1, p2, _, x):
    """Recurrence relation for Chebyshev polynomials.

    T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)
    """
    return 2 * znp.multiply(x, p1) - p2


def chebyshev_shape(x, coeffs):
    return create_poly(
        x=x, polys=chebyshev_polys, coeffs=coeffs, recurrence=chebyshev_recurrence
    )


class Chebyshev(RecursivePolynomial, SerializableMixin):
    def __init__(
        self,
        obs,
        coeffs: list,
        apply_scaling: bool = True,
        coeff0: ztyping.ParamTypeInput | None = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Chebyshev",
    ):  # noqa
        """Linear combination of Chebyshev (first kind) polynomials of order len(coeffs), coeffs are scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of **a single order** of the polynomial is

        .. math::
            T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

            with
            T_{0} = 1
            T_{1} = x

        Notice that :math:`T_1` is x as opposed to 2x in Chebyshev polynomials of the second kind.

        Args:
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
            coeffs: A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling: Rescale the data so that the actual limits represent (-1, 1).
            coeff0: The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.name|
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
        )

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return chebyshev_shape(x=x, coeffs=coeffs)


class ChebyshevRepr(BaseRecursivePolynomialRepr):
    _implementation = Chebyshev
    hs3_type: Literal["Chebyshev"] = pydantic.Field("Chebyshev", alias="type")


def func_integral_chebyshev1(limits, norm, params, model):
    lower, upper = limits.rect_limits
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    integral = model.params["c_0"] * (
        upper - lower
    )  # if polynomial 0 is defined as T_0 = 1
    if model.degree >= 1:
        integral += (
            model.params["c_1"] * 0.5 * (upper**2 - lower**2)
        )  # if polynomial 0 is defined as T_0 = 1
    if model.degree >= 2:

        def indefinite_integral(limits):
            max_degree = model.degree + 1
            polys = do_recurrence(
                x=limits,
                polys=chebyshev_polys,
                degree=max_degree,
                recurrence=chebyshev_recurrence,
            )
            one_limit_integrals = []
            for degree in range(2, max_degree):
                coeff = model.params[f"c_{degree}"]
                n_float = z.convert_to_tensor(degree)
                integral = n_float * polys[degree + 1] / (
                    z.square(n_float) - 1
                ) - limits * polys[degree] / (n_float - 1)
                one_limit_integrals.append(coeff * integral)
            return z.reduce_sum(one_limit_integrals, axis=0)

        integral += indefinite_integral(upper) - indefinite_integral(lower)
        integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width
    integral = tf.gather(integral, indices=0, axis=-1)
    return integral


chebyshev1_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev.register_analytic_integral(
    func=func_integral_chebyshev1, limits=chebyshev1_limits_integral
)

chebyshev2_polys = [lambda x: tf.ones_like(x), lambda x: x * 2]


def chebyshev2_shape(x, coeffs):
    return create_poly(
        x=x, polys=chebyshev2_polys, coeffs=coeffs, recurrence=chebyshev_recurrence
    )


class Chebyshev2(RecursivePolynomial, SerializableMixin):
    def __init__(
        self,
        obs,
        coeffs: list,
        apply_scaling: bool = True,
        coeff0: ztyping.ParamTypeInput | None = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Chebyshev2",
    ):  # noqa
        """Linear combination of Chebyshev (second kind) polynomials of order len(coeffs), coeffs are scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of **a single order** of the polynomial is

        .. math::
            T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

            with
            T_{0} = 1
            T_{1} = 2x

        Notice that :math:`T_1` is 2x as opposed to x in Chebyshev polynomials of the first kind.



        Args:
            obs: The default space the PDF is defined in.
            coeffs: A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling: Rescale the data so that the actual limits represent (-1, 1).
            coeff0: The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: Name of the polynomial
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
        )

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return chebyshev2_shape(x=x, coeffs=coeffs)


class Chebyshev2Repr(BaseRecursivePolynomialRepr):
    _implementation = Chebyshev2
    hs3_type: Literal["Chebyshev2"] = pydantic.Field("Chebyshev2", alias="type")


def func_integral_chebyshev2(limits, norm, params, model):
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    # the integral of cheby2_ni is a cheby1_ni+1/(n+1). We add the (n+1) to the coeffs. The cheby1 shape makes
    # the sum for us.
    coeffs_cheby1 = {"c_0": z.constant(0.0, dtype=model.dtype)}

    for name, coeff in params.items():
        n_plus1 = int(name.split("_", 1)[-1]) + 1
        coeffs_cheby1[f"c_{n_plus1}"] = coeff / z.convert_to_tensor(
            n_plus1, dtype=model.dtype
        )
    coeffs_cheby1 = convert_coeffs_dict_to_list(coeffs_cheby1)

    def indefinite_integral(limits):
        return chebyshev_shape(x=limits, coeffs=coeffs_cheby1)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width

    return integral


chebyshev2_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev2.register_analytic_integral(
    func=func_integral_chebyshev2, limits=chebyshev2_limits_integral
)


def generalized_laguerre_polys_factory(alpha=0.0):
    return [lambda x: tf.ones_like(x), lambda x: 1 + alpha - x]


laguerre_polys = generalized_laguerre_polys_factory(alpha=0.0)


def generalized_laguerre_recurrence_factory(alpha=0.0):
    @z.function(wraps="zfit_tensor", stateless_args=False)
    def generalized_laguerre_recurrence(p1, p2, n, x):
        """Recurrence relation for Laguerre polynomials.

        :math:`(n+1) L_{n+1}(x) = (2n + 1 + \alpha - x) L_{n}(x) - (n + \alpha) L_{n-1}(x)`
        """
        return (znp.multiply(2 * n + 1 + alpha - x, p1) - (n + alpha) * p2) / (n + 1)

    return generalized_laguerre_recurrence


laguerre_recurrence = generalized_laguerre_recurrence_factory(alpha=0.0)


def generalized_laguerre_shape_factory(alpha=0.0):
    recurrence = generalized_laguerre_recurrence_factory(alpha=alpha)
    polys = generalized_laguerre_polys_factory(alpha=alpha)

    def general_laguerre_shape(x, coeffs):
        return create_poly(x=x, polys=polys, coeffs=coeffs, recurrence=recurrence)

    return general_laguerre_shape


laguerre_shape = generalized_laguerre_shape_factory(alpha=0.0)
laguerre_shape_alpha_minusone = generalized_laguerre_shape_factory(
    alpha=-1.0
)  # for integral


class Laguerre(RecursivePolynomial, SerializableMixin):
    def __init__(
        self,
        obs,
        coeffs: list,
        apply_scaling: bool = True,
        coeff0: ztyping.ParamTypeInput | None = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Laguerre",
    ):  # noqa
        """Linear combination of Laguerre polynomials of order len(coeffs), the coeffs are overall scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of **a single order** of the polynomial is

        .. math::

            (n+1) L_{n+1}(x) = (2n + 1 + \alpha - x) L_{n}(x) - (n + \alpha) L_{n-1}(x)

        with
        P_0 = 1
        P_1 = 1 - x


        Args:
            obs: The default space the PDF is defined in.
            coeffs: A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling: Rescale the data so that the actual limits represent (-1, 1).
            coeff0: The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: Name of the polynomial
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
        )

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return laguerre_shape(x=x, coeffs=coeffs)


class LaguerreRepr(BaseRecursivePolynomialRepr):
    _implementation = Laguerre
    hs3_type: Literal["Laguerre"] = pydantic.Field("Laguerre", alias="type")


def func_integral_laguerre(limits, norm, params: dict, model):
    """The integral of the simple laguerre polynomials.

    Defined as :math:`\\int L_{n} = (-1) L_{n+1}^{(-1)}` with :math:`L^{(\alpha)}` the generalized Laguerre polynom.

    Args:
        limits:
        norm:
        params:
        model:

    Returns:
    """
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    # The laguerre shape makes the sum for us. setting the 0th coeff to 0, since no -1 term exists.
    coeffs_laguerre_nup = {
        f'c_{int(n.split("_", 1)[-1]) + 1}': c
        for i, (n, c) in enumerate(params.items())
    }  # increase n -> n+1 of naming
    coeffs_laguerre_nup["c_0"] = tf.constant(0.0, dtype=model.dtype)
    coeffs_laguerre_nup = convert_coeffs_dict_to_list(coeffs_laguerre_nup)

    def indefinite_integral(limits):
        return -1 * laguerre_shape_alpha_minusone(x=limits, coeffs=coeffs_laguerre_nup)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width
    return integral


laguerre_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Laguerre.register_analytic_integral(
    func=func_integral_laguerre, limits=laguerre_limits_integral
)

hermite_polys = [lambda x: tf.ones_like(x), lambda x: 2 * x]


@z.function(wraps="zfit_tensor", stateless_args=False)
def hermite_recurrence(p1, p2, n, x):
    """Recurrence relation for Hermite polynomials (physics).

    :math:`H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)`
    """
    return 2 * (znp.multiply(x, p1) - n * p2)


def hermite_shape(x, coeffs):
    return create_poly(
        x=x, polys=hermite_polys, coeffs=coeffs, recurrence=hermite_recurrence
    )


class Hermite(RecursivePolynomial, SerializableMixin):
    def __init__(
        self,
        obs,
        coeffs: list,
        apply_scaling: bool = True,
        coeff0: ztyping.ParamTypeInput | None = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Hermite",
    ):  # noqa
        """Linear combination of Hermite polynomials (for physics) of order len(coeffs), with coeffs as scaling factors.

        The 0th coefficient is set to 1 by default but can be explicitly set with *coeff0*. Since the PDF normalization
        removes a degree of freedom, the 0th coefficient is redundant and leads to an arbitrary overall scaling of all
        parameters.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The recursive definition of **a single order** of the polynomial is

        .. math::

            H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)

        with
        P_0 = 1
        P_1 = 2x

        Args:
            obs: The default space the PDF is defined in.
            coeffs: A list of the coefficients for the polynomials of order 1+ in the sum.
            apply_scaling: Rescale the data so that the actual limits represent (-1, 1).
            coeff0: The scaling factor of the 0th order polynomial. If not given, it is set to 1.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: Name of the polynomial
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
        )

    def _poly_func(self, x):
        coeffs = convert_coeffs_dict_to_list(self.params)
        return hermite_shape(x=x, coeffs=coeffs)


class HermiteRepr(BaseRecursivePolynomialRepr):
    _implementation = Hermite
    hs3_type: Literal["Hermite"] = pydantic.Field("Hermite", alias="type")


def func_integral_hermite(limits, norm, params, model):
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    # the integral of hermite is a hermite_ni. We add the ni to the coeffs.
    coeffs = {"c_0": z.constant(0.0, dtype=model.dtype)}

    for name, coeff in params.items():
        ip1_coeff = int(name.split("_", 1)[-1]) + 1
        coeffs[f"c_{ip1_coeff}"] = coeff / z.convert_to_tensor(
            ip1_coeff * 2.0, dtype=model.dtype
        )
    coeffs = convert_coeffs_dict_to_list(coeffs)

    def indefinite_integral(limits):
        return hermite_shape(x=limits, coeffs=coeffs)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.area()  # rescale back to whole width

    return integral


hermite_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Hermite.register_analytic_integral(
    func=func_integral_hermite, limits=hermite_limits_integral
)


def convert_coeffs_dict_to_list(coeffs: Mapping) -> list:
    # HACK(Mayou36): how to solve elegantly? yield not a param, only a dependent?
    coeffs_list = []
    for i in range(len(coeffs)):
        try:
            coeffs_list.append(coeffs[f"c_{i}"])
        except (
            KeyError
        ):  # happens, if there are other parameters in there, such as a yield
            break
    return coeffs_list


# EOF
