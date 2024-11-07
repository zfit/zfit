#  Copyright (c) 2024 zfit
"""Recurrent polynomials."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import pydantic.v1 as pydantic

from ..core.serialmixin import SerializableMixin
from ..serialization import Serializer, SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..util.ztyping import ExtendedInputType, NormInputType

if TYPE_CHECKING:
    import zfit

import abc
from typing import Mapping

import tensorflow as tf

import zfit.z.numpy as znp
from zfit import z

from ..core.basepdf import BasePDF
from ..core.space import Space, supports
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
    return (2 * x - lim_low - lim_high) / (lim_high - lim_low)


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
        label: str | None = None,
    ):
        """Base class to create 1 dimensional recursive polynomials that can be rescaled. Overwrite _poly_func.

        Args:
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        # 0th coefficient set to 1 by default
        coeff0 = z.constant(1.0) if coeff0 is None else znp.asarray(coeff0, dtype=ztypes.float)
        coeffs = convert_to_container(coeffs).copy()
        coeffs.insert(0, coeff0)
        params = {f"c_{i}": coeff for i, coeff in enumerate(coeffs)}
        self._degree = len(coeffs) - 1  # 1 coeff -> 0th degree
        self._apply_scale = apply_scaling
        if apply_scaling and not (isinstance(obs, Space) and obs._depr_n_limits == 1):
            msg = "obs need to be a Space with exactly one limit if rescaling is requested."
            raise ValueError(msg)
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

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

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm has to be False"
        x = x.unstack_x()
        x = self._polynomials_rescale(x)
        return self._poly_func(x=x, params=params)

    @staticmethod
    @abc.abstractmethod
    def _poly_func(x, params):
        raise SpecificFunctionNotImplemented


class BaseRecursivePolynomialRepr(BasePDFRepr):
    x: SpaceRepr
    params: Mapping[str, Serializer.types.ParamTypeDiscriminated] = pydantic.Field(alias="coeffs")
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
    return znp.sum([coeff * poly for coeff, poly in zip(coeffs, polys)], axis=0)


def do_recurrence(x, polys, degree, recurrence):
    polys = [polys[0](x), polys[1](x)]
    for i_deg in range(1, degree):  # recurrence returns for the n+1th degree
        polys.append(recurrence(polys[-1], polys[-2], i_deg, x))
    return polys


legendre_polys = [lambda x: tf.ones_like(x), lambda x: x]


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def legendre_recurrence(p1, p2, n, x):
    """Recurrence relation for Legendre polynomials.

    .. math::
         (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)
    """
    return ((2 * n + 1) * znp.multiply(x, p1) - n * p2) / (n + 1)


def legendre_shape(x, coeffs):
    return create_poly(x=x, polys=legendre_polys, coeffs=coeffs, recurrence=legendre_recurrence)


def legendre_integral(
    limits: ztyping.SpaceType,
    norm: ztyping.SpaceType,
    params: list[zfit.Parameter],
    model: RecursivePolynomial,
):
    """Recursive integral of Legendre polynomials."""
    del norm  # not used here
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)
    # if np.allclose((lower_rescaled, upper_rescaled), (-1, 1)):
    #     return z.constant(2.)  #

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    integral_0 = params["c_0"] * (upper - lower)  # if polynomial 0 is 1
    if model.degree == 0:
        integral = integral_0
    else:

        def indefinite_integral(limits):
            max_degree = model.degree + 1  # needed +1 for integral, max poly in term for n is n+1
            polys = do_recurrence(
                x=limits,
                polys=legendre_polys,
                degree=max_degree,
                recurrence=legendre_recurrence,
            )
            one_limit_integrals = []
            for degree in range(1, max_degree):
                coeff = params[f"c_{degree}"]
                one_limit_integrals.append(
                    coeff * (polys[degree + 1] - polys[degree - 1]) / (2.0 * (z.convert_to_tensor(degree)) + 1)
                )
            return z.reduce_sum(one_limit_integrals, axis=0)

        integral = indefinite_integral(upper) - indefinite_integral(lower) + integral_0
        integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.volume  # rescale back to whole width

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
        label: str | None = None,
    ):
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

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            apply_scaling=apply_scaling,
            coeff0=coeff0,
            extended=extended,
            norm=norm,
            label=label,
        )

    @staticmethod
    def _poly_func(x, params):
        coeffs = convert_coeffs_dict_to_list(params)
        return legendre_shape(x=x, coeffs=coeffs)


class LegendreRepr(BaseRecursivePolynomialRepr):
    _implementation = Legendre
    hs3_type: Literal["Legendre"] = pydantic.Field("Legendre", alias="type")


legendre_limits = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Legendre.register_analytic_integral(func=legendre_integral, limits=legendre_limits)

chebyshev_polys = [lambda x: tf.ones_like(x), lambda x: x]


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def chebyshev_recurrence(p1, p2, _, x):
    """Recurrence relation for Chebyshev polynomials.

    T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)
    """
    return 2 * znp.multiply(x, p1) - p2


def chebyshev_shape(x, coeffs):
    return create_poly(x=x, polys=chebyshev_polys, coeffs=coeffs, recurrence=chebyshev_recurrence)


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
        label: str | None = None,
    ):
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

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
            label=label,
        )

    @staticmethod
    def _poly_func(x, params):
        coeffs = convert_coeffs_dict_to_list(params)
        return chebyshev_shape(x=x, coeffs=coeffs)


class ChebyshevRepr(BaseRecursivePolynomialRepr):
    _implementation = Chebyshev
    hs3_type: Literal["Chebyshev"] = pydantic.Field("Chebyshev", alias="type")


def func_integral_chebyshev1(limits, norm, params, model):
    del norm  # not used here
    lower, upper = limits.v1.limits
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    integral = params["c_0"] * (upper - lower)  # if polynomial 0 is defined as T_0 = 1
    if model.degree >= 1:
        integral += params["c_1"] * 0.5 * (upper**2 - lower**2)  # if polynomial 0 is defined as T_0 = 1
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
                coeff = params[f"c_{degree}"]
                n_float = z.convert_to_tensor(degree)
                integral = n_float * polys[degree + 1] / (z.square(n_float) - 1) - limits * polys[degree] / (
                    n_float - 1
                )
                one_limit_integrals.append(coeff * integral)
            return z.reduce_sum(one_limit_integrals, axis=0)

        integral += indefinite_integral(upper) - indefinite_integral(lower)
        integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.volume  # rescale back to whole width
    return tf.gather(integral, indices=0, axis=-1)


chebyshev1_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev.register_analytic_integral(func=func_integral_chebyshev1, limits=chebyshev1_limits_integral)

chebyshev2_polys = [lambda x: tf.ones_like(x), lambda x: x * 2]


def chebyshev2_shape(x, coeffs):
    return create_poly(x=x, polys=chebyshev2_polys, coeffs=coeffs, recurrence=chebyshev_recurrence)


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
        label: str | None = None,
    ):
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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
            label=label,
        )

    @staticmethod
    def _poly_func(x, params):
        coeffs = convert_coeffs_dict_to_list(params)
        return chebyshev2_shape(x=x, coeffs=coeffs)


class Chebyshev2Repr(BaseRecursivePolynomialRepr):
    _implementation = Chebyshev2
    hs3_type: Literal["Chebyshev2"] = pydantic.Field("Chebyshev2", alias="type")


def func_integral_chebyshev2(limits, norm, params, model):
    del norm
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
        coeffs_cheby1[f"c_{n_plus1}"] = coeff / z.convert_to_tensor(n_plus1, dtype=model.dtype)
    coeffs_cheby1 = convert_coeffs_dict_to_list(coeffs_cheby1)

    def indefinite_integral(limits):
        return chebyshev_shape(x=limits, coeffs=coeffs_cheby1)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.volume  # rescale back to whole width

    return integral


chebyshev2_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev2.register_analytic_integral(func=func_integral_chebyshev2, limits=chebyshev2_limits_integral)


def generalized_laguerre_polys_factory(alpha=0.0):
    return [lambda x: tf.ones_like(x), lambda x: 1 + alpha - x]


laguerre_polys = generalized_laguerre_polys_factory(alpha=0.0)


def generalized_laguerre_recurrence_factory(alpha=0.0):
    @z.function(wraps="tensor", keepalive=True, stateless_args=False)
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
laguerre_shape_alpha_minusone = generalized_laguerre_shape_factory(alpha=-1.0)  # for integral


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
        label: str | None = None,
    ):
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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
            label=label,
        )

    @staticmethod
    def _poly_func(x, params):
        coeffs = convert_coeffs_dict_to_list(params)
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
    del norm
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    # The laguerre shape makes the sum for us. setting the 0th coeff to 0, since no -1 term exists.
    coeffs_laguerre_nup = {
        f'c_{int(n.split("_", 1)[-1]) + 1}': c for i, (n, c) in enumerate(params.items())
    }  # increase n -> n+1 of naming
    coeffs_laguerre_nup["c_0"] = tf.constant(0.0, dtype=model.dtype)
    coeffs_laguerre_nup = convert_coeffs_dict_to_list(coeffs_laguerre_nup)

    def indefinite_integral(limits):
        return -1 * laguerre_shape_alpha_minusone(x=limits, coeffs=coeffs_laguerre_nup)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.volume  # rescale back to whole width
    return integral


laguerre_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Laguerre.register_analytic_integral(func=func_integral_laguerre, limits=laguerre_limits_integral)

hermite_polys = [lambda x: tf.ones_like(x), lambda x: 2 * x]


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def hermite_recurrence(p1, p2, n, x):
    """Recurrence relation for Hermite polynomials (physics).

    :math:`H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)`
    """
    return 2 * (znp.multiply(x, p1) - n * p2)


def hermite_shape(x, coeffs):
    return create_poly(x=x, polys=hermite_polys, coeffs=coeffs, recurrence=hermite_recurrence)


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
        label: str | None = None,
    ):
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
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        super().__init__(
            obs=obs,
            name=name,
            coeffs=coeffs,
            coeff0=coeff0,
            apply_scaling=apply_scaling,
            extended=extended,
            norm=norm,
            label=label,
        )

    @staticmethod
    def _poly_func(x, params):
        coeffs = convert_coeffs_dict_to_list(params)
        return hermite_shape(x=x, coeffs=coeffs)


class HermiteRepr(BaseRecursivePolynomialRepr):
    _implementation = Hermite
    hs3_type: Literal["Hermite"] = pydantic.Field("Hermite", alias="type")


def func_integral_hermite(limits, norm, params, model):
    del norm
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)

    lower = z.convert_to_tensor(lower_rescaled)
    upper = z.convert_to_tensor(upper_rescaled)

    # the integral of hermite is a hermite_ni. We add the ni to the coeffs.
    coeffs = {"c_0": z.constant(0.0, dtype=model.dtype)}

    for name, coeff in params.items():
        ip1_coeff = int(name.split("_", 1)[-1]) + 1
        coeffs[f"c_{ip1_coeff}"] = coeff / z.convert_to_tensor(ip1_coeff * 2.0, dtype=model.dtype)
    coeffs = convert_coeffs_dict_to_list(coeffs)

    def indefinite_integral(limits):
        return hermite_shape(x=limits, coeffs=coeffs)

    integral = indefinite_integral(upper) - indefinite_integral(lower)
    integral = znp.reshape(integral, newshape=())
    integral *= 0.5 * model.space.volume  # rescale back to whole width

    return integral


hermite_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Hermite.register_analytic_integral(func=func_integral_hermite, limits=hermite_limits_integral)


def rescale_zero_one(x, limits):
    """Rescale and shift *x* as *limits* were rescaled and shifted to be in (0, 1). Useful for polynomials defined in
    (0, 1).

    Args:
        x: Array like data
        limits: 1-D limits

    Returns:
        The rescaled tensor
    """
    lim_low, lim_high = limits.limit1d
    return (x - lim_low) / (lim_high - lim_low)


@z.function(wraps="tensor")
def de_casteljau(x, coeffs):
    """De Casteljau's algorithm."""
    beta = list(coeffs)  # values in this list are overridden
    n = len(beta)
    if n < 1:
        msg = "Need at least one coefficient in de_casteljau of Bernstein."
        raise ValueError(msg)
    for j in range(1, n):
        for k in range(n - j):
            beta[k] = beta[k] * (1 - x) + beta[k + 1] * x
    if n == 1:
        beta[0] = beta[0] * znp.ones_like(x)  # needed for no coefficients, cannot just return scalar beta[0]
    return beta[0]


def bernstein_shape(x, coeffs):
    return de_casteljau(x, coeffs)


class Bernstein(BasePDF, SerializableMixin):
    def __init__(
        self,
        obs,
        coeffs: list,
        apply_scaling: bool = True,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "Bernstein",
        label: str | None = None,
    ):
        """Linear combination of Bernstein polynomials of order len(coeffs) - 1, the coeffs are overall scaling factors.

        Notice that this is already a sum of polynomials and the coeffs are simply scaling the individual orders of the
        polynomials.

        The linear combination of Bernstein polynomials is implemented using De Casteljau's algorithm.

        Args:
            obs: The default space the PDF is defined in.
            coeffs: A list of the coefficients for the polynomials of order len(coeffs) in the sum.
            apply_scaling: Rescale the data so that the actual limits represent (0, 1).
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
        """
        coeffs = convert_to_container(coeffs)
        params = {f"c_{i}": coeff for i, coeff in enumerate(coeffs)}
        self._degree = len(coeffs) - 1
        self._apply_scale = apply_scaling
        if apply_scaling and not (isinstance(obs, Space) and obs._depr_n_limits == 1):
            msg = "obs need to be a Space with exactly one limit if rescaling is requested."
            raise ValueError(msg)
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    def _polynomials_rescale(self, x):
        if self._apply_scale:
            x = rescale_zero_one(x, limits=self.space)
        return x

    @property
    def apply_scaling(self):
        return self._apply_scale

    @property
    def degree(self):
        """Int: degree of the polynomial, starting from 0."""
        return self._degree

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        del norm
        x = x[0]
        x = self._polynomials_rescale(x)
        return self._poly_func(x=x, params=params)

    @staticmethod
    def _poly_func(x, params):
        coeffs = convert_coeffs_dict_to_list(params)
        return bernstein_shape(x=x, coeffs=coeffs)


class BernsteinPDFRepr(BasePDFRepr):
    _implementation = Bernstein
    hs3_type: Literal["Bernstein"] = pydantic.Field("Bernstein", alias="type")
    x: SpaceRepr
    params: Mapping[str, Serializer.types.ParamTypeDiscriminated] = pydantic.Field(alias="coeffs")
    apply_scaling: Optional[bool]

    @pydantic.root_validator(pre=True)
    def convert_params(cls, values):  # does not propagate `params` into the fields
        if cls.orm_mode(values):
            values = dict(values)
            values["x"] = values.pop("space")
        return values

    def _to_orm(self, init):
        init["coeffs"] = list(init.pop("params").values())
        return super()._to_orm(init)


def _coeffs_int(coeffs):
    n = len(coeffs)
    r = [0] * (n + 1)
    for j in range(1, n + 1):
        for k in range(j):
            r[j] += coeffs[k]
    return [rj / n for rj in r]


@z.function(wraps="tensor")
def bernstein_integral_from_xmin_to_x(x, coeffs, limits):
    x = rescale_zero_one(x, limits)
    coeffs = _coeffs_int(coeffs)
    return bernstein_shape(x, coeffs) * limits.volume


def func_integral_bernstein(limits, params, model):
    lower, upper = limits.limit1d

    coeffs = convert_coeffs_dict_to_list(params)

    upper_integral = bernstein_integral_from_xmin_to_x(upper, coeffs, model.space)
    lower_integral = bernstein_integral_from_xmin_to_x(lower, coeffs, model.space)

    return upper_integral - lower_integral


bernstein_limits_integral = Space(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Bernstein.register_analytic_integral(func=func_integral_bernstein, limits=bernstein_limits_integral)


def convert_coeffs_dict_to_list(coeffs: Mapping) -> list:
    # HACK(Mayou36): how to solve elegantly? yield not a param, only a dependent?
    coeffs_list = []
    for i in range(len(coeffs)):
        try:
            coeffs_list.append(coeffs[f"c_{i}"])
        except KeyError:  # happens, if there are other parameters in there, such as a yield
            break
    return coeffs_list


# EOF
