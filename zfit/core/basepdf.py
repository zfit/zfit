"""This  module defines the `BasePdf` that can be used to inherit from in order to build a custom PDF.

The `BasePDF` implements already a lot of ready-to-use functionality like integral, automatic normalization
and sampling.

Defining your own pdf
---------------------

A simple example:
>>> import zfit
>>> import zfit.z.numpy as znp
>>>
>>> class MyGauss(BasePDF):
>>>     def __init__(self, mean, stddev, name="MyGauss"):
>>>         super().__init__(mean=mean, stddev=stddev, name=name)
>>>
>>>     def _unnormalized_pdf(self, x):
>>>         return znp.exp((x - mean) ** 2 / (2 * stddev**2))

Notice that *here* we only specify the *function* and no normalization. This
**No** attempt to **explicitly** normalize the function should be done inside `_unnormalized_pdf`.
The normalization is handled with another method depending on the normalization range specified.
(It *is* possible, though discouraged, to directly provide the *normalized probability* by overriding _pdf(), but
there are other, more convenient ways to add improvements like providing an analytical integrals.)

Before we create an instance, we need to create the variables to initialize it
>>> mean = zfit.Parameter("mean1", 2., 0.1, 4.2)  # signature as in RooFit: *name, initial, lower, upper*
>>> stddev = zfit.Parameter("stddev1", 5., 0.3, 10.)
Let's create an instance and some example data
>>> gauss = MyGauss(mean=mean, stddev=stddev)
>>> example_data = np.random.random(10)
Now we can get the probability
>>> probs = gauss.pdf(x=example_data,norm_range=(-30., 30))  # `norm_range` specifies over which range to normalize
Or the integral
>>> integral = gauss.integrate(limits=(-5, 3.1), norm_range=False)  # norm_range is False -> return unnormalized
integral
Or directly sample from it
>>> sample = gauss.sample(n_draws=1000, limits=(-10, 10))  # draw 1000 samples within (-10, 10)

We can create an extended PDF, which will result in anything using a `norm_range` to not return the
probability but the number probability (the function will be normalized to `yield` instead of 1 inside
the `norm_range`)
>>> yield1 = Parameter("yield1", 100, 0, 1000)
>>> gauss_extended = gauss.create_extended(yield1)
>>> gauss.is_extended
True

>>> integral_extended = gauss.integrate(limits=(-10, 10), norm_range=(-10, 10))  # yields approx 100

For more advanced methods and ways to register analytic integrals or overwrite certain methods, see
also the advanced models in `zfit models <https://github.com/zfit/zfit-tutorials>`_
"""

#  Copyright (c) 2021 zfit

import warnings
from contextlib import suppress
from typing import Dict, Optional, Set, Type, Union

import tensorflow as tf

import zfit.z.numpy as znp
from zfit import z

from ..settings import run, ztypes
from ..util import ztyping
from ..util.cache import invalidate_graph
from ..util.deprecation import deprecated
from ..util.exception import (AlreadyExtendedPDFError, BreakingAPIChangeError,
                              FunctionNotImplemented, NotExtendedPDFError,
                              SpecificFunctionNotImplemented)
from ..util.temporary import TemporarilySet
from .basemodel import BaseModel
from .baseobject import extract_filter_params
from .interfaces import ZfitParameter, ZfitPDF
from .parameter import Parameter, convert_to_parameter
from .sample import extended_sampling
from .space import Space

_BasePDF_USER_IMPL_METHODS_TO_CHECK = {}


def _BasePDF_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the `@supports` decorator.

    Args:
        has_support: If True, flags that it **requires** the `@supports` decorator. If False,
            flags that the `@supports` decorator is **not allowed**.
    """
    if not isinstance(has_support, bool):
        raise TypeError("Has to be boolean.")

    def register(func):
        """Register a method to be checked to (if True) *has* `support` or (if False) has *no* `support`.

        Args:
            func:

        Returns:
            Function:
        """
        name = func.__name__
        _BasePDF_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BasePDF_register_check_support
        return func

    return register


class BasePDF(ZfitPDF, BaseModel):

    def __init__(self, obs: ztyping.ObsTypeInput, params: Dict[str, ZfitParameter] = None, dtype: Type = ztypes.float,
                 name: str = "BasePDF",
                 **kwargs):
        super().__init__(obs=obs, dtype=dtype, name=name, params=params, **kwargs)

        self._yield = None
        self._norm_range = None
        self._normalization_value = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclass_check_support(methods_to_check=_BasePDF_USER_IMPL_METHODS_TO_CHECK,
                                    wrapper_not_overwritten=_BasePDF_register_check_support)

    # @property
    # def space(self) -> "zfit.Space":
    #     if self._norm_range is not None:
    #         space = self._norm_range
    #     else:
    #         space = super().space
    #
    #     return space

    def _check_input_norm_range(self, norm_range, none_is_error=False):
        if norm_range is None:
            norm_range = self.norm_range
        return super()._check_input_norm_range(norm_range=norm_range, none_is_error=none_is_error)

    def _check_input_params(self, *params):
        return tuple(convert_to_parameter(p) for p in params)

    def _func_to_integrate(self, x: ztyping.XType):
        return self.pdf(x, norm_range=False)

    def _func_to_sample_from(self, x):
        return self.pdf(x, norm_range=False)

    @property
    def norm_range(self) -> Union[Space, None, bool]:
        """Return the current normalization range. If None and the `obs` have limits, they are returned.

        Returns:
            The current normalization range.
        """
        norm_range = self._norm_range
        if norm_range is None:
            norm_range = self.space
        return norm_range

    @invalidate_graph
    def set_norm_range(self, norm_range: ztyping.LimitsTypeInput):
        """Set the normalization range (temporarily if used with contextmanager).

        Args:
            norm_range:
        """
        norm_range = self._check_input_norm_range(norm_range=norm_range)

        def setter(value):
            self._norm_range = value

        def getter():
            return self._norm_range

        return TemporarilySet(value=norm_range, setter=setter, getter=getter)

    # TODO: remove below?
    # @property
    # def _yield(self):
    #     """For internal use, the yield or None"""
    #     return self.params.get('yield')
    #
    # @_yield.setter
    # def _yield(self, value):
    #     if value is None:
    #         # unset
    #         self._params.pop('yield', None)  # safely remove if still there
    #     else:
    #         self._params['yield'] = value

    @_BasePDF_register_check_support(True)
    def _normalization(self, limits):
        raise SpecificFunctionNotImplemented

    def normalization(self, limits: ztyping.LimitsType) -> ztyping.XType:
        """Return the normalization of the function (usually the integral over `limits`).

        Args:
            limits:  The limits on where to normalize over

        Returns:
            The normalization value
        """
        limits = self._check_input_limits(limits=limits)

        return self._single_hook_normalization(limits=limits)

    def _single_hook_normalization(self, limits):  # TODO(Mayou36): add yield?
        return self._hook_normalization(limits=limits)

    def _hook_normalization(self, limits):
        return self._call_normalization(limits=limits)  # no _norm_* needed

    def _call_normalization(self, limits):
        # TODO: caching? alternative
        with suppress(FunctionNotImplemented):
            return self._normalization(limits=limits)
        return self._fallback_normalization(limits)

    def _fallback_normalization(self, limits):
        return self._hook_integrate(limits=limits, norm_range=False)

    def _unnormalized_pdf(self, x):
        raise SpecificFunctionNotImplemented

    @deprecated(None, "Use `pdf(norm_range=False)` instead")
    def unnormalized_pdf(self, x: ztyping.XType, component_norm_range: ztyping.LimitsTypeInput = None) -> ztyping.XType:
        """PDF "unnormalized". Use `functions` for unnormalized pdfs. this is only for performance in special cases.

        Args:
            x: The value, have to be convertible to a Tensor
            component_norm_range: The normalization range for the components. Needed for
            certain composition
                pdfs.

        Returns:
            1-dimensional :py:class:`tf.Tensor` containing the unnormalized pdf.
        """
        if component_norm_range is not None:
            raise BreakingAPIChangeError("component norm range should not be given anymore. If you want to set the norm"
                                         " range for the components, use `set_norm_range(..., propagate=True)")
        with self._convert_sort_x(x) as x:
            return self._single_hook_unnormalized_pdf(x)

    def _single_hook_unnormalized_pdf(self, x):
        return self._call_unnormalized_pdf(x=x)

    def _call_unnormalized_pdf(self, x):
        # try:
        return self._unnormalized_pdf(x)

    # except ValueError as error:
    #     raise ShapeIncompatibleError("Most probably, the number of obs the pdf was designed for"
    #                                  "does not coincide with the `n_obs` from the `space`/`obs`"
    #                                  "it received on initialization."
    #                                  "Original Error: {}".format(error))

    @z.function(wraps='model')
    def ext_pdf(self, x: ztyping.XTypeInput, norm_range: ztyping.LimitsTypeInput = None) -> ztyping.XType:
        """Probability density function scaled by yield, normalized over `norm_range`.

        Args:
          x: `float` or `double` `Tensor`.
          norm_range: :py:class:`~zfit.Space` to normalize over

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        if not self.is_extended:
            raise NotExtendedPDFError(f"{self} is not extended, cannot call `ext_pdf`")
        return self.pdf(x=x, norm_range=norm_range) * self.get_yield()

    @z.function(wraps='model')
    def ext_log_pdf(self, x: ztyping.XTypeInput, norm_range: ztyping.LimitsTypeInput = None) -> ztyping.XType:
        """Log of probability density function scaled by yield, normalized over `norm_range`.

        Args:
          x: `float` or `double` `Tensor`.
          norm_range: :py:class:`~zfit.Space` to normalize over

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        if not self.is_extended:
            raise NotExtendedPDFError(f"{self} is not extended, cannot call `ext_pdf`")
        return self.log_pdf(x=x, norm_range=norm_range) + znp.log(self.get_yield())

    @_BasePDF_register_check_support(False)
    def _pdf(self, x, norm_range):
        raise SpecificFunctionNotImplemented

    @z.function(wraps='model')
    def pdf(self, x: ztyping.XTypeInput, norm_range: ztyping.LimitsTypeInput = None) -> ztyping.XType:
        """Probability density function, normalized over `norm_range`.

        Args:
          x: `float` or `double` `Tensor`.
          norm_range: :py:class:`~zfit.Space` to normalize over

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, none_is_error=True)
        with self._convert_sort_x(x) as x:
            value = self._single_hook_pdf(x=x, norm_range=norm_range)
            if run.numeric_checks:
                z.check_numerics(value, message="Check if pdf output contains any NaNs of Infs")
            return z.to_real(value)

    def _single_hook_pdf(self, x, norm_range):
        return self._hook_pdf(x=x, norm_range=norm_range)

    def _hook_pdf(self, x, norm_range):
        return self._norm_pdf(x=x, norm_range=norm_range)

    def _norm_pdf(self, x, norm_range):
        return self._call_pdf(x=x, norm_range=norm_range)

    def _call_pdf(self, x, norm_range):
        with suppress(FunctionNotImplemented):
            return self._pdf(x, norm_range=norm_range)
        with suppress(FunctionNotImplemented):
            return znp.exp(self._log_pdf(x=x, norm_range=norm_range))
        return self._fallback_pdf(x=x, norm_range=norm_range)

    def _fallback_pdf(self, x, norm_range):
        pdf = self._call_unnormalized_pdf(x)
        if norm_range.has_limits:
            pdf /= self._hook_normalization(limits=norm_range)
        return pdf

    @_BasePDF_register_check_support(False)
    def _log_pdf(self, x, norm_range):
        raise SpecificFunctionNotImplemented

    def log_pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        """Log probability density function normalized over `norm_range`.

        Args:
          x: `float` or `double` `Tensor`.
          norm_range: :py:class:`~zfit.Space` to normalize over

        Returns:
          A `Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range)
        with self._convert_sort_x(x) as x:
            return self._single_hook_log_pdf(x=x, norm_range=norm_range)

    def _single_hook_log_pdf(self, x, norm_range):
        return self._hook_log_pdf(x=x, norm_range=norm_range)

    def _hook_log_pdf(self, x, norm_range):
        log_prob = self._norm_log_pdf(x=x, norm_range=norm_range)
        return log_prob

    def _norm_log_pdf(self, x, norm_range):
        return self._call_log_pdf(x=x, norm_range=norm_range)

    def _call_log_pdf(self, x, norm_range):
        with suppress(FunctionNotImplemented):
            return self._log_pdf(x=x, norm_range=norm_range)
        with suppress(FunctionNotImplemented):
            return znp.log(self._pdf(x=x, norm_range=norm_range))
        return self._fallback_log_pdf(x=x, norm_range=norm_range)

    def _fallback_log_pdf(self, x, norm_range):
        return znp.log(self._hook_pdf(x=x, norm_range=norm_range))

    def gradient(self, x: ztyping.XType, norm_range: ztyping.LimitsType, params: ztyping.ParamsTypeOpt = None):
        raise BreakingAPIChangeError("Removed with 0.5.x: is this needed?")

    @z.function(wraps='model')
    def ext_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None) -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits: the limits to integrate over
            norm_range: the limits to normalize over or False to integrate the
                unnormalized probability

        Returns:
            The integral value as a scalar with shape ()
        """
        norm_range = self._check_input_norm_range(norm_range)
        limits = self._check_input_limits(limits=limits)
        if not self.is_extended:
            raise NotExtendedPDFError(f"{self} is not extended, cannot call `ext_pdf`")
        return self.integrate(limits=limits, norm_range=norm_range) * self.get_yield()

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool) -> Union[float, tf.Tensor]:
        if self.is_extended and not norm_range.limits_are_false:
            if log:
                value += znp.log(self.get_yield())
            else:
                value *= self.get_yield()
        return value

    def apply_yield(self, value: Union[float, tf.Tensor], norm_range: ztyping.LimitsTypeInput = False,
                    log: bool = False) -> Union[float, tf.Tensor]:
        """If a norm_range is given, the value will be multiplied by the yield.

        Args:
            value:
            norm_range:
            log:

        Returns:
            Numerical
        """
        norm_range = self._check_input_norm_range(norm_range=norm_range)
        return self._apply_yield(value=value, norm_range=norm_range, log=log)

    @deprecated(None, "Use the public `set_yield` instead.")
    def _set_yield_inplace(self, value: Union[ZfitParameter, float, None]):
        """Make the model extended by setting a yield.

        This does not alter the general behavior of the PDF. If there is a
        `norm_range` given, the output of the above functions does not represent a normalized
        probability density function anymore but corresponds to a number probability.

        Args:
            value:
        """

        self._set_yield(value=value)

    def create_extended(self, yield_: ztyping.ParamTypeInput, name_addition="_extended") -> "ZfitPDF":
        """Return an extended version of this pdf with yield `yield_`. The parameters are shared.

        Args:
            yield_:
            name_addition:

        Returns:
            :py:class:`~zfit.core.interfaces.ZfitPDF`
        """
        # TODO(Mayou36): fix copy
        from zfit.models.functor import ProductPDF
        if isinstance(self, ProductPDF):
            warnings.warn(
                "As `copy` is not yet properly implemented, this may fails (for ProductPDF for example?). This"
                "will be fixed in the future.")
        if self.is_extended:
            raise AlreadyExtendedPDFError("This PDF is already extended, cannot create an extended one.")
        new_pdf = self.copy(name=self.name + str(name_addition))
        new_pdf.set_yield(value=yield_)
        return new_pdf

    def set_yield(self, value):

        """Make the model extended by setting a yield. If possible, prefer to use `create_extended`.

        This does not alter the general behavior of the PDF. The `pdf` and `integrate` and similar methods will
        continue to return the same - normalized to 1 - values. However, not only can this parameter be accessed
        via `get_yield`, the methods `ext_pdf` and `ext_integral` provide a version of `pdf` and `integrate`
        respecetively that is multiplied by the yield.

        These can be useful for plotting and for binned likelihoods.

        Args:
            value ():
        """
        self._set_yield(value=value)

    def _set_yield(self, value: ztyping.ParamTypeInput):
        if value is None:
            raise BreakingAPIChangeError("Cannot unset a yield (anymore).")
        if self.is_extended:
            raise AlreadyExtendedPDFError(f"Cannot extend {self}, is already extended.")
        value = convert_to_parameter(value)
        self.add_cache_deps(value)
        self._yield = value

    @property
    def is_extended(self) -> bool:
        """Flag to tell whether the model is extended or not.

        Returns:
            A boolean.
        """
        return self._yield is not None

    def _hook_sample(self, limits, n):
        if n is None and self.is_extended:
            n = 'extended'
        if isinstance(n, str) and n == 'extended':
            if not self.is_extended:
                raise NotExtendedPDFError("Cannot use 'extended' as value for `n` on a non-extended pdf.")
            samples = extended_sampling(pdfs=self, limits=limits)
        elif isinstance(n, str):
            raise ValueError("`n` is a string and not 'extended'. Other options are currently not implemented.")
        elif n is None:
            raise tf.errors.InvalidArgumentError("`n` cannot be `None` if pdf is not extended.")
        else:
            samples = super()._hook_sample(limits=limits, n=n)
        return samples

    def get_yield(self) -> Union[Parameter, None]:
        """Return the yield (only for extended models).

        Returns:
            The yield of the current model or None
        """
        # if not self.is_extended:
        #     raise zexception.ExtendedPDFError("PDF is not extended, cannot get yield.")
        return self._yield

    def _get_params(self,
                    floating: Optional[bool] = True,
                    is_yield: Optional[bool] = None,
                    extract_independent: Optional[bool] = True) -> Set[ZfitParameter]:

        params = super()._get_params(floating, is_yield=is_yield,
                                     extract_independent=extract_independent)

        if is_yield is not False:
            if self.is_extended:
                yield_params = extract_filter_params(self.get_yield(), floating=floating,
                                                     extract_independent=extract_independent)
                yield_params.update(params)  # putting the yields at the beginning
                params = yield_params
            elif is_yield is True:
                raise NotExtendedPDFError("PDF is not extended but only yield parameters were requested.")
        return params

    def create_projection_pdf(self, limits_to_integrate: ztyping.LimitsTypeInput) -> 'ZfitPDF':
        """Create a PDF projection by integrating out some of the dimensions.

        The new projection pdf is still fully dependent on the pdf it was created with.

        Args:
            limits_to_integrate:

        Returns:
            A pdf without the dimensions from `limits_to_integrate`.
        """
        from ..models.special import SimpleFunctorPDF

        def partial_integrate_wrapped(self_simple, x):
            return self.partial_integrate(x, limits=limits_to_integrate, norm_range=False)

        new_pdf = SimpleFunctorPDF(obs=self.space.get_subspace(obs=[obs for obs in self.obs
                                                                    if obs not in limits_to_integrate.obs]),
                                   pdfs=(self,),
                                   func=partial_integrate_wrapped)
        return new_pdf

    def copy(self, **override_parameters) -> 'BasePDF':
        """Creates a copy of the model.

        Note: the copy model may continue to depend on the original
        initialization arguments.

        Args:
          **override_parameters: String/value dictionary of initialization
            arguments to override with new value.

        Returns:
          A new instance of `type(self)` initialized from the union
            of self.parameters and override_parameters, i.e.,
            `dict(self.parameters, **override_parameters)`.
        """
        obs = self.norm_range

        # HACK(Mayou36): remove once copy is proper implemented
        from ..models.dist_tfp import WrapDistribution
        from ..models.kde import GaussianKDE1DimV1
        from ..models.polynomials import RecursivePolynomial

        if type(self) == WrapDistribution:  # NOT isinstance! Because e.g. Gauss wraps that and takes different args
            parameters = dict(distribution=self._distribution, dist_params=self.dist_params)
        else:
            # HACK END

            parameters = dict(self.params)
            lam = parameters.pop('lambda', None)
            if lam is not None:
                parameters['lam'] = lam

        if type(self) == GaussianKDE1DimV1:
            raise RuntimeError("Cannot copy `GaussianKDE1DimV1` (yet). If you tried to make it extended, use "
                               "`set_yield`"
                               " instead and set it inplace.")
            parameters['data'] = self._original_data

        # HACK(Mayou36): copy the polynomial correct, replace 'c_0' with coeff0/coeff_0 or similar
        if isinstance(self, RecursivePolynomial):
            parameters['coeff0'] = parameters.pop('c_0', None)
            coeffs = []
            i_coeff = 1
            # collect coeffs and convert to 'coeff' list
            while True:
                coeff_name = f'c_{i_coeff}'
                try:
                    coeff = parameters.pop(coeff_name)
                except KeyError:
                    break
                else:
                    coeffs.append(coeff)
                i_coeff += 1
            parameters['coeffs'] = coeffs

        from zfit.models.functor import BaseFunctor, SumPDF
        if isinstance(self, BaseFunctor):
            parameters = {}
            if isinstance(self, SumPDF):
                fracs = self.fracs
                if not self.is_extended:
                    fracs = fracs[:-1]
                parameters.update(fracs=fracs)
            parameters.update(pdfs=self.pdfs)

        parameters.update(obs=obs, name=self.name)
        parameters.update(**override_parameters)
        # if hasattr(self, "distribution"):
        #     parameters.update(distribution=self.distribution)
        yield_ = parameters.pop('yield', None)
        new_instance = type(self)(**parameters)
        if yield_ is not None:
            new_instance.set_yield(yield_)
        return new_instance

    def as_func(self, norm_range: ztyping.LimitsType = False):
        """Return a `Function` with the function `model(x, norm_range=norm_range)`.

        Args:
            norm_range:
        """
        from .operations import convert_pdf_to_func  # prevent circular import

        return convert_pdf_to_func(pdf=self, norm_range=norm_range)

    def __str__(self):
        return ("zfit.model.{type_name}("
                "\"{self_name}\""
                ", dtype={dtype})".format(
            type_name=type(self).__name__,
            self_name=self.name,
            dtype=self.dtype.name))
