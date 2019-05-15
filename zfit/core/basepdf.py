"""
This  module defines the `BasePdf` that can be used to inherit from in order to build a custom PDF.

The `BasePDF` implements already a lot of ready-to-use functionality like integral, automatic normalization
and sampling.

Defining your own pdf
---------------------

A simple example:
>>> class MyGauss(BasePDF):
>>>     def __init__(self, mean, stddev, name="MyGauss"):
>>>         super().__init__(mean=mean, stddev=stddev, name=name)
>>>
>>>     def _unnormalized_pdf(self, x):
>>>         return tf.exp((x - mean) ** 2 / (2 * stddev**2))

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
>>> probs = gauss.pdf(x=example_data, norm_range=(-30., 30))  # `norm_range` specifies over which range to normalize
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
also the advanced tutorials in `zfit tutorials <https://github.com/zfit/zfit-tutorials>`_
"""

#  Copyright (c) 2019 zfit

import abc
from contextlib import suppress
from typing import Union, Any, Type, Dict
import warnings

import tensorflow as tf

from zfit import ztf
from zfit.core.sample import extended_sampling
from zfit.util.cache import invalidates_cache
from .interfaces import ZfitPDF, ZfitParameter
from .limits import Space
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (AlreadyExtendedPDFError, DueToLazynessNotImplementedError, IntentionNotUnambiguousError,
                              AlreadyExtendedPDFError,
                              NormRangeNotSpecifiedError, ShapeIncompatibleError, NotExtendedPDFError, )
from ..util.temporary import TemporarilySet
from .basemodel import BaseModel
from .parameter import Parameter, convert_to_parameter
from ..settings import ztypes, run

_BasePDF_USER_IMPL_METHODS_TO_CHECK = {}


def _BasePDF_register_check_support(has_support: bool):
    """Marks a method that the subclass either *has* to or *can't* use the `@supports` decorator.

    Args:
        has_support (bool): If True, flags that it **requires** the `@supports` decorator. If False,
            flags that the `@supports` decorator is **not allowed**.

    """
    if not isinstance(has_support, bool):
        raise TypeError("Has to be boolean.")

    def register(func):
        """Register a method to be checked to (if True) *has* `support` or (if False) has *no* `support`.

        Args:
            func (function):

        Returns:
            function:
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
        self._temp_yield = None
        self._norm_range = None
        self._normalization_value = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclass_check_support(methods_to_check=_BasePDF_USER_IMPL_METHODS_TO_CHECK,
                                    wrapper_not_overwritten=_BasePDF_register_check_support)

    @property
    def space(self) -> "zfit.Space":
        if self._norm_range is not None:
            space = self._norm_range
        else:
            space = super().space

        return space

    def _check_input_norm_range(self, norm_range, caller_name="", none_is_error=True):
        if norm_range is None:
            norm_range = self.norm_range
        return super()._check_input_norm_range(norm_range=norm_range, caller_name=caller_name,
                                               none_is_error=none_is_error)

    def _check_input_params(self, *params):
        return tuple(convert_to_parameter(p) for p in params)

    def _func_to_integrate(self, x: ztyping.XType):
        return self.unnormalized_pdf(x)

    def _func_to_sample_from(self, x):
        return self.unnormalized_pdf(x)

    def _single_hook_integrate(self, limits, norm_range, name='hook_integrate'):
        integral = super()._single_hook_integrate(limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _single_hook_analytic_integrate(self, limits, norm_range, name='hook_analytic_integrate'):
        integral = super()._single_hook_analytic_integrate(limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _single_hook_numeric_integrate(self, limits, norm_range, name='hook_numeric_integrate'):
        numeric_integral = super()._single_hook_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        numeric_integral = self.apply_yield(numeric_integral, norm_range=norm_range)
        return numeric_integral

    def _single_hook_partial_integrate(self, x, limits, norm_range, name='hook_partial_integrate'):
        partial_integral = super()._single_hook_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        partial_integral = self.apply_yield(partial_integral, norm_range=norm_range)
        return partial_integral

    def _single_hook_partial_analytic_integrate(self, x, limits, norm_range, name='hook_partial_analytic_integrate'):
        part_analytic_int = super()._single_hook_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range,
                                                                            name=name)
        part_analytic_int = self.apply_yield(part_analytic_int, norm_range=norm_range)
        return part_analytic_int

    def _single_hook_partial_numeric_integrate(self, x, limits, norm_range, name='hook_partial_numeric_integrate'):
        part_numeric_int = super()._single_hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range,
                                                                          name=name)
        part_numeric_int = self.apply_yield(part_numeric_int, norm_range=norm_range)
        return part_numeric_int

    @property
    def norm_range(self) -> Union[Space, None, bool]:
        """Return the current normalization range. If None and the `obs`have limits, they are returned.

        Returns:
            :py:class:`~zfit.Space` or None: The current normalization range

        """
        norm_range = self._norm_range
        if norm_range is None:
            norm_range = self.space
        return norm_range

    @invalidates_cache
    def set_norm_range(self, norm_range: ztyping.LimitsTypeInput):
        """Set the normalization range (temporarily if used with contextmanager).

        Args:
            norm_range (tuple, :py:class:`~zfit.Space`):

        """
        norm_range = self._check_input_norm_range(norm_range=norm_range)

        def setter(value):
            self._norm_range = value

        def getter():
            return self._norm_range

        return TemporarilySet(value=norm_range, setter=setter, getter=getter)

    @property
    def _yield(self):
        """For internal use, the yield or None"""
        return self.params.get('yield')

    @_yield.setter
    def _yield(self, value):
        if value is None:
            # unset
            self._params.pop('yield', None)  # safely remove if still there
        else:
            self._params['yield'] = value

    @_BasePDF_register_check_support(True)
    def _normalization(self, limits):
        raise NotImplementedError

    def normalization(self, limits: ztyping.LimitsType, name: str = "normalization") -> ztyping.XType:
        """Return the normalization of the function (usually the integral over `limits`).

        Args:
            limits (tuple, :py:class:`~zfit.Space`): The limits on where to normalize over
            name (str):

        Returns:
            Tensor: the normalization value
        """
        limits = self._check_input_limits(limits=limits, caller_name=name)

        return self._single_hook_normalization(limits=limits, name=name)

    def _single_hook_normalization(self, limits, name):  # TODO(Mayou36): add yield?
        return self._hook_normalization(limits=limits, name=name)

    def _hook_normalization(self, limits, name="_hook_normalization"):
        return self._call_normalization(limits=limits, name=name)  # no _norm_* needed

    def _call_normalization(self, limits, name):
        # TODO: caching? alternative
        with self._name_scope(name, values=[limits]):
            with suppress(NotImplementedError):
                return self._normalization(limits=limits)
            return self._fallback_normalization(limits)

    def _fallback_normalization(self, limits):
        return self._hook_integrate(limits=limits, norm_range=False)

    @abc.abstractmethod
    def _unnormalized_pdf(self, x):
        raise NotImplementedError

    def unnormalized_pdf(self, x: ztyping.XType, component_norm_range: ztyping.LimitsTypeInput = None,
                         name: str = "unnormalized_pdf") -> ztyping.XType:
        """PDF "unnormalized". Use `functions` for unnormalized pdfs. this is only for performance in special cases.

        Args:
            x (numerical): The value, have to be convertible to a Tensor
            component_norm_range (:py:class:`~zfit.Space`): The normalization range for the components. Needed for
            certain composition
                pdfs.
            name (str):

        Returns:
            :py:class:`tf.Tensor`: 1-dimensional :py:class:`tf.Tensor` containing the unnormalized pdf.
        """
        # if component_norm_range is None:
        #     component_norm_range = self._get
        with self._convert_sort_x(x) as x:
            component_norm_range = self._check_input_norm_range(component_norm_range, caller_name=name,
                                                                none_is_error=False)
            return self._single_hook_unnormalized_pdf(x, component_norm_range, name)

    def _single_hook_unnormalized_pdf(self, x, component_norm_range, name):
        return self._call_unnormalized_pdf(x=x, name=name)

    def _call_unnormalized_pdf(self, x, name):
        with self._name_scope(name, values=[x]):
            try:
                return self._unnormalized_pdf(x)
            except ValueError as error:
                raise ShapeIncompatibleError("Most probably, the number of obs the pdf was designed for"
                                             "does not coincide with the `n_obs` from the `space`/`obs`"
                                             "it received on initialization."
                                             "Original Error: {}".format(error))

    @_BasePDF_register_check_support(False)
    def _pdf(self, x, norm_range):
        raise NotImplementedError

    def pdf(self, x: ztyping.XTypeInput, norm_range: ztyping.LimitsTypeInput = None,
            name: str = "model") -> ztyping.XType:
        """Probability density function, normalized over `norm_range`.

        Args:
          x (numerical): `float` or `double` `Tensor`.
          norm_range (tuple, :py:class:`~zfit.Space`): :py:class:`~zfit.Space` to normalize over
          name (str): Prepended to names of ops created by this function.

        Returns:
          :py:class:`tf.Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, caller_name=name, none_is_error=True)
        with self._convert_sort_x(x) as x:
            value = self._single_hook_pdf(x=x, norm_range=norm_range, name=name)
            if run.numeric_checks:
                assert_op = ztf.check_numerics(value, message="Check if pdf output contains any NaNs of Infs")
                assert_op = [assert_op]
            else:
                assert_op = []
            with tf.control_dependencies(assert_op):
                return ztf.to_real(value)

    def _single_hook_pdf(self, x, norm_range, name):
        return self._hook_pdf(x=x, norm_range=norm_range, name=name)

    def _hook_pdf(self, x, norm_range, name="_hook_pdf"):
        return self._norm_pdf(x=x, norm_range=norm_range, name=name)

    def _norm_pdf(self, x, norm_range, name='norm_pdf'):
        return self._call_pdf(x=x, norm_range=norm_range, name=name)

    def _call_pdf(self, x, norm_range, name):
        with self._name_scope(name, values=[x, norm_range]):
            with suppress(NotImplementedError):
                return self._pdf(x, norm_range=norm_range)
            with suppress(NotImplementedError):
                return tf.exp(self._log_pdf(x=x, norm_range=norm_range))
            return self._fallback_pdf(x=x, norm_range=norm_range)

    def _fallback_pdf(self, x, norm_range):
        pdf = self._call_unnormalized_pdf(x, name="_call_unnormalized_pdf")
        if norm_range.limits is not False:  # identity check!
            pdf /= self._hook_normalization(limits=norm_range)
        return pdf

    @_BasePDF_register_check_support(False)
    def _log_pdf(self, x, norm_range):
        raise NotImplementedError

    def log_pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None,
                name: str = "log_pdf") -> ztyping.XType:
        """Log probability density function normalized over `norm_range`.

        Args:
          x (numerical): `float` or `double` `Tensor`.
          norm_range (tuple, :py:class:`~zfit.Space`): :py:class:`~zfit.Space` to normalize over
          name (str): Prepended to names of ops created by this function.

        Returns:
          log_pdf: a `Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, caller_name=name)
        with self._convert_sort_x(x) as x:
            return self._single_hook_log_pdf(x=x, norm_range=norm_range, name=name)

    def _single_hook_log_pdf(self, x, norm_range, name):
        return self._hook_log_pdf(x=x, norm_range=norm_range, name=name)

    def _hook_log_pdf(self, x, norm_range, name):
        log_prob = self._norm_log_pdf(x=x, norm_range=norm_range, name=name)
        return log_prob

    def _norm_log_pdf(self, x, norm_range, name='norm_log_pdf'):
        return self._call_log_pdf(x=x, norm_range=norm_range, name=name)

    def _call_log_pdf(self, x, norm_range, name):
        with self._name_scope(name, values=[x, norm_range]):
            with suppress(NotImplementedError):
                return self._log_pdf(x=x, norm_range=norm_range)
            with suppress(NotImplementedError):
                return tf.log(self._pdf(x=x, norm_range=norm_range))
            return self._fallback_log_pdf(x=x, norm_range=norm_range)

    def _fallback_log_pdf(self, x, norm_range):
        return tf.log(self._hook_pdf(x=x, norm_range=norm_range))

    def gradients(self, x: ztyping.XType, norm_range: ztyping.LimitsType, params: ztyping.ParamsTypeOpt = None):
        warnings.warn("Taking the gradient *this way* in TensorFlow is inefficient! Consider taking it with"
                      "respect to the loss function.")
        if params is not None:
            params = convert_to_container(params)
        if params is None or isinstance(params[0], str):
            params = self.get_params(only_floating=False, names=params)

        probs = self.pdf(x, norm_range=norm_range)
        gradients = [tf.gradients(prob, params) for prob in ztf.unstack_x(probs, always_list=True)]
        return tf.stack(gradients)

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool) -> Union[float, tf.Tensor]:
        if self.is_extended and norm_range.limits is not False:
            if log:
                value += tf.log(self.get_yield())
            else:
                value *= self.get_yield()
        return value

    def apply_yield(self, value: Union[float, tf.Tensor], norm_range: ztyping.LimitsTypeInput = False,
                    log: bool = False) -> Union[float, tf.Tensor]:
        """If a norm_range is given, the value will be multiplied by the yield.

        Args:
            value (numerical):
            norm_range ():
            log (bool):

        Returns:
            numerical
        """
        norm_range = self._check_input_norm_range(norm_range=norm_range)
        return self._apply_yield(value=value, norm_range=norm_range, log=log)

    @invalidates_cache
    def _set_yield_inplace(self, value: Union[Parameter, float, None]):
        """Make the model extended by (temporarily) setting a yield.

        This alters the behavior of `model` and similar and `integrate` and similar. If there is a
        `norm_range` given, the output of the above functions does not represent a normalized
        probability density function anymore but corresponds to a number probability.

        Args:
            value ():
        """

        # TODO(Mayou36): check input for yield?
        def setter(value):
            self._set_yield(value=value)

        def getter():
            return self.get_yield()

        return TemporarilySet(value=value, setter=setter, getter=getter)

    def create_extended(self, yield_: ztyping.ParamTypeInput, name_addition="_extended") -> "ZfitPDF":
        """Return an extended version of this pdf with yield `yield_`. The parameters are shared.

        Args:
            yield_ (numeric, :py:class:`~zfit.Parameter`):
            name_addition (str):

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
        new_pdf._set_yield_inplace(value=yield_)
        return new_pdf

    def _set_yield(self, value: Union[Parameter, None]):
        if value is not None:
            value = convert_to_parameter(value)
        self._yield = value

    @property
    def is_extended(self) -> bool:
        """Flag to tell whether the model is extended or not.

        Returns:
            bool:
        """
        return self._yield is not None

    def _hook_sample(self, limits, n, name='hook_sample'):
        if n is None and self.is_extended:
            n = 'extended'
        if n == 'extended':
            if not self.is_extended:
                raise NotExtendedPDFError("Cannot use 'extended' as value for `n` on a non-extended pdf.")
            samples = extended_sampling(pdfs=self, limits=limits)
        elif isinstance(n, str):
            raise ValueError("`n` is a string and not 'extended'. Other options are currently not implemented.")
        elif n is None:
            raise tf.errors.InvalidArgumentError("`n` cannot be `None` if pdf is not extended.")
        else:
            samples = super()._hook_sample(limits=limits, n=n, name=name)
        return samples

    def get_yield(self) -> Union[Parameter, None]:
        """Return the yield (only for extended models).

        Returns:
            :py:class:`~zfit.Parameter`: the yield of the current model or None
        """
        # if not self.is_extended:
        #     raise zexception.ExtendedPDFError("PDF is not extended, cannot get yield.")
        return self._yield

    def create_projection_pdf(self, limits_to_integrate: ztyping.LimitsTypeInput) -> 'ZfitPDF':
        """Create a PDF projection by integrating out some of the dimensions.

        The new projection pdf is still fully dependent on the pdf it was created with.

        Args:
            limits_to_integrate (:py:class:`~zfit.Space`):

        Returns:
            ZfitPDF: a pdf without the dimensions from `limits_to_integrate`.
        """
        from ..models.special import SimpleFunctorPDF

        def partial_integrate_wrapped(self_simple, x):
            norm_range = self_simple._get_component_norm_range()
            if norm_range not in (None, False) and norm_range.limits not in (None, False):
                from zfit.models.functor import BaseFunctor

                if isinstance(self, BaseFunctor):
                    self._set_component_norm_range(norm_range)
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
          model: A new instance of `type(self)` initialized from the union
            of self.parameters and override_parameters, i.e.,
            `dict(self.parameters, **override_parameters)`.
        """
        obs = self.norm_range
        # if obs.limits is None:
        #     obs = self.space

        # HACK(Mayou36): remove once copy is proper implemented
        from ..models.dist_tfp import WrapDistribution

        if type(self) == WrapDistribution:  # NOT isinstance! Because e.g. Gauss wraps that and takes different args
            parameters = dict(distribution=self._distribution, dist_params=self.dist_params)
        else:
            # HACK END

            parameters = dict(self.params)
            lambda_ = parameters.pop('lambda', None)
            if lambda_ is not None:
                parameters['lambda_'] = lambda_
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
            new_instance._set_yield_inplace(yield_)
        return new_instance

    def as_func(self, norm_range: ztyping.LimitsType = False):
        """Return a `Function` with the function `model(x, norm_range=norm_range)`.

        Args:
            norm_range ():
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
