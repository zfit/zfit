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
>>>     def _unnormalized_prob(self, x):
>>>         return tf.exp((x - mean) ** 2 / (2 * stddev**2))

Notice that *here* we only specify the *function* and no normalization. This
**No** attempt to **explicitly** normalize the function should be done inside `_unnormalized_prob`.
The normalization is handled with another method depending on the normalization range specified.
(It *is* possible, though discouraged, to directly provide the *normalized probability* by overriding _prob(), but
there are other, more convenient ways to add improvements like providing an analytical integrals.)

Before we create an instance, we need to create the variables to initialize it
>>> mean = zfit.FitParameter("mean1", 2., 0.1, 4.2)  # signature as in RooFit: *name, initial, lower, upper*
>>> stddev = zfit.FitParameter("stddev1", 5., 0.3, 10.)
Let's create an instance and some example data
>>> gauss = MyGauss(mean=mean, stddev=stddev)
>>> example_data = np.random.random(10)
Now we can get the probability
>>> probs = gauss.prob(x=example_data, norm_range=(-30., 30))  # `norm_range` specifies over which range to normalize
Or the integral
>>> integral = gauss.integrate(limits=(-5, 3.1), norm_range=False)  # norm_range is False -> return unnormalized
integral
Or directly sample from it
>>> sample = gauss.sample(n_draws=1000, limits=(-10, 10))  # draw 1000 samples within (-10, 10)

We can create an extended PDF, which will result in anything using a `norm_range` to not return the
probability but the number probability (the function will be normalized to `yield` instead of 1 inside
the `norm_range`)
>>> yield1 = FitParameter("yield1", 100, 0, 1000)
>>> gauss.set_yield(yield1)
>>> gauss.is_extended
True

>>> integral_extended = gauss.integrate(limits=(-10, 10), norm_range=(-10, 10))  # yields approx 100

For more advanced methods and ways to register analytic integrals or overwrite certain methods, see
also the advanced tutorials in `zfit tutorials <https://github.com/zfit/zfit-tutorials>`_
"""

import abc
import builtins
from collections import OrderedDict
import contextlib
from contextlib import suppress
import typing
from typing import Union
import warnings

import tensorflow as tf
import tensorflow_probability.python.mcmc as mc
import pep487

from zfit.core.limits import Range, convert_to_range, no_norm_range, no_multiple_limits, supports
from zfit.util import ztyping
from zfit.util.exception import NormRangeNotImplementedError, MultipleLimitsNotImplementedError, BasePDFSubclassingError
from ..settings import types as ztypes
from . import integrate as zintegrate
from . import sample as zsample
from .parameter import FitParameter
from ..util import exception as zexception
from ..util import container as zcontainer
from zfit import ztf

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
        name = func.__name__
        _BasePDF_USER_IMPL_METHODS_TO_CHECK[name] = has_support
        func.__wrapped__ = _BasePDF_register_check_support
        return func

    return register


class BasePDF(pep487.ABC):  # __init_subclass__ backport
    """Base class for any generic pdf.

    # TODO instructions on how to use

    """
    _DEFAULTS_integration = zcontainer.DotDict()
    _DEFAULTS_integration.mc_sampler = mc.sample_halton_sequence
    _DEFAULTS_integration.draws_per_dim = 4000
    _DEFAULTS_integration.auto_numeric_integrator = zintegrate.auto_integrate

    _analytic_integral = None
    _inverse_analytic_integral = None
    _additional_repr = None

    def __init__(self, dtype: typing.Type = ztypes.float, name: str = "BaseDistribution",
                 reparameterization_type: bool = False,
                 validate_args: bool = False,
                 allow_nan_stats: bool = True, graph_parents: tf.Graph = None, **parameters: typing.Any):
        """The base pdf to inherit from and overwrite `_unnormalized_prob`.

        Args:
            dtype (typing.Type): the dtype of the pdf
            name (str): the name of the pdf
            reparameterization_type (): currently not used, but for forward compatibility
            validate_args (): currently not used, but for forward compatibility
            allow_nan_stats (): currently not used, but for forward compatibility
            graph_parents (): currently not used, but for forward compatibility
            **parameters (): the parameters the distribution depends on
        """
        self._dtype = dtype
        self._reparameterization_type = reparameterization_type
        self._allow_nan_stats = allow_nan_stats
        self._validate_args = validate_args
        self._parameters = parameters or {}
        self._graph_parents = [] if graph_parents is None else graph_parents
        self._name = name

        self.n_dims = None
        self._yield = None
        self._temp_yield = None
        self._norm_range = None
        self._integration = zcontainer.DotDict()
        self._integration.mc_sampler = self._DEFAULTS_integration.mc_sampler
        self._integration.draws_per_dim = self._DEFAULTS_integration.draws_per_dim
        self._integration.auto_numeric_integrator = self._DEFAULTS_integration.auto_numeric_integrator
        self._normalization_value = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        # check if subclass has decorator if required
        for method_name, has_support in _BasePDF_USER_IMPL_METHODS_TO_CHECK.items():
            method = getattr(cls, method_name)
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == _BasePDF_register_check_support:
                    continue  # not overwritten, fine

            # here means: overwritten
            if hasattr(method, "__wrapped__"):
                if method.__wrapped__ == supports:
                    if has_support:
                        continue  # needs support, has been wrapped
                    else:
                        raise BasePDFSubclassingError("Method {} has been wrapped with supports "
                                                      "but is not allowed to. Has to handle all "
                                                      "arguments.".format(method_name))
                elif has_support:
                    raise BasePDFSubclassingError("Method {} has been overwritten and *has to* be "
                                                  "wrapped by `supports` decorator (don't forget () )"
                                                  "to call the decorator as it takes arguments"
                                                  "".format(method_name))
                elif not has_support:
                    continue  # no support, has not been wrapped with
            else:
                if not has_support:
                    continue  # not wrapped, no support, need no

            # if we reach this points, somethings wrong
            raise BasePDFSubclassingError("Method {} has not been correctly wrapped with @supports "
                                          "OR been been wrapped but it should not be".format(method_name))

        cls._analytic_integral = zintegrate.AnalyticIntegral()
        cls._inverse_analytic_integral = []
        cls._additional_repr = {}

    @property
    def name(self):
        """Name prepended to all ops created by this `pdf`."""
        return self._name

    @property
    def dtype(self):
        """The `DType` of `Tensor`s handled by this `pdf`."""
        return self._dtype

    @property
    def parameters(self):
        """Dictionary of parameters used to instantiate this `pdf`."""
        # Remove "self", "__class__", or other special variables. These can appear
        # if the subclass used:
        # `parameters = dict(locals())`.
        return dict((k, v) for k, v in self._parameters.items()
                    if not k.startswith("__") and k != "self")

    @contextlib.contextmanager
    def temp_norm_range(self, norm_range: ztyping.LimitsType) -> Union[
        'Range', None]:  # TODO: rename to better expression
        """Temporarily set a normalization range for the pdf.

        Args:
            norm_range (): The new normalization range
        """
        old_norm_range = self.norm_range
        self.set_norm_range(norm_range)
        if self.n_dims and self._norm_range is not None:
            if not self.n_dims == self._norm_range.n_dims:
                raise ValueError("norm_range n_dims {} does not match dist.n_dims {}"
                                 "".format(self._norm_range.n_dims, self.n_dims))
        else:
            self.n_dims = norm_range.n_dims
        try:
            yield self.norm_range  # or None, not needed
        finally:
            self.set_norm_range(old_norm_range)

    @property
    def norm_range(self) -> Union[Range, None]:
        """Return the current normalization range

        Returns:
            Range or None: The current normalization range

        """
        return self._norm_range

    def set_norm_range(self, norm_range: Union[Range, None]):
        """Fix the normalization range to a certain value. Use with caution!

        It is, in general, better to use either the explicit `norm_range` argument when calling
        a function or the `temp_norm_range` context manager to set a normalization range for a
        limited amount of code.

        Args:
            norm_range ():

        """
        self._norm_range = convert_to_range(norm_range, dims=Range.FULL)
        return self

    def _check_input_norm_range(self, norm_range, dims, caller_name="",
                                none_is_error=False) -> typing.Union[Range, bool]:
        """If `norm_range` is None, take `self.norm_range`. Convert to :py:class:`Range`

        Args:
            norm_range (None or Range compatible):
            dims (tuple(int)):
            caller_name (str): name of the calling function. Used for exception message.
            none_is_error (bool): if both `norm_range` and `self.norm_range` are None, the default
                value is `False` (meaning: no range specified-> no normalization to be done). If
                this is set to true, two `None` will raise a Value error.

        Returns:
            Union[Range, False]:

        """
        if norm_range is None:
            if self.norm_range is None:
                if none_is_error:
                    raise ValueError("Normalization range `norm_range` has to be specified, either by"
                                     "\na) explicitly passing it to the function {name}"
                                     "\nb) using the temp_norm_range context manager to temporary set "
                                     "a default normalization range. Currently, both are None/False."
                                     "".format(name=caller_name))
                else:
                    norm_range = False
            else:
                norm_range = self.norm_range
        return convert_to_range(limits=norm_range, dims=dims)

    @property
    def n_dims(self):
        # TODO: improve n_dims setting
        return self._n_dims

    @n_dims.setter
    def n_dims(self, n_dims):
        self._n_dims = n_dims

    @property
    def is_extended(self) -> bool:
        """Flag to tell whether the pdf is extended or not.

        Returns:
            bool:
        """
        return self._yield is not None

    def set_yield(self, value: Union[FitParameter, None]):
        """Make the pdf extended by setting a yield.

        This alters the behavior of `prob` and similar and `integrate` and similar. If there is a
        `norm_range` given, the output of the above functions does not represent a normalized
        probability density function anymore but corresponds to a number probability.

        Args:
            value ():
        """
        self._yield = value

    @contextlib.contextmanager
    def temp_yield(self, value: Union[FitParameter, None]) -> Union[FitParameter, None]:
        """Temporary set (or unset with None) the yield of the pdf.

        Args:
            value ():
        """
        old_yield = self.get_yield()
        self.set_yield(value)
        try:
            yield value
        finally:
            self.set_yield(old_yield)

    def get_yield(self) -> Union[FitParameter, None]:
        """Return the yield (only for extended pdfs).

        Returns:
            FitParameter: the yield of the current pdf or None
        """
        if not self.is_extended:
            raise zexception.ExtendedPDFError("PDF is not extended, cannot get yield.")
        return self._yield

    def apply_yield(self, value: float, norm_range: ztyping.LimitsType = False, log: bool = False) -> float:
        """If a norm_range is given, the value will be multiplied by the yield.

        Args:
            value (numerical):
            norm_range ():
            log (bool):

        Returns:
            numerical
        """
        return self._apply_yield(value=value, norm_range=norm_range, log=log)

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool) -> float:
        if self.is_extended and norm_range is not False:
            if log:
                value += tf.log(self.get_yield())
            else:
                value *= self.get_yield()
        return value

    @property
    def _yield(self):
        """For internal use, the yield or None"""
        return self.parameters.get('yield')

    @_yield.setter
    def _yield(self, value):
        if value is None:
            # unset
            self._parameters.pop('yield', None)  # safely remove if still there
        else:
            self._parameters['yield'] = value

    @abc.abstractmethod
    def _unnormalized_prob(self, x):
        raise NotImplementedError

    def unnormalized_prob(self, x: ztyping.XType, name: str = "unnormalized_prob") -> ztyping.XType:
        """Return the function unnormalized

        Args:
            x (numerical): The values, have to be convertible to a Tensor
            name (str):

        Returns:
            graph: A runnable graph
        """
        return self._hook_unnormalized_prob(x=x, name=name)

    def _hook_unnormalized_prob(self, x, name="_hook_unnormalized_prob"):
        return self._call_unnormalized_prob(x=x, name=name)

    def _call_unnormalized_prob(self, x, name):
        with self._name_scope(name, values=[x]):
            x = tf.convert_to_tensor(x, name="x")
            try:
                return self._unnormalized_prob(x)
            except NotImplementedError as error:
                # yeah... just to be explicit
                raise
            # alternative implementation below
            # with suppress(NotImplementedError):
            #     return self._prob(x, norm_range="TODO")
            # with suppress(NotImplementedError):
            #     return tf.exp(self._log_prob(x=x, norm_range="TODO"))

            # No fallback, if unnormalized_prob is not implemented

    @_BasePDF_register_check_support(False)
    def _prob(self, x, norm_range):
        raise NotImplementedError

    def prob(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None, name: str = "prob") -> ztyping.XType:
        """Probability density/mass function, normalized over `norm_range`.

        Args:
          x (numerical): `float` or `double` `Tensor`.
          norm_range (tuple, Range): Range to normalize over
          name (str): Prepended to names of ops created by this function.

        Returns:
          prob: a `Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name,
                                                  none_is_error=True)
        return self._hook_prob(x, norm_range, name)

    def _hook_prob(self, x, norm_range, name="_hook_prob"):
        probability = self._norm_prob(x=x, norm_range=norm_range, name=name)
        return self.apply_yield(value=probability, norm_range=norm_range)

    def _norm_prob(self, x, norm_range, name='_norm_prob'):
        probability = self._call_prob(x, norm_range, name)
        return probability

    def _call_prob(self, x, norm_range, name):
        with self._name_scope(name, values=[x, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            with suppress(NotImplementedError):
                return self._prob(x, norm_range=norm_range)
            with suppress(NotImplementedError):
                return tf.exp(self._log_prob(x=x, norm_range=norm_range))
            return self._fallback_prob(x=x, norm_range=norm_range)

    def _fallback_prob(self, x, norm_range):
        pdf = self._call_unnormalized_prob(x, name="_call_unnormalized_prob") / self._hook_normalization(
            limits=norm_range)
        return pdf

    @_BasePDF_register_check_support(False)
    def _log_prob(self, x, norm_range):
        raise NotImplementedError

    def log_prob(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None,
                 name: str = "log_prob") -> ztyping.XType:
        """Log probability density/mass function normalized over `norm_range`

        Args:
          x (numerical): `float` or `double` `Tensor`.
          norm_range (tuple, Range): Range to normalize over
          name (str): Prepended to names of ops created by this function.

        Returns:
          log_prob: a `Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)

        return self._hook_log_prob(x, norm_range, name)

    def _hook_log_prob(self, x, norm_range, name):
        log_prob = self._norm_log_prob(x, norm_range, name)

        # apply yield
        try:
            log_prob = self.apply_yield(value=log_prob, norm_range=norm_range, log=True)
        except NotImplementedError:
            log_prob = tf.log(self.apply_yield(value=tf.exp(log_prob, norm_range=norm_range)))
        return log_prob

    def _norm_log_prob(self, x, norm_range, name='_norm_log_prob'):
        log_prob = self._call_log_prob(x, norm_range, name)
        return log_prob

    def _call_log_prob(self, x, norm_range, name):
        with self._name_scope(name, values=[x, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            with suppress(NotImplementedError):
                return self._log_prob(x=x, norm_range=norm_range)
            with suppress(NotImplementedError):
                return tf.log(self._prob(x=x, norm_range=norm_range))
            return self._fallback_log_prob(norm_range, x)

    def _fallback_log_prob(self, norm_range, x):
        return tf.log(self._norm_prob(x=x, norm_range=norm_range))  # TODO: call not normalized?

    @_BasePDF_register_check_support(True)
    def _normalization(self, norm_range):
        raise NotImplementedError

    def normalization(self, limits: ztyping.LimitsType, name: str = "normalization") -> ztyping.XType:
        """Return the normalization of the function (usually the integral over `limits`).

        Args:
            limits (tuple, Range): The limits on where to normalize over
            name (str):

        Returns:
            Tensor: the normalization value
        """
        limits = convert_to_range(limits, dims=Range.FULL)

        return self._hook_normalization(limits=limits, name=name)

    def _hook_normalization(self, limits, name="_hook_normalization"):
        normalization = self._call_normalization(norm_range=limits, name=name)  # no _norm_* needed
        return normalization

    def _call_normalization(self, norm_range, name):
        # TODO: caching? alternative
        with self._name_scope(name, values=[norm_range]):
            with suppress(NotImplementedError):
                return self._normalization(norm_range)
            return self._fallback_normalization(norm_range)

    def _fallback_normalization(self, norm_range):
        # TODO: multi-dim, more complicated range
        normalization_value = self.integrate(limits=norm_range, norm_range=False)
        return normalization_value

    # Integrals
    @_BasePDF_register_check_support(True)
    def _integrate(self, limits, norm_range):
        raise NotImplementedError()

    def integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                  name: str = "integrate") -> ztyping.XType:
        """Integrate the function over `limits` (normalized over `norm_range` if not False).

        Args:
            limits (tuple, Range): the limits to integrate over
            norm_range (tuple, Range): the limits to normalize over or False to integrate the
                unnormalized probability
            name (str):

        Returns:
            Tensor: the integral value
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=Range.FULL)
        return self._hook_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        integral = self._norm_integrate(limits, norm_range, name)

        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _norm_integrate(self, limits, norm_range, name='_norm_integrate'):
        try:
            integral = self._limits_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            unnormalized_integral = self._limits_integrate(limits=limits, norm_range=False, name=name)
            normalization = self._call_normalization(norm_range=limits, name=name)
            integral = unnormalized_integral / normalization
        return integral

    def _limits_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_integrate(limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(integrals, axis=0)
        return integral

    def _call_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with suppress(NotImplementedError):
                return self._integrate(limits=limits, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._analytic_integrate(limits=limits, norm_range=norm_range)  # TODO: use _norm_integrate?
            return self._fallback_integrate(limits=limits, norm_range=norm_range)

    def _fallback_integrate(self, limits, norm_range):
        dims = limits.dims
        max_dims = self._analytic_integral.get_max_dims(limits=limits, dims=dims)

        integral = None
        if frozenset(max_dims) == frozenset(dims):
            with suppress(NotImplementedError):
                integral = self._norm_analytic_integrate(limits=limits, norm_range=norm_range)
        if max_dims and integral is None:  # TODO improve handling of available analytic integrals
            with suppress(NotImplementedError):
                def part_int(x):
                    """Temporary partial integration function."""
                    return self._norm_partial_analytic_integrate(x, limits=limits, norm_range=norm_range)

                integral = self._auto_numeric_integrate(func=part_int, limits=limits)
        if integral is None:
            integral = self._norm_numeric_integrate(limits=limits, norm_range=norm_range)
        return integral

    @classmethod
    def register_analytic_integral(cls, func: typing.Callable, limits: ztyping.LimitsType = None,
                                   dims: ztyping.DimsType = None, priority: int = 50, *,
                                   supports_norm_range: bool = False,
                                   supports_multiple_limits: bool = False) -> None:
        """Register an analytic integral with the class.

        Args:
            func ():
            limits (): |limits_arg_descr|
            dims (tuple(int)):
            priority (int):
            supports_multiple_limits (bool):
            supports_norm_range (bool):

        Returns:

        """
        cls._analytic_integral.register(func=func, dims=dims, limits=limits, supports_norm_range=supports_norm_range,
                                        priority=priority, supports_multiple_limits=supports_multiple_limits)

    @classmethod
    def register_inverse_analytic_integral(cls, func: typing.Callable) -> None:
        """Register an inverse analytical integral, the inverse (unnormalized) cdf.

        Args:
            func ():
        """
        if len(cls._inverse_analytic_integral) > 0:
            cls._inverse_analytic_integral[0] = func
        else:
            cls._inverse_analytic_integral.append(func)

    @_BasePDF_register_check_support(True)
    def _analytic_integrate(self, limits, norm_range):
        raise NotImplementedError

    def analytic_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                           name: str = "analytic_integrate") -> ztyping.XType:
        """Do analytical integration over function and raise Error if not possible.

        Args:
            limits (tuple, Range): the limits to integrate over
            norm_range (tuple, Range, False): the limits to normalize over
            name (str):

        Returns:
            Tensor: the integral value
        Raises:
            NotImplementedError: If no analytical integral is available (for this limits).
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=Range.FULL)
        return self._hook_analytic_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_analytic_integrate(self, limits, norm_range, name="_hook_analytic_integrate"):
        integral = self._norm_analytic_integrate(limits, norm_range, name)

        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _norm_analytic_integrate(self, limits, norm_range, name='_norm_analytic_integrate'):
        try:
            integral = self._limits_analytic_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:

            unnormalized_integral = self._limits_analytic_integrate(limits, norm_range=None,
                                                                    name=name)
            try:
                normalization = self._limits_analytic_integrate(limits=norm_range, norm_range=False, name=name)
            except NotImplementedError:
                raise NormRangeNotImplementedError("Function {} does not support this (or even any)"
                                                   "normalization range 'norm_range'."
                                                   " This usually means,that no analytic integral "
                                                   "is available for this function. Due to rule "
                                                   "safety, an analytical normalization has to "
                                                   "be available and no attempt of numerical "
                                                   "normalization was made.".format(name))
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_analytic_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_analytic_integrate(limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_analytic_integrate(limits=sub_limits, norm_range=norm_range,
                                                               name=name))
            integral = ztf.reduce_sum(integrals, axis=0)
        return integral

    def _call_analytic_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            with suppress(NotImplementedError):
                return self._analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_analytic_integrate(limits=limits, norm_range=norm_range)

    def _fallback_analytic_integrate(self, limits, norm_range):
        return self._analytic_integral.integrate(x=None, limits=limits, dims=limits.dims,
                                                 norm_range=norm_range, params=self.parameters)

    @_BasePDF_register_check_support(True)
    def _numeric_integrate(self, limits, norm_range):
        raise NotImplementedError

    def numeric_integrate(self, limits: ztyping.LimitsType, norm_range: ztyping.LimitsType = None,
                          name: str = "numeric_integrate") -> ztyping.XType:
        """Do numerical integration over the function.

        Args:
            limits (tuple, Range): the limits to integrate over
            norm_range (tuple, Range, False): the limits to normalize over
            name (str):

        Returns:
            Tensor: the integral value

        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=Range.FULL)

        return self._hook_numeric_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_numeric_integrate(self, limits, norm_range, name='_hook_numeric_integrate'):
        integral = self._norm_numeric_integrate(limits, norm_range, name)

        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _norm_numeric_integrate(self, limits, norm_range, name='_norm_numeric_integrate'):
        try:
            integral = self._limits_numeric_integrate(limits, norm_range, name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_numeric_integrate(limits=limits, norm_range=False,
                                                                   name=name)
            normalization = self._limits_numeric_integrate(limits=norm_range, norm_range=False,
                                                           name=name + "_normalization")
            integral = unnormalized_integral / normalization
        return integral

    def _limits_numeric_integrate(self, limits, norm_range, name):
        try:
            integral = self._call_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_numeric_integrate(limits=sub_limits, norm_range=norm_range, name=name))
            integral = ztf.reduce_sum(integrals, axis=0)

        return integral

    def _call_numeric_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            # TODO: anything?
            with suppress(NotImplementedError):
                return self._numeric_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_numeric_integrate(limits=limits, norm_range=norm_range)

    def _fallback_numeric_integrate(self, limits, norm_range):
        integral = self._auto_numeric_integrate(func=self.unnormalized_prob, limits=limits, norm_range=norm_range)

        return integral

    @_BasePDF_register_check_support(True)
    def _partial_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.DimsType,
                          norm_range: ztyping.LimitsType = None,
                          name: str = "partial_integrate") -> ztyping.XType:
        """Partially integrate the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some dims
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all dims
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL,
                                                  caller_name=name)  # TODO: FULL reasonable?
        limits = convert_to_range(limits, dims=dims)  # TODO: don't request dims but check (if not already Range

        return self._hook_partial_integrate(x=x, limits=limits,
                                            norm_range=norm_range, name=name)

    def _hook_partial_integrate(self, x, limits, norm_range, name='_hook_partial_integrate'):
        integral = self._norm_partial_integrate(x, limits, norm_range, name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _norm_partial_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._limits_partial_integrate(x, limits, norm_range, name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_integrate(x=x, limits=limits, norm_range=None, name=name)
            normalization = self._hook_normalization(limits=norm_range)  # TODO: _call_normalization?
            integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_integrate(x=x, limits=limits,
                                                    norm_range=norm_range,
                                                    name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limit in limits.subbounds():
                integrals.append(self._call_partial_integrate(x=x, limits=sub_limit, norm_range=norm_range,
                                                              name=name))
            integral = ztf.reduce_sum(integrals, axis=0)

        return integral

    def _call_partial_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            x = tf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._partial_integrate(x=x, limits=limits, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

            return self._fallback_partial_integrate(x=x, limits=limits,
                                                    norm_range=norm_range)

    def _fallback_partial_integrate(self, x, limits, norm_range):
        max_dims = self._analytic_integral.get_max_dims(limits=limits, dims=limits.dims)
        if max_dims:
            sublimits = limits.subspace(max_dims)

            def part_int(x):  # change to partial integrate max dims?
                """Temporary partial integration function."""
                return self._norm_partial_analytic_integrate(x=x, limits=sublimits, norm_range=norm_range)

            dims = list(set(limits.dims) - set(max_dims))
        else:
            part_int = self._unnormalized_prob
            dims = limits.dims

        if norm_range is False:
            integral_vals = self._auto_numeric_integrate(func=part_int, limits=limits, dims=dims, x=x)
        else:
            raise NormRangeNotImplementedError
        return integral_vals

    @_BasePDF_register_check_support(True)
    def _partial_analytic_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_analytic_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.DimsType,
                                   norm_range: ztyping.LimitsType = None,
                                   name: str = "partial_analytic_integrate") -> ztyping.XType:
        """Do analytical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some dims
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all dims
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.

        Raises:
            NotImplementedError: if the *analytic* integral (over this limits) is not implemented
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available, explicitly: the **analytical**
                integral over the limits = norm_range is not available.

        """
        norm_range = self._check_input_norm_range(norm_range=norm_range, dims=Range.FULL,
                                                  caller_name=name)  # TODO: full reasonable?
        limits = convert_to_range(limits, dims=dims)  # TODO: replace by limits.dims if dims is None?
        return self._hook_partial_analytic_integrate(x=x, limits=limits,
                                                     norm_range=norm_range, name=name)

    def _hook_partial_analytic_integrate(self, x, limits, norm_range,
                                         name='_hook_partial_analytic_integrate'):

        integral = self._norm_partial_analytic_integrate(x, limits, norm_range, name)

        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _norm_partial_analytic_integrate(self, x, limits, norm_range, name='_norm_partial_analytic_integrate'):
        try:
            integral = self._limits_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_analytic_integrate(x=x, limits=limits,
                                                                            norm_range=False,
                                                                            name=name)
            try:
                normalization = self._limits_analytic_integrate(limits=norm_range, norm_range=False, name=name)
            except NotImplementedError:
                raise NormRangeNotImplementedError("Function {} does not support this (or even any)"
                                                   "normalization range 'norm_range'."
                                                   " This usually means,that no analytic integral "
                                                   "is available for this function. Due to rule "
                                                   "safety, an analytical normalization has to "
                                                   "be available and no attempt of numerical "
                                                   "normalization was made.".format(name))
            else:
                integral = unnormalized_integral / normalization
        return integral

    def _limits_partial_analytic_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_analytic_integrate(x=x, limits=limits,
                                                             norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_partial_analytic_integrate(x=x, limits=sub_limits, norm_range=norm_range,
                                                                       name=name))
            integral = ztf.reduce_sum(integrals, axis=0)

        return integral

    def _call_partial_analytic_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            x = tf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits,
                                                        norm_range=norm_range)
            return self._fallback_partial_analytic_integrate(x=x, limits=limits,
                                                             norm_range=norm_range)

    def _fallback_partial_analytic_integrate(self, x, limits, norm_range):
        return self._analytic_integral.integrate(x=x, limits=limits, dims=limits.dims,
                                                 norm_range=norm_range, params=self.parameters)

    @_BasePDF_register_check_support(True)
    def _partial_numeric_integrate(self, x, limits, norm_range):
        raise NotImplementedError

    def partial_numeric_integrate(self, x: ztyping.XType, limits: ztyping.LimitsType, dims: ztyping.DimsType,
                                  norm_range: ztyping.LimitsType = None,
                                  name: str = "partial_numeric_integrate") -> ztyping.XType:
        """Force numerical partial integration of the function over the `limits` and evaluate it at `x`.

        Dimension of `limits` and `x` have to add up to the full dimension and be therefore equal
        to the dimensions of `norm_range` (if not False)

        Args:
            x (numerical): The values at which the partially integrated function will be evaluated
            limits (tuple, Range): the limits to integrate over. Can contain only some dims
            dims (tuple(int): The dimensions to partially integrate over
            norm_range (tuple, Range, False): the limits to normalize over. Has to have all dims
            name (str):

        Returns:
            Tensor: the values of the partially integrated function evaluated at `x`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        limits = convert_to_range(limits, dims=dims)

        return self._hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)

    def _hook_partial_numeric_integrate(self, x, limits, norm_range,
                                        name='_hook_partial_numeric_integrate'):
        integral = self._norm_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _norm_partial_numeric_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._limits_partial_numeric_integrate(x=x, limits=limits,
                                                              norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the caught Error should not be raised."
            unnormalized_integral = self._limits_partial_numeric_integrate(x=x, limits=limits,
                                                                           norm_range=None, name=name)
            integral = unnormalized_integral / self._hook_normalization(limits=norm_range)
        return integral

    def _limits_partial_numeric_integrate(self, x, limits, norm_range, name):
        try:
            integral = self._call_partial_numeric_integrate(x=x, limits=limits,
                                                            norm_range=norm_range, name=name)
        except MultipleLimitsNotImplementedError:
            integrals = []
            for sub_limits in limits.subbounds():
                integrals.append(self._call_partial_numeric_integrate(x=x, limits=sub_limits, norm_range=norm_range,
                                                                      name=name))
            integral = ztf.reduce_sum(integrals, axis=0)
        return integral

    def _call_partial_numeric_integrate(self, x, limits, norm_range, name):
        with self._name_scope(name, values=[x, limits, norm_range]):
            x = tf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._partial_numeric_integrate(x=x, limits=limits,
                                                       norm_range=norm_range)
            return self._fallback_partial_numeric_integrate(x=x, limits=limits,
                                                            norm_range=norm_range)

    @no_norm_range
    def _fallback_partial_numeric_integrate(self, x, limits, norm_range=False):
        return self._auto_numeric_integrate(func=self.unnormalized_prob, limits=limits, x=x)

    def _auto_numeric_integrate(self, func, limits, x=None, norm_range=False, **overwrite_options):
        integration_options = dict(func=func, limits=limits, n_dims=limits.n_dims, x=x, norm_range=norm_range,
                                   # auto from self
                                   dtype=self.dtype,
                                   mc_sampler=self._integration.mc_sampler,
                                   mc_options={
                                       "draws_per_dim": self._integration.draws_per_dim},
                                   **overwrite_options)
        return self._integration.auto_numeric_integrator(**integration_options)

    @no_norm_range
    def _inverse_analytic_integrate(self, x):
        if self._inverse_analytic_integral is None:
            raise NotImplementedError
        else:
            return self._inverse_analytic_integral[0](x=x, params=self.parameters)

    @_BasePDF_register_check_support(True)
    def _sample(self, n_draws, limits):
        raise NotImplementedError

    def sample(self, n_draws: int, limits: ztyping.LimitsType, name: str = "sample") -> ztyping.XType:
        """Sample `n_draws` within `limits` from the pdf.

        Args:
            n_draws (int): The number of samples to be generated
            limits (tuple, Range): In which region to sample in
            name (str):

        Returns:
            Tensor(n_dims, n_samples)
        """
        limits = convert_to_range(limits, dims=Range.FULL)
        return self._hook_sample(n_draws=n_draws, limits=limits, name=name)

    def _hook_sample(self, limits, n_draws, name='_hook_sample'):
        return self._norm_sample(n_draws=n_draws, limits=limits, name=name)

    def _norm_sample(self, n_draws, limits, name):
        """Dummy function"""
        return self._limits_sample(n_draws=n_draws, limits=limits, name=name)

    def _limits_sample(self, n_draws, limits, name):
        try:
            return self._call_sample(n_draws=n_draws, limits=limits, name=name)
        except MultipleLimitsNotImplementedError:
            raise NotImplementedError("MultipleLimits auto handling in sample currently not supported.")

    def _call_sample(self, n_draws, limits, name):
        with self._name_scope(name, values=[n_draws, limits]):
            n_draws = tf.convert_to_tensor(n_draws, name="n_draws")

            with suppress(NotImplementedError):
                return self._sample(n_draws=n_draws, limits=limits)
            with suppress(NotImplementedError):
                return self._analytic_sample(n_draws=n_draws, limits=limits)
            return self._fallback_sample(n_draws=n_draws, limits=limits)

    def _analytic_sample(self, n_draws, limits: Range):
        if len(limits) > 1:
            raise MultipleLimitsNotImplementedError()

        for lower_bound, upper_bound in zip(*limits.get_boundaries()):
            neg_infinities = (tuple((-float("inf"),) * limits.n_dims),)  # py34 change float("inf") to math.inf
            try:
                lower_prob_lim = self._norm_analytic_integrate(limits=Range.from_boundaries(lower=neg_infinities,
                                                                                            upper=lower_bound,
                                                                                            dims=limits.dims,
                                                                                            convert_none=True),
                                                               norm_range=False)

                upper_prob_lim = self._norm_analytic_integrate(limits=Range.from_boundaries(lower=neg_infinities,
                                                                                            upper=upper_bound,
                                                                                            dims=limits.dims,
                                                                                            convert_none=True),
                                                               norm_range=False)
            except NotImplementedError:
                raise NotImplementedError("analytic sampling not possible because the analytic integral is not"
                                          "implemented for the boundaries:".format(limits.get_boundaries()))
            prob_sample = ztf.random_uniform(shape=(n_draws, limits.n_dims), minval=lower_prob_lim,
                                             maxval=upper_prob_lim)
            sample = self._inverse_analytic_integrate(x=prob_sample)
            return sample

    def _fallback_sample(self, n_draws, limits):
        sample = zsample.accept_reject_sample(prob=self._unnormalized_prob,  # no need to normalize
                                              n_draws=n_draws,
                                              limits=limits, prob_max=None)  # None -> auto
        return sample

    def copy(self, **override_parameters_kwargs):
        """Creates a deep copy of the pdf.

        Note: the copy pdf may continue to depend on the original
        initialization arguments.

        Args:
          **override_parameters_kwargs: String/value dictionary of initialization
            arguments to override with new values.

        Returns:
          pdf: A new instance of `type(self)` initialized from the union
            of self.parameters and override_parameters_kwargs, i.e.,
            `dict(self.parameters, **override_parameters_kwargs)`.
        """
        parameters = dict(self.parameters, **override_parameters_kwargs)
        yield_ = parameters.pop('yield', None)
        new_instance = type(self)(**parameters)
        if yield_ is not None:
            new_instance.set_yield(yield_)
        return new_instance

    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""
        with tf.name_scope(self.name):
            with tf.name_scope(name, values=(
                ([] if values is None else values) + self._graph_parents)) as scope:
                yield scope

    def __str__(self):
        return ("tf.distributions.{type_name}("
                "\"{self_name}\""
                "{maybe_batch_shape}"
                "{maybe_event_shape}"
                ", dtype={dtype})".format(
            type_name=type(self).__name__,
            self_name=self.name,
            maybe_batch_shape=(", batch_shape={}".format(self.batch_shape)
                               if self.batch_shape.ndims is not None
                               else ""),
            maybe_event_shape=(", event_shape={}".format(self.event_shape)
                               if self.event_shape.ndims is not None
                               else ""),
            dtype=self.dtype.name))

    @classmethod
    def register_additional_repr(cls, **kwargs):
        """Register an additional attribute to add to the repr.

        Args:
            any keyword argument. The value has to be gettable from the instance (has to be an
            attribute or callable method of self.
        """
        if cls._additional_repr is None:
            cls._additional_repr = {}
        overwritten_keys = set(kwargs).intersection(cls._additional_repr)
        if overwritten_keys:
            warnings.warn("The following keys have been overwritten while registering additional repr:"
                          "\n{}".format([str(k) for k in overwritten_keys]))
        cls._additional_repr = dict(cls._additional_repr, **kwargs)

    def _get_additional_repr(self, sorted=True):

        # nice name change
        sorted_ = sorted
        sorted = builtins.sorted
        # nice name change end

        additional_repr = OrderedDict()
        for key, val in self._additional_repr.items():
            try:
                new_obj = getattr(self, val)
            except AttributeError:
                raise AttributeError("The attribute {} is not a valid attribute of this class {}."
                                     "Cannot use it in __repr__. It was added using the"
                                     "`register_additional_repr` function".format(val, type(self)))
            else:
                if callable(new_obj):
                    new_obj = new_obj()
            additional_repr[key] = new_obj
        if sorted_:
            additional_repr = sorted(additional_repr)
        return additional_repr

    def __repr__(self):

        return ("<zfit.pdf.{type_name} "
                # "'{self_name}'"
                " parameters=[{params}]"
                " dtype={dtype}>".format(
            type_name=type(self).__name__,
            # self_name=self.name,
            params=", ".join(sorted(str(p.name) for p in self.parameters.values())),
            dtype=self.dtype.name) +
                sum(" {k}={v}".format(k=str(k), v=str(v)) for k, v in self._get_additional_repr(sorted=True).items()))

    def __eq__(self, other):
        if not type(self) == type(other):
            raise TypeError("Cannot compare objects of type {} and {}".format(type(self), type(other)))
        params_equal = set(other.parameters) == set(self.parameters)
        return params_equal
