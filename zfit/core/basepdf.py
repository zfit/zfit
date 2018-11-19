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
>>> mean = zfit.FitParameter("mean1", 2., 0.1, 4.2)  # signature as in RooFit: *name, initial, lower, upper*
>>> stddev = zfit.FitParameter("stddev1", 5., 0.3, 10.)
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
>>> gauss.set_yield(yield1)
>>> gauss.is_extended
True

>>> integral_extended = gauss.integrate(limits=(-10, 10), norm_range=(-10, 10))  # yields approx 100

For more advanced methods and ways to register analytic integrals or overwrite certain methods, see
also the advanced tutorials in `zfit tutorials <https://github.com/zfit/zfit-tutorials>`_
"""

import abc
import contextlib
from contextlib import suppress
import typing
from typing import Union
import warnings

import tensorflow as tf

from .basemodel import BaseModel, _BaseModel_USER_IMPL_METHODS_TO_CHECK
from zfit.core.limits import Range, convert_to_range
from zfit.util import ztyping
from ..settings import types as ztypes

from .parameter import Parameter
from ..util import exception as zexception
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

class BasePDF(BaseModel):

    def __init__(self, dtype: typing.Type = ztypes.float, name: str = "BasePDF",
                 reparameterization_type: bool = False,
                 validate_args: bool = False,
                 allow_nan_stats: bool = True, graph_parents: tf.Graph = None, **parameters: typing.Any):
        super().__init__(dtype=dtype, name=name, reparameterization_type=reparameterization_type,
                         validate_args=validate_args,
                         allow_nan_stats=allow_nan_stats, graph_parents=graph_parents, **parameters)

        self.n_dims = None
        self._yield = None
        self._temp_yield = None
        self._norm_range = None
        self._normalization_value = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclass_check_support(methods_to_check=_BasePDF_USER_IMPL_METHODS_TO_CHECK,
                                    wrapper_not_overwritten=_BasePDF_register_check_support)

    def copy(self, **override_parameters):
        """Creates a deep copy of the pdf.

        Note: the copy pdf may continue to depend on the original
        initialization arguments.

        Args:
          **override_parameters: String/value dictionary of initialization
            arguments to override with new values.

        Returns:
          pdf: A new instance of `type(self)` initialized from the union
            of self.parameters and override_parameters_kwargs, i.e.,
            `dict(self.parameters, **override_parameters_kwargs)`.
        """
        parameters = dict(self.parameters, **override_parameters)
        yield_ = parameters.pop('yield', None)
        new_instance = type(self)(**parameters)
        if yield_ is not None:
            new_instance.set_yield(yield_)
        return new_instance

    def _func_to_integrate(self, x: ztyping.XType):
        return self.unnormalized_pdf(x)

    def _func_to_sample_from(self, x):
        return self.unnormalized_pdf(x)

    def _hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        integral = self._norm_integrate(limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _hook_analytic_integrate(self, limits, norm_range, name="_hook_analytic_integrate"):
        integral = super()._hook_analytic_integrate(limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _hook_numeric_integrate(self, limits, norm_range, name='_hook_numeric_integrate'):
        integral = super()._hook_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _hook_partial_integrate(self, x, limits, norm_range, name='_hook_partial_integrate'):
        integral = super()._hook_partial_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _hook_partial_analytic_integrate(self, x, limits, norm_range,
                                         name='_hook_partial_analytic_integrate'):

        integral = super()._hook_partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

    def _hook_partial_numeric_integrate(self, x, limits, norm_range, name='_hook_partial_numeric_integrate'):
        integral = super()._hook_partial_numeric_integrate(x=x, limits=limits, norm_range=norm_range, name=name)
        integral = self.apply_yield(integral, norm_range=norm_range)
        return integral

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

    @_BasePDF_register_check_support(True)
    def _normalization(self, norm_range):
        raise NotImplementedError

    def _hook_normalization(self, limits, name="_hook_normalization"):
        normalization = self._call_normalization(norm_range=limits, name=name)  # no _norm_* needed
        return normalization

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

    def _fallback_normalization(self, norm_range):
        # TODO: multi-dim, more complicated range
        normalization_value = self.integrate(limits=norm_range, norm_range=False)
        return normalization_value

    def _call_normalization(self, norm_range, name):
        # TODO: caching? alternative
        with self._name_scope(name, values=[norm_range]):
            with suppress(NotImplementedError):
                return self._normalization(norm_range)
            return self._fallback_normalization(norm_range)

    @abc.abstractmethod
    def _unnormalized_pdf(self, x, norm_range=False):
        raise NotImplementedError

    def _call_unnormalized_pdf(self, x, name):
        with self._name_scope(name, values=[x]):
            x = ztf.convert_to_tensor(x, name="x")

            try:
                return self._unnormalized_pdf(x)
            except NotImplementedError as error:
                # yeah... just to be explicit
                raise
            # alternative implementation below
            # with suppress(NotImplementedError):
            #     return self._pdf(x, norm_range="TODO")
            # with suppress(NotImplementedError):
            #     return tf.exp(self._log_pdf(x=x, norm_range="TODO"))

            # No fallback, if unnormalized_pdf is not implemented

    def _hook_unnormalized_pdf(self, x, name="_hook_unnormalized_pdf"):
        return self._call_unnormalized_pdf(x=x, name=name)

    def unnormalized_pdf(self, x: ztyping.XType, name: str = "unnormalized_pdf") -> ztyping.XType:
        """Return the function unnormalized

        Args:
            x (numerical): The values, have to be convertible to a Tensor
            name (str):

        Returns:
            graph: A runnable graph
        """
        return self._hook_unnormalized_pdf(x=x, name=name)

    @_BasePDF_register_check_support(False)
    def _pdf(self, x, norm_range):
        raise NotImplementedError

    def pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None, name: str = "pdf") -> ztyping.XType:
        """Probability density/mass function, normalized over `norm_range`.

        Args:
          x (numerical): `float` or `double` `Tensor`.
          norm_range (tuple, Range): Range to normalize over
          name (str): Prepended to names of ops created by this function.

        Returns:
          pdf: a `Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name,
                                                  none_is_error=True)
        return self._hook_pdf(x, norm_range, name)

    def _hook_pdf(self, x, norm_range, name="_hook_pdf"):
        probability = self._norm_pdf(x=x, norm_range=norm_range, name=name)
        return self.apply_yield(value=probability, norm_range=norm_range)

    def _norm_pdf(self, x, norm_range, name='_norm_pdf'):
        probability = self._call_pdf(x, norm_range, name)
        return probability

    def _call_pdf(self, x, norm_range, name):
        with self._name_scope(name, values=[x, norm_range]):
            x = ztf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._pdf(x, norm_range=norm_range)
            with suppress(NotImplementedError):
                return tf.exp(self._log_pdf(x=x, norm_range=norm_range))
            return self._fallback_pdf(x=x, norm_range=norm_range)

    def _fallback_pdf(self, x, norm_range):
        pdf = self._call_unnormalized_pdf(x, name="_call_unnormalized_pdf") / self._hook_normalization(
            limits=norm_range)
        return pdf

    @_BasePDF_register_check_support(False)
    def _log_pdf(self, x, norm_range):
        raise NotImplementedError

    def _hook_log_pdf(self, x, norm_range, name):
        log_prob = self._norm_log_pdf(x, norm_range, name)

        # apply yield
        try:
            log_prob = self.apply_yield(value=log_prob, norm_range=norm_range, log=True)
        except NotImplementedError:
            log_prob = tf.log(self.apply_yield(value=tf.exp(log_prob, norm_range=norm_range)))
        return log_prob

    def log_pdf(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None,
                name: str = "log_pdf") -> ztyping.XType:
        """Log probability density/mass function normalized over `norm_range`

        Args:
          x (numerical): `float` or `double` `Tensor`.
          norm_range (tuple, Range): Range to normalize over
          name (str): Prepended to names of ops created by this function.

        Returns:
          log_pdf: a `Tensor` of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)

        return self._hook_log_pdf(x, norm_range, name)

    def _norm_log_pdf(self, x, norm_range, name='_norm_log_pdf'):
        log_prob = self._call_log_pdf(x, norm_range, name)
        return log_prob

    def _fallback_log_pdf(self, norm_range, x):
        return tf.log(self._norm_pdf(x=x, norm_range=norm_range))  # TODO: call not normalized?

    def _call_log_pdf(self, x, norm_range, name):
        with self._name_scope(name, values=[x, norm_range]):
            x = ztf.convert_to_tensor(x, name="x")

            with suppress(NotImplementedError):
                return self._log_pdf(x=x, norm_range=norm_range)
            with suppress(NotImplementedError):
                return tf.log(self._pdf(x=x, norm_range=norm_range))
            return self._fallback_log_pdf(norm_range, x)

    def gradient(self, x: ztyping.XType, norm_range: ztyping.LimitsType, params: ztyping.ParamsType = None):
        warnings.warn("Taking the gradient *this way* in TensorFlow is inefficient! Consider taking it with"
                      "respect to the loss function.")
        if params is None:
            params = list(self.parameters.values())

        probs = self.pdf(x, norm_range=norm_range)

        gradients = [tf.gradients(prob, params) for prob in tf.unstack(probs)]
        return tf.stack(gradients)

    def _apply_yield(self, value: float, norm_range: ztyping.LimitsType, log: bool) -> float:
        if self.is_extended and norm_range is not False:
            if log:
                value += tf.log(self.get_yield())
            else:
                value *= self.get_yield()
        return value

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

    def set_yield(self, value: Union[Parameter, None]):
        """Make the pdf extended by setting a yield.

        This alters the behavior of `pdf` and similar and `integrate` and similar. If there is a
        `norm_range` given, the output of the above functions does not represent a normalized
        probability density function anymore but corresponds to a number probability.

        Args:
            value ():
        """
        self._set_yield(value=value)

    def _set_yield(self, value: Union[Parameter, None]):
        self._yield = value

    @contextlib.contextmanager
    def temp_norm_range(self, norm_range: ztyping.LimitsType) -> Union['Range', None]:  # TODO: rename, better?
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
            self.n_dims = self.norm_range.n_dims
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

    @property
    def is_extended(self) -> bool:
        """Flag to tell whether the pdf is extended or not.

        Returns:
            bool:
        """
        return self._yield is not None

    @contextlib.contextmanager
    def temp_yield(self, value: Union[Parameter, None]) -> Union[Parameter, None]:
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

    def get_yield(self) -> Union[Parameter, None]:
        """Return the yield (only for extended pdfs).

        Returns:
            Parameter: the yield of the current pdf or None
        """
        if not self.is_extended:
            raise zexception.ExtendedPDFError("PDF is not extended, cannot get yield.")
        return self._yield

    def as_func(self, norm_range: ztyping.LimitsType = False):
        """Return a `Function` with the function `pdf(x, norm_range=norm_range)`.

        Args:
            norm_range ():
        """
        from .operations import convert_pdf_to_func  # prevent circular import

        return convert_pdf_to_func(pdf=self, norm_range=norm_range)

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
