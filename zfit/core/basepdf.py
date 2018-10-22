"""
Definition of the pdf interface, base etc.
"""
from __future__ import print_function, division, absolute_import

import contextlib
from contextlib import suppress
import typing

import tensorflow as tf
import tensorflow_probability.python.mcmc as mc
from zfit.core import limits

from zfit.core.limits import Range, convert_to_range, no_norm_range
from zfit.util.exception import NormRangeNotImplementedError
# import zfit.core.integrate
from ..settings import types as ztypes
from . import integrate as zintegrate
from . import sample as zsample
# from zfit.settings import types as ztypes
from ..util import exception as zexception
from ..util import container as zcontainer


class AbstractBasePDF(object):

    def sample(self, sample_shape=(), seed=None, name='sample'):
        raise NotImplementedError

    def unnormalized_prob(self, x, name='unnormalized_prob'):
        raise NotImplementedError

    def log_prob(self, x, name='log_prob'):
        raise NotImplementedError

    def integrate(self, limits, name='integrate'):
        self.error = NotImplementedError
        raise self.error

    def batch_shape_tensor(self, name='batch_shape_tensor'):
        raise NotImplementedError

    def event_shape_tensor(self, name='event_shape_tensor'):
        raise NotImplementedError


# class BasePDF(tf.distributions.Distribution, AbstractBasePDF):
class BasePDF(AbstractBasePDF):
    _DEFAULTS_integration = zcontainer.dotdict()
    _DEFAULTS_integration.mc_sampler = mc.sample_halton_sequence
    _DEFAULTS_integration.draws_per_dim = 4000
    _DEFAULTS_integration.auto_numeric_integrate = zintegrate.auto_integrate

    _analytic_integral = zintegrate.AnalyticIntegral()

    def __init__(self, dtype=ztypes.float, name="BaseDistribution", reparameterization_type=False, validate_args=False,
                 allow_nan_stats=True, graph_parents=None, **parameters):
        # TODO: catch some args from kwargs that belong to the super init?
        # super(BasePDF, self).__init__(dtype=ztypes.float,
        #                               reparameterization_type=False,
        #                               validate_args=True, parameters=kwargs,
        #                               allow_nan_stats=False, name=name)

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
        self._integration = zcontainer.dotdict()
        self._integration.mc_sampler = self._DEFAULTS_integration.mc_sampler
        self._integration.draws_per_dim = self._DEFAULTS_integration.draws_per_dim
        self._integration.auto_numeric_integrate = self._DEFAULTS_integration.auto_numeric_integrate
        self._normalization_value = None

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
    def temp_norm_range(self, norm_range):  # TODO: rename to better expression
        old_norm_range = self.norm_range
        self.set_norm_range(norm_range)
        if self.n_dims:
            if not self.n_dims == self._norm_range.n_dims:
                raise ValueError("norm_range n_dims {} does not match dist.n_dims {}"
                                 "".format(self._norm_range.n_dims, self.n_dims))
        else:
            self.n_dims = self.n_dims_from_limits(norm_range)
        try:
            yield self.norm_range  # or None, not needed
        finally:
            self.set_norm_range(old_norm_range)

    @property
    def norm_range(self):
        return self._norm_range

    def set_norm_range(self, norm_range):
        self._norm_range = convert_to_range(norm_range, dims=Range.FULL)

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
                                     "\nb) using the temp_norm_range context manager to temprarely set "
                                     "a default normalization range. Currently, both are None/False."
                                     "".format(name=caller_name))
                else:
                    norm_range = False
            else:
                norm_range = self.norm_range
        return convert_to_range(limits=norm_range, dims=dims)

    @property
    def dims(self):
        # TODO: improve dim handling
        return tuple(range(self.n_dims))

    @property
    def n_dims(self):
        # TODO: improve n_dims setting
        return self._n_dims

    @n_dims.setter
    def n_dims(self, n_dims):
        self._n_dims = n_dims

    @property
    def is_extended(self):
        return self._yield is not None

    @staticmethod
    def n_dims_from_limits(limits):
        """Return the number of dimensions from the limits."""
        if limits is None or limits is False:
            n_dims = None
        else:
            n_dims = limits.n_dims
        return n_dims

    def set_yield(self, value):
        self._yield = value

    def get_yield(self):
        if not self.is_extended:
            raise zexception.ExtendedPDFError("PDF is not extended, cannot get yield.")
        return self._yield

    @property
    def _yield(self):
        """For internal use, the yield or None"""
        # return self.parameters.get('yield')
        # HACK
        return self._temp_yield
        # return self.parameters['yield']

    @_yield.setter
    def _yield(self, value):
        if value is None:
            # unset
            self.parameters.pop('yield', None)  # safely remove if still there
        else:
            # self.parameters['yield'] = value
            # HACK
            self._temp_yield = value

    def _unnormalized_prob(self, x):
        raise NotImplementedError

    def _call_unnormalized_prob(self, x, name, **kwargs):
        with self._name_scope(name, values=[x]):
            x = tf.convert_to_tensor(x, name="x")
            try:
                return self._unnormalized_prob(x, **kwargs)
            except NotImplementedError:
                return self._prob(x, norm_range="TODO")
            # No fallback, if unnormalized_prob and prob is not implemented

    def unnormalized_prob(self, x, name="unnormalized_prob"):
        return self._call_unnormalized_prob(x, name)

    def _prob(self, x, norm_range):
        raise NotImplementedError

    def _call_prob(self, x, norm_range, name, **kwargs):
        with self._name_scope(name, values=[x, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            with suppress(NotImplementedError):
                return self._prob(x, norm_range=norm_range, **kwargs)
            with suppress(NotImplementedError):
                return tf.exp(self._log_prob(x, norm_range=norm_range))
            return self._fallback_prob(x=x, norm_range=norm_range)

    def _fallback_prob(self, x, norm_range):
        pdf = self.unnormalized_prob(x) / self.normalization(norm_range=norm_range)
        return pdf

    def prob(self, x, norm_range=None, name="prob"):
        """Probability density/mass function.

        Args:
          x: `float` or `double` `Tensor`.
          norm_range (): Range to normalize over
          name: Python `str` prepended to names of ops created by this function.

        Returns:
          prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
            values of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        probability = self._call_prob(x, norm_range, name)
        if self.is_extended:
            probability *= self.get_yield()
        return probability

    def _log_prob(self, x, norm_range):
        raise NotImplementedError

    def _call_log_prob(self, x, norm_range, name, **kwargs):
        with self._name_scope(name, values=[x, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            with suppress(NotImplementedError):
                return self._log_prob(x=x, norm_range=norm_range)
            with suppress(NotImplementedError):
                return tf.log(self._prob(x=x, norm_range=norm_range))
            return self._fallback_log_prob(norm_range, x)

    def _fallback_log_prob(self, norm_range, x):
        return tf.log(self.prob(x=x, norm_range=norm_range))

    def log_prob(self, x, norm_range=None, name="log_prob"):
        """Log probability density/mass function.

        Args:
          x: `float` or `double` `Tensor`.
          name: Python `str` prepended to names of ops created by this function.

        Returns:
          log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
            values of type `self.dtype`.
        """
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)

        return self._call_log_prob(x, norm_range, name)

    def _normalization(self, norm_range):
        raise NotImplementedError

    def _call_normalization(self, norm_range, name):
        # TODO: caching? alternative
        with self._name_scope(name, values=[norm_range]):
            with suppress(NotImplementedError):
                return self._normalization(norm_range)
            return self._fallback_normalization(norm_range)

    def _fallback_normalization(self, norm_range):
        # TODO: multidim, more complicated range
        normalization_value = self.integrate(limits=norm_range, norm_range=False)
        return normalization_value

    def normalization(self, norm_range, name="normalization"):
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)

        return self._hook_normalization(norm_range=norm_range, name=name)

    def _hook_normalization(self, name, norm_range):
        normalization = self._call_normalization(norm_range=norm_range, name=name)
        return normalization

    # Integrals

    def _integrate(self, limits, norm_range):
        raise NotImplementedError()

    def _call_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            limits = convert_to_range(limits, dims=Range.FULL)
            with suppress(NotImplementedError):
                return self._integrate(limits=limits, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_integrate(limits=limits, norm_range=norm_range, )

    def _fallback_integrate(self, limits, norm_range):
        n_dims = limits.n_dims
        dims = limits.dims
        max_dims = self._analytic_integral.get_max_dims()
        print("DEBUG, max_dims, dims", max_dims, dims)

        integral = None
        if max_dims == frozenset(dims):
            with suppress(NotImplementedError):
                integral = self._hook_analytic_integrate(limits=limits, norm_range=norm_range)
        if max_dims and integral is None:  # TODO improve handling of available analytic integrals
            with suppress(NotImplementedError):
                def part_int(x):
                    return self.partial_analytic_integrate(x, limits=limits, dims=dims)

                integral = self._integration.auto_numeric_integrate(func=part_int, limits=limits,
                                                                    n_dims=n_dims,  # HACK
                                                                    mc_options={
                                                                        "draws_per_dim":
                                                                            self._integration.draws_per_dim}
                                                                    )
        if integral is None:
            integral = self.numeric_integrate(limits=limits, norm_range=norm_range)
        return integral

    def integrate(self, limits, norm_range=None, name="integrate"):
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        return self._hook_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_integrate(self, limits, norm_range, name='_hook_integrate'):
        try:
            return self._call_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            unnormalized_integral = self._call_integrate(limits=limits, norm_range=False, name=name)
            normalization = self._hook_normalization(norm_range=limits)
            return unnormalized_integral / normalization

    @classmethod
    def register_analytic_integral(cls, func, dims=None, limits=None):
        """

        Args:
            func ():
            dims (tuple(int)):
            limits ():

        Returns:

        """
        cls._analytic_integral.register(func=func, dims=dims, limits=limits)

    def _analytic_integrate(self, limits, norm_range):
        # TODO: user implementation requested
        raise NotImplementedError

    def _call_analytic_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            limits = convert_to_range(limits, dims=Range.FULL)
            with suppress(NotImplementedError):
                return self._analytic_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_analytic_integrate(limits=limits, norm_range=norm_range)

    def _fallback_analytic_integrate(self, limits, norm_range):
        return self._analytic_integral.integrate(x=None, limits=limits, dims=limits.dims,
                                                 norm_range=norm_range, params=self.parameters)

    def analytic_integrate(self, limits, norm_range=None, name="analytic_integrate"):
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)
        return self._hook_analytic_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_analytic_integrate(self, limits, norm_range, name="_hook_analytic_integrate"):
        try:
            return self._call_analytic_integrate(limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:

            unnormalized_integral = self._call_analytic_integrate(limits, norm_range=None,
                                                                  name=name)
            try:
                normalization = self.analytic_integrate(limits=norm_range, norm_range=False)
            except NotImplementedError:
                raise NormRangeNotImplementedError("Function {} does not support this (or even any)"
                                                   "normalization range 'norm_range'."
                                                   " This usually means,that no analytic integral "
                                                   "is available for this function. Due to rule "
                                                   "safety, an analytical normalization has to "
                                                   "be available and no attempt of numerical "
                                                   "normalization was made.".format(name))
            else:
                return unnormalized_integral / normalization

    def _numeric_integrate(self, limits, norm_range):
        raise NotImplementedError

    def _call_numeric_integrate(self, limits, norm_range, name):
        with self._name_scope(name, values=[limits, norm_range]):
            limits = convert_to_range(limits)
            # TODO: anything?
            with suppress(NotImplementedError):
                return self._numeric_integrate(limits=limits, norm_range=norm_range)
            return self._fallback_numeric_integrate(limits=limits, norm_range=norm_range)

    def _fallback_numeric_integrate(self, limits, norm_range):
        # TODO: get limits properly
        # HACK
        integral = self._integration.auto_numeric_integrate(func=self.unnormalized_prob,
                                                            limits=limits, norm_range=norm_range,
                                                            dtype=self.dtype, n_dims=limits.n_dims,
                                                            mc_sampler=self._integration.mc_sampler,
                                                            mc_options={
                                                                'draws_per_dim':
                                                                    self._integration.draws_per_dim})

        return integral

    def numeric_integrate(self, limits, norm_range=None, name="numeric_integrate"):
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)

        return self._hook_numeric_integrate(limits=limits, norm_range=norm_range, name=name)

    def _hook_numeric_integrate(self, limits, norm_range, name='_hook_numeric_integrate'):
        try:
            return self._call_numeric_integrate(limits=limits, norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the catched Error should not be raised."
            unnormalized_integral = self._call_numeric_integrate(limits=limits, norm_range=None,
                                                                 name=name)
            normalization = self.numeric_integrate(limits=norm_range, name=name + "_normalization")
            return unnormalized_integral / normalization

    def _partial_integrate(self, x, limits, dims, norm_range):
        raise NotImplementedError

    def _call_partial_integrate(self, x, limits, dims, norm_range, name):
        with self._name_scope(name, values=[x, limits, dims, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            limits = convert_to_range(limits, dims=dims)

            with suppress(NotImplementedError):
                return self._partial_integrate(x=x, limits=limits, dims=dims, norm_range=norm_range)
            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits, norm_range=norm_range)

            return self._fallback_partial_integrate(x=x, limits=limits, dims=dims,
                                                    norm_range=norm_range)

    def _fallback_partial_integrate(self, x, limits, dims, norm_range):
        max_dims = self._analytic_integral.get_max_dims(out_of_dims=dims)
        if max_dims:
            def part_int(x):
                return self.partial_analytic_integrate(x=x, limits=limits,
                                                       dims=max_dims, norm_range=norm_range)

            dims = list(set(dims) - set(max_dims))
        else:
            part_int = self.unnormalized_prob

        integral_vals = zintegrate.auto_integrate(func=part_int, limits=limits, dims=dims,
                                                  x=x, dtype=self.dtype,
                                                  mc_sampler=self._integration.mc_sampler,
                                                  mc_options={"draws_per_dim":
                                                                  self._integration.draws_per_dim})
        return integral_vals

    def partial_integrate(self, x, limits, dims, norm_range=None, name="partial_integrate"):
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL,
                                                  caller_name=name)  # TODO: FULL reasonable?
        return self._hook_partial_integrate(x=x, limits=limits, dims=dims,
                                            norm_range=norm_range, name=name)

    def _hook_partial_integrate(self, x, limits, dims, norm_range, name='_hook_partial_integrate'):
        try:
            return self._call_partial_integrate(x=x, limits=limits, dims=dims,
                                                norm_range=norm_range,
                                                name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the catched Error should not be raised."
            unnormalized_integral = self._call_partial_integrate(x=x, limits=limits, dims=dims,
                                                                 norm_range=None,
                                                                 name=name)
            normalization = self.normalization(norm_range=norm_range)
            return unnormalized_integral / normalization

    def _partial_analytic_integrate(self, x, limits, dims, norm_range):
        raise NotImplementedError

    def _call_partial_analytic_integrate(self, x, limits, dims, norm_range, name):
        with self._name_scope(name, values=[x, limits, dims, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            limits = convert_to_range(limits, dims=dims)

            with suppress(NotImplementedError):
                return self._partial_analytic_integrate(x=x, limits=limits, dims=dims,
                                                        norm_range=norm_range)
            return self._fallback_partial_analytic_integrate(x=x, limits=limits, dims=dims,
                                                             norm_range=norm_range)

    def _fallback_partial_analytic_integrate(self, x, limits, dims, norm_range):
        return self._analytic_integral.integrate(x=x, limits=limits, dims=dims,
                                                 norm_range=norm_range, params=self.parameters)

    def partial_analytic_integrate(self, x, limits, dims, norm_range=None,
                                   name="partial_analytic_integrate"):
        """Partial integral over dims.

        Args:
            x ():
            dims (tuple(int)): The dims to integrate over

        Returns:
            Tensor:

        Raises:
            NotImplementedError: if the function is not implemented
            NormRangeNotImplementedError: if the *norm_range* argument is not supported. This
                means that no analytical normalization is available.

        """
        # TODO: implement meaningful, how communicate integrated, not integrated vars?
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL,
                                                  caller_name=name)  # TODO: full reasonable?
        return self._hook_partial_analytic_integrate(x, limits=limits, dims=dims,
                                                     norm_range=norm_range, name=name)

    def _hook_partial_analytic_integrate(self, x, limits, dims, norm_range,
                                         name='_hook_partial_analytic_integrate'):
        try:
            return self._call_partial_analytic_integrate(x=x, limits=limits, dims=dims,
                                                         norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the catched Error should not be raised."
            unnormalized_integral = self._call_partial_analytic_integrate(x=x, limits=limits,
                                                                          dims=dims,
                                                                          norm_range=None,
                                                                          name=name)
            try:
                normalization = self.analytic_integrate(limits=norm_range, norm_range=False)
            except NotImplementedError:
                raise NormRangeNotImplementedError("Function {} does not support this (or even any)"
                                                   "normalization range 'norm_range'."
                                                   " This usually means,that no analytic integral "
                                                   "is available for this function. Due to rule "
                                                   "safety, an analytical normalization has to "
                                                   "be available and no attempt of numerical "
                                                   "normalization was made.".format(name))
            else:
                return unnormalized_integral / normalization

    def _partial_numeric_integrate(self, x, limits, dims, norm_range):
        raise NotImplementedError

    def _call_partial_numeric_integrate(self, x, limits, dims, norm_range, name):
        with self._name_scope(name, values=[x, limits, dims, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            limits = convert_to_range(limits, dims=dims)
            with suppress(NotImplementedError):
                return self._partial_numeric_integrate(x=x, limits=limits, dims=dims,
                                                       norm_range=norm_range)
            return self._fallback_partial_numeric_integrate(x=x, limits=limits, dims=dims,
                                                            norm_range=norm_range)

    @no_norm_range
    def _fallback_partial_numeric_integrate(self, x, limits, dims):
        return zintegrate.auto_integrate(func=self.unnormalized_prob, limits=limits, dims=dims,
                                         x=x, dtype=self.dtype,
                                         mc_sampler=self._integration.mc_sampler,
                                         mc_options={
                                             "draws_per_dim": self._integration.draws_per_dim})

    def partial_numeric_integrate(self, x, limits, dims, norm_range=None, name="partial_numeric_integrate"):
        norm_range = self._check_input_norm_range(norm_range, dims=Range.FULL, caller_name=name)

        return self._hook_partial_numeric_integrate(dims, limits, name, norm_range, x)

    def _hook_partial_numeric_integrate(self, x, limits, dims, norm_range,
                                        name='_hook_partial_numeric_integrate'):
        try:
            return self._call_partial_numeric_integrate(x=x, limits=limits, dims=dims,
                                                        norm_range=norm_range, name=name)
        except NormRangeNotImplementedError:
            assert norm_range is not False, "Internal: the catched Error should not be raised."
            unnormalized_integral = self._call_partial_numeric_integrate(x=x, limits=limits,
                                                                         dims=dims, norm_range=None,
                                                                         name=name)
            return unnormalized_integral / self._hook_normalization(norm_range=norm_range)

    def _sample(self, n_draws, limits):
        raise NotImplementedError

    def _call_sample(self, n_draws, limits, name):
        with self._name_scope(name, values=[n_draws, limits]):
            n_draws = tf.convert_to_tensor(n_draws, name="n_draws")
            limits = convert_to_range(limits, dims=Range.FULL)
            with suppress(NotImplementedError):
                return self._sample(n_draws=n_draws, limits=limits)
            return self._fallback_sample(n_draws=n_draws, limits=limits)

    def _fallback_sample(self, n_draws, limits):
        sample = zsample.accept_reject_sample(prob=self.unnormalized_prob,  # no need to normalize
                                              n_draws=n_draws,
                                              limits=limits, prob_max=None)  # None -> auto
        return sample

    def sample(self, n_draws, limits, name="sample"):
        return self._hook_sample(limits=limits, n_draws=n_draws, name=name)

    def _hook_sample(self, limits, n_draws, name='_hook_sample'):
        return self._call_sample(n_draws=n_draws, limits=limits, name=name)

    # def copy(self, **override_parameters_kwargs):
    #     """Creates a deep copy of the distribution.
    #
    #     Note: the copy distribution may continue to depend on the original
    #     initialization arguments.
    #
    #     Args:
    #       **override_parameters_kwargs: String/value dictionary of initialization
    #         arguments to override with new values.
    #
    #     Returns:
    #       distribution: A new instance of `type(self)` initialized from the union
    #         of self.parameters and override_parameters_kwargs, i.e.,
    #         `dict(self.parameters, **override_parameters_kwargs)`.
    #     """
    #     parameters = dict(self.parameters, **override_parameters_kwargs)
    #     yield_ = parameters.pop('yield', None)
    #     new_instance = type(self)(**parameters)
    #     if yield_ is not None:
    #         new_instance.set_yield(yield_)
    #     return new_instance

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

    def __repr__(self):
        return ("<tf.distributions.{type_name} "
                "'{self_name}'"
                " batch_shape={batch_shape}"
                " event_shape={event_shape}"
                " dtype={dtype}>".format(
            type_name=type(self).__name__,
            self_name=self.name,
            batch_shape=self.batch_shape,
            event_shape=self.event_shape,
            dtype=self.dtype.name))


class WrapDistribution(BasePDF):

    def __init__(self, distribution, name=None, **kwargs):
        # Check if subclass of distribution?
        name = name or distribution.name
        super(WrapDistribution, self).__init__(distribution=distribution, name=name, **kwargs)
        # self.tf_distribution = self.parameters['distribution']
        self.tf_distribution = distribution

    def _unnormalized_prob(self, x):
        return self.tf_distribution.prob(value=x, name="asdf")  # TODO name

    # TODO: register integral
    def _analytic_integrate(self, limits, norm_range):  # TODO deal with norm_range
        lower, upper = limits.get_boundaries()  # TODO: limits
        upper = tf.cast(upper, dtype=tf.float64)
        lower = tf.cast(lower, dtype=tf.float64)
        integral = self.tf_distribution.cdf(upper, name="asdf2") - self.tf_distribution.cdf(
            lower,
            name="asdf3")  # TODO name
        return integral


# TODO: remove below, play around while developing
if __name__ == "__main":
    import zfit

    mu_true = 1.4
    sigma_true = 1.8


    class TestGaussian(zfit.core.basepdf.BasePDF):
        def _func(self, x):
            return tf.exp(-(x - mu_true) ** 2 / sigma_true ** 2)  # non-normalized gaussian


    dist1 = TestGaussian()
    tf_gauss1 = tf.distributions.Normal(loc=mu_true, scale=sigma_true)
    wrapped = WrapDistribution(tf_gauss1)

    with tf.Session() as sess:
        res = sess.run(dist1.event_shape_tensor())
        print(res)
