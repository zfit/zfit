"""
Definition of the pdf interface, base etc.
"""
from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability.python.mcmc as mc
import numpy as np

from ..utils import container
# import zfit.core.integrate
from ..settings import types as ztypes
from . import integrate as zintegrate
from . import sample as zsample
# from zfit.settings import types as ztypes
from ..utils import exception as zexception
from ..utils import container as zcontainer


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


class NormRange(object):
    __ANY_RANGE = "Not yet implemented"

    @property
    def ANY_RANGE(self):
        """Any range, it does not matter"""
        raise NotImplementedError("NOT YET!")
        return self.__ANY_RANGE


class BasePDF(tf.distributions.Distribution, AbstractBasePDF):
    _DEFAULTS_integration = container.dotdict()
    _DEFAULTS_integration.mc_sampler = mc.sample_halton_sequence
    _DEFAULTS_integration.draws_per_dim = 4000
    _DEFAULTS_integration.auto_numeric_integrate = zintegrate.auto_integrate

    _analytic_integral = zintegrate.AnalyticIntegral()

    def __init__(self, name="BaseDistribution", **kwargs):
        # TODO: catch some args from kwargs that belong to the super init?
        super(BasePDF, self).__init__(dtype=ztypes.float,
                                      reparameterization_type=False,
                                      validate_args=True, parameters=kwargs,
                                      allow_nan_stats=False, name=name)

        self.n_dims = None
        self._yield = None
        self._temp_yield = None
        # self.norm_range = None
        self.norm_range = ((1, 2),)  # HACK! Take line above
        # self.normalization_opt = {'n_draws': 10000000, 'range': (-100., 100.)}
        self._integration = zcontainer.dotdict()
        self._integration.mc_sampler = self._DEFAULTS_integration.mc_sampler
        self._integration.draws_per_dim = self._DEFAULTS_integration.draws_per_dim
        self._integration.auto_numeric_integrate = self._DEFAULTS_integration.auto_numeric_integrate
        self._normalization_value = None

    @property
    def norm_range(self):
        return self._norm_range

    @norm_range.setter
    def norm_range(self, norm_range):
        if not self.n_dims:
            self.n_dims = self.n_dims_from_limits(norm_range)
        self._norm_range = norm_range

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
        # TODO: replace with more intelligent limits object
        if limits is None:
            n_dims = None
        else:
            n_dims = len(limits)
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
                return self._prob(x, norm_range=NormRange.ANY_RANGE)
            # No fallback, if unnormalized_prob and prob is not implemented

    def unnormalized_prob(self, x, name="unnormalized_prob"):
        return self._call_unnormalized_prob(x, name)

    def _prob(self, x, norm_range):
        raise NotImplementedError

    def _call_prob(self, x, norm_range, name, **kwargs):
        with self._name_scope(name, values=[x, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            norm_range = convert_to_range(norm_range)
            try:
                return self._prob(x, norm_range=norm_range, **kwargs)
            except NotImplementedError:
                pass
            try:
                return tf.exp(self._log_prob(x, norm_range=norm_range))
            except NotImplementedError:
                pass
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
        norm_range = norm_range or self.norm_range
        if norm_range is None:
            raise ValueError("Normalization range not specified.")
        probability = self._call_prob(x, norm_range, name)
        if self.is_extended:
            probability *= self._yield
        return probability

    def _log_prob(self, x, norm_range):
        raise NotImplementedError

    def _call_log_prob(self, x, norm_range, name, **kwargs):
        with self._name_scope(name, values=[x, norm_range]):
            x = tf.convert_to_tensor(x, name="x")
            norm_range = convert_to_range(norm_range)

            try:
                return self._log_prob(x=x, norm_range=norm_range)
            except NotImplementedError:
                pass
            try:
                return tf.log(self._prob(x=x, norm_range=norm_range))
            except NotImplementedError:
                pass
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
        return self._call_log_prob(x, norm_range, name)

    def _normalization(self, norm_range):
        raise NotImplementedError

    def _call_normalization(self, norm_range, name):
        # TODO: caching? alternative
        with self._name_scope(name, values=[norm_range]):
            norm_range = convert_to_range(norm_range)
            try:
                return self._normalization(norm_range)
            except NotImplementedError:
                pass
            return self._fallback_normalization(norm_range)

    def _fallback_normalization(self, norm_range):
        # TODO: multidim, more complicated range
        normalization_value = self.integrate(limits=norm_range)
        return normalization_value

    def normalization(self, norm_range, name="normalization"):
        normalization = self._call_normalization(norm_range, name=name)
        return normalization

    # Integrals

    def _integrate(self, limits):
        raise NotImplementedError()

    def _call_integrate(self, limits, name):
        with self._name_scope(name, values=[limits]):
            limits = convert_to_range(limits)
            try:
                return self._integrate(limits)
            except NotImplementedError:
                pass
            try:
                return self._analytic_integrate(limits)
            except NotImplementedError:
                pass
            return self._fallback_integrate(limits)

    def _fallback_integrate(self, limits):
        n_dims = self.n_dims  # HACK
        dims = self.dims  # HACK
        max_dims = self._analytic_integral.get_max_dims()
        print("DEBUG, max_dims, dims", max_dims, dims)

        integral = None
        if max_dims == frozenset(dims):
            try:
                integral = self.analytic_integrate(limits=limits)
            except NotImplementedError:
                pass
        if max_dims and integral is None:  # TODO improve handling of available analytic integrals
            try:
                def part_int(x):
                    return self.partial_analytic_integrate(x, limits=limits, dims=dims)

                integral = self._integration.auto_numeric_integrate(func=part_int, limits=limits,
                                                                    n_dims=n_dims,  # HACK
                                                                    mc_options={
                                                                        "draws_per_dim":
                                                                            self._integration.draws_per_dim}
                                                                    )
            except NotImplementedError:
                pass
        if integral is None:
            integral = self.numeric_integrate(limits=limits)
        return integral

    def integrate(self, limits, name="integrate"):
        return self._call_integrate(limits=limits, name=name)

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

    def _analytic_integrate(self, limits):
        # TODO: user implementation requested
        raise NotImplementedError

    def _call_analytic_integrate(self, limits, name):
        with self._name_scope(name, values=[limits]):
            limits = convert_to_range(limits)
            try:
                return self._analytic_integrate(limits=limits)
            except NotImplementedError:
                pass
            return self._fallback_analytic_integrate(limits)

    def _fallback_analytic_integrate(self, limits):
        return self._analytic_integral.integrate(x=None, limits=limits, dims=self.dims,
                                                 params=self.parameters)

    def analytic_integrate(self, limits, name="analytic_integrate"):
        # TODO: get limits
        integral = self._call_analytic_integrate(limits, name=name)
        return integral

    def _numeric_integrate(self, limits):
        raise NotImplementedError

    def _call_numeric_integrate(self, limits, name):
        with self._name_scope(name, values=[limits]):
            limits = convert_to_range(limits)
            # TODO: anything?
            try:
                return self._numeric_integrate(limits=limits)
            except NotImplementedError:
                pass
            return self._fallback_numeric_integrate(limits=limits)

    def _fallback_numeric_integrate(self, limits):
        # TODO: get limits properly
        # HACK
        n_dims = 1
        integral = self._integration.auto_numeric_integrate(func=self.unnormalized_prob,
                                                            limits=limits,
                                                            dtype=self.dtype, n_dims=n_dims,
                                                            mc_sampler=self._integration.mc_sampler,
                                                            mc_options={
                                                                'draws_per_dim':
                                                                    self._integration.draws_per_dim})

        return integral

    def numeric_integrate(self, limits, name="numeric_integrate"):
        return self._call_numeric_integrate(limits=limits, name=name)

    def _partial_integrate(self, x, limits, dims):
        raise NotImplementedError

    def _call_partial_integrate(self, x, limits, dims, name):
        with self._name_scope(name, values=[x, limits, dims]):
            x = tf.convert_to_tensor(x, name="x")
            limits = convert_to_range(limits, dims=dims)
            try:
                return self._partial_integrate(x=x, limits=limits, dims=dims)
            except NotImplementedError:
                pass
            try:
                return self._partial_analytic_integrate(x=x, limits=limits)
            except NotImplementedError:
                pass

            return self._fallback_partial_integrate(x=x, limits=limits, dims=dims)

    def _fallback_partial_integrate(self, x, limits, dims):
        max_dims = self._analytic_integral.get_max_dims(out_of_dims=dims)
        if max_dims:
            def part_int(x):
                return self.partial_analytic_integrate(x=x, limits=limits,
                                                       dims=max_dims)

            dims = list(set(dims) - set(max_dims))
        else:
            part_int = self.unnormalized_prob

        integral_vals = zintegrate.auto_integrate(func=part_int, limits=limits, dims=dims,
                                                  x=x, dtype=self.dtype,
                                                  mc_sampler=self._integration.mc_sampler,
                                                  mc_options={"draws_per_dim":
                                                                  self._integration.draws_per_dim})
        return integral_vals

    def partial_integrate(self, x, limits, dims, name="partial_integrate"):
        return self._call_partial_integrate(x=x, limits=limits, dims=dims, name=name)

    def _partial_analytic_integrate(self, x, limits, dims):
        raise NotImplementedError

    def _call_partial_analytic_integrate(self, x, limits, dims, name):
        with self._name_scope(name, values=[x, limits, dims]):
            x = tf.convert_to_tensor(x, name="x")
            limits = convert_to_range(limits, dims)
            try:
                return self._partial_analytic_integrate(x=x, limits=limits, dims=dims)
            except NotImplementedError:
                pass
            return self._fallback_partial_analytic_integrate(x=x, limits=limits, dims=dims)

    def _fallback_partial_analytic_integrate(self, x, limits, dims):
        return self._analytic_integral.integrate(x=x, limits=limits, dims=dims,
                                                 params=self.parameters)

    def partial_analytic_integrate(self, x, limits, dims, name="partial_analytic_integrate"):
        """Partial integral over dims.

        Args:
            x ():
            dims (tuple(int)): The dims to integrate over

        Returns:
            Tensor:

        Raises:
            NotImplementedError: if the function is not implemented

        """
        # TODO: implement meaningful, how communicate integrated, not integrated vars?
        return self._call_partial_analytic_integrate(x=x, limits=limits, dims=dims, name=name)

    def _partial_numeric_integrate(self, x, limits, dims):
        raise NotImplementedError

    def _call_partial_numeric_integrate(self, x, limits, dims, name):
        with self._name_scope(name, values=[x, limits, dims]):
            x = tf.convert_to_tensor(x, name="x")
            limits = convert_to_range(limits, dims=dims)
            try:
                return self._partial_numeric_integrate(x=x, limits=limits, dims=dims)
            except NotImplementedError:
                pass
            return self._fallback_partial_numeric_integrate(x=x, limits=limits, dims=dims)

    def _fallback_partial_numeric_integrate(self, x, limits, dims):
        return zintegrate.auto_integrate(func=self.unnormalized_prob, limits=limits, dims=dims,
                                         x=x, dtype=self.dtype,
                                         mc_sampler=self._integration.mc_sampler,
                                         mc_options={
                                             "draws_per_dim": self._integration.draws_per_dim})

    def partial_numeric_integrate(self, x, limits, dims, name="partial_numeric_integrate"):
        return self._call_partial_numeric_integrate(x=x, limits=limits, dims=dims,
                                                    name=name)

    def _sample(self, n_draws, limits):
        raise NotImplementedError

    def _call_sample(self, n_draws, limits, name):
        with self._name_scope(name, values=[n_draws, limits]):
            n_draws = tf.convert_to_tensor(n_draws, name="n_draws")
            limits = convert_to_range(limits)
            try:
                return self._sample(n_draws=n_draws, limits=limits)
            except NotImplementedError:
                pass
            return self._fallback_sample(n_draws=n_draws, limits=limits)

    def _fallback_sample(self, n_draws, limits):
        sample = zsample.accept_reject_sample(prob=self.prob, n_draws=n_draws, limits=limits)
        return sample

    def sample(self, n_draws, limits=None, name="sample"):
        limits = limits or self.norm_range  # TODO: catch better
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
    def _analytic_integrate(self, limits):
        lower, upper = limits  # TODO: limits
        upper = tf.cast(upper, dtype=tf.float64)
        lower = tf.cast(lower, dtype=tf.float64)
        integral = self.tf_distribution.cdf(upper, name="asdf2") - self.tf_distribution.cdf(lower,
                                                                                            name="asdf3")  # TODO name
        return integral


class Range(object):
    def __init__(self, limits, dims=None):
        self._area = None
        self._set_limits_and_dims(limits, dims)

    def _set_limits_and_dims(self, limits, dims):
        # TODO all the conversions come here
        limits, inferred_dims, has_none = self.sanitize_limits(limits)
        assert len(limits) == len(inferred_dims)
        dims = self.sanitize_dims(dims)
        if dims is None:
            if has_none:
                dims = inferred_dims
            else:
                raise ValueError(
                    "Due to safety: no dims provided, no Nones in limits. Provide dims.")
        else:  # only check if dims from user input
            if len(dims) != len(limits):
                raise ValueError("Dims and limits have different number of axis.")

        self._limits = limits
        self._dims = dims

    @staticmethod
    def sanitize_limits(limits):
        inferred_dims = []
        sanitized_limits = []
        has_none = False
        for i, dim in enumerate(limits):
            if dim is not None:
                sanitized_limits.append(dim)
                inferred_dims.append(i)
            else:
                has_none = True
        if len(np.shape(sanitized_limits)) == 1:
            are_scalars = [np.shape(l) == () for l in sanitized_limits]
            all_scalars = all(are_scalars)
            all_tuples = not any(are_scalars)
            if not (all_scalars or all_tuples):
                raise ValueError("Invalid format for limits: {}".format(limits))
            elif all_scalars:
                sanitized_limits = (tuple(sanitized_limits),)
                inferred_dims = (0,)
        sanitized_limits = tuple(sanitized_limits)
        inferred_dims = tuple(inferred_dims)
        return sanitized_limits, inferred_dims, has_none

    @staticmethod
    def sanitize_dims(dims):

        if dims is not None and len(np.shape(dims)) == 0:
            dims = (dims,)
        return dims

    @property
    def area(self):
        if self._area is None:
            self._calculate_save_area()
        return self._area

    def _calculate_save_area(self):
        area = 1.
        for dims in self:
            sub_area = 0
            for lower, upper in iter_limits(dims):
                sub_area += upper - lower
            area *= sub_area
        self._area = area
        return area

    @property
    def dims(self):
        return self._dims

    def as_tuple(self):
        return self._limits

    def as_array(self):
        return np.array(self._limits)

    def __lt__(self, other):
        if self.dims != other.dims:
            return False
        for dim, other_dim in zip(self, other):
            for lower, upper in iter_limits(dim):
                is_smaller = False
                for other_lower, other_upper in iter_limits(other_dim):
                    is_smaller = other_lower <= lower and upper <= other_upper
                    if is_smaller:
                        break
                if not is_smaller:
                    return False
        return True

    def __gt__(self, other):
        return other < self

    def __eq__(self, other):
        if self.dims != other.dims:
            return False
        return self.as_tuple() == other.as_tuple()

    def __getitem__(self, key):
        return self.as_tuple()[key]


def convert_to_range(limits, dims=None):
    if isinstance(limits, Range):
        return limits
    else:
        return Range(limits, dims=dims)


def iter_limits(limits):
    """Returns (lower, upper) for an iterable containing several such pairs

    Args:
        limits (iterable): A 1-dimensional iterable containing an even number of values. The odd
            values are takes as the lower limit while the even values are taken as the upper limit.
            Example: [a_lower, a_upper, b_lower, b_upper]

    Returns:
        iterable(tuples(lower, upper)): Returns an iterable containing the lower, upper tuples.
            Example (from above): [(a_lower, a_upper), (b_lower, b_upper)]

    Raises:
        ValueError: if limits does not contain an even number of elements.
    """
    if not len(limits) % 2 == 0:
        raise ValueError("limits has to be from even length, not: {}".format(limits))
    return zip(limits[::2], limits[1::2])


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
