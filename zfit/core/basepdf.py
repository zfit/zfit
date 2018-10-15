"""
Definition of the pdf interface, base etc.
"""
from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow_probability.python.mcmc as mc
import tensorflow_probability as tfp
import numpy as np

import zfit.core.utils as utils
# import zfit.core.integrate
import zfit.settings
import zfit.core.integrate as zintegrate


class AbstractBasePDF(object):

    def sample(self, sample_shape=(), seed=None, name='sample'):
        raise NotImplementedError

    def unnormalized_prob(self, value, name='unnormalized_prob'):
        raise NotImplementedError

    def log_prob(self, value, name='log_prob'):
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
    _DEFAULTS_integration = utils.dotdict()
    _DEFAULTS_integration.mc_sampler = mc.sample_halton_sequence
    _DEFAULTS_integration.draws_per_dim = 10000
    _DEFAULTS_integration.auto_numeric_integrate = zfit.core.integrate.auto_integrate

    _analytic_integral = zfit.core.integrate.AnalyticIntegral()

    def __init__(self, name="BaseDistribution", **kwargs):
        # TODO: catch some args from kwargs that belong to the super init?
        super(BasePDF, self).__init__(dtype=zfit.settings.fptype, reparameterization_type=False,
                                      validate_args=True, parameters=kwargs,
                                      allow_nan_stats=False, name=name)

        self.n_dims = None
        # self.norm_range = None
        self.norm_range = ((1, 2),)  # HACK! Take line above
        # self.normalization_opt = {'n_draws': 10000000, 'range': (-100., 100.)}
        self._integration = utils.dotdict()
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

    @staticmethod
    def n_dims_from_limits(limits):
        """Return the number of dimensions from the limits."""
        # TODO: replace with more intelligent limits object
        if limits is None:
            n_dims = None
        else:
            n_dims = len(limits)
        return n_dims

    def _unnormalized_prob(self, value):
        raise NotImplementedError

    def _call_unnormalized_prob(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = tf.convert_to_tensor(value, name="value")
            try:
                return self._unnormalized_prob(value, **kwargs)
            except NotImplementedError:
                return self._prob(value, norm_range=NormRange.ANY_RANGE)
            # No fallback, if unnormalized_prob and prob is not implemented

    def unnormalized_prob(self, value, name="unnormalized_prob"):
        return self._call_unnormalized_prob(value, name)

    def _prob(self, value, norm_range):
        raise NotImplementedError

    def _call_prob(self, value, norm_range, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = tf.convert_to_tensor(value, name="value")
            try:
                return self._prob(value, norm_range=norm_range, **kwargs)
            except NotImplementedError:
                pass
            try:
                return tf.exp(self._log_prob(value, norm_range=norm_range))
            except NotImplementedError:
                pass
            return self._fallback_prob(value=value, norm_range=norm_range)

    def _fallback_prob(self, value, norm_range):
        pdf = self.unnormalized_prob(value) / self.normalization(norm_range=norm_range)
        return pdf

    def prob(self, value, norm_range=None, name="prob"):
        """Probability density/mass function.

        Args:
          value: `float` or `double` `Tensor`.
          norm_range (): Range to normalize over
          name: Python `str` prepended to names of ops created by this function.

        Returns:
          prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
            values of type `self.dtype`.
        """
        norm_range = norm_range or self.norm_range
        if norm_range is None:
            raise ValueError("Normalization range not specified.")
        return self._call_prob(value, norm_range, name)

    def _log_prob(self, value, norm_range):
        raise NotImplementedError

    def _call_log_prob(self, value, norm_range, name):
        try:
            return self._log_prob(value=value, norm_range=norm_range)
        except NotImplementedError:
            pass
        try:
            return tf.log(self._prob(value=value, norm_range=norm_range))
        except NotImplementedError:
            pass
        return self._fallback_log_prob(norm_range, value)

    def _fallback_log_prob(self, norm_range, value):
        return tf.log(self.prob(value=value, norm_range=norm_range))

    def log_prob(self, value, norm_range=None, name="log_prob"):
        """Log probability density/mass function.

        Args:
          value: `float` or `double` `Tensor`.
          name: Python `str` prepended to names of ops created by this function.

        Returns:
          log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
            values of type `self.dtype`.
        """
        return self._call_log_prob(value, norm_range, name)

    def _normalization(self, norm_range):
        raise NotImplementedError

    def _call_normalization(self, norm_range, name):
        # TODO: caching? alternative
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
        n_dims = 1
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
        n_dims = 1  # HACK
        max_dims = self._analytic_integral.get_max_dims()
        if max_dims:
            def part_int(value):
                return self._partial_analytic_integrate(value, limits=limits)

            integral = self._integration.auto_numeric_integrate(func=part_int, limits=limits,
                                                                n_dims=n_dims,  # HACK
                                                                mc_options={
                                                                    "draws_per_dim":
                                                                        self._integration.draws_per_dim}
                                                                )
        else:
            integral = self.numeric_integrate(limits=limits)
        return integral

    def integrate(self, limits, name="integrate"):
        return self._call_integrate(limits=limits, name=name)

    @classmethod
    def register_analytic_integral(cls, func, dims=None):
        """

        Args:
            func ():
            dims (tuple(int)):

        Returns:

        """
        cls._analytic_integral.register(func=func, dims=dims)

    def _analytic_integrate(self, limits):
        # TODO: user implementation requested
        raise NotImplementedError

    def _call_analytic_integrate(self, limits, name):
        try:
            return self._analytic_integrate(limits=limits)
        except NotImplementedError:
            pass
        return self._fallback_analytic_integrate(limits)

    def _fallback_analytic_integrate(self, limits):
        self._analytic_integral.integrate(value=None, limits=limits, dims=self.dims)

    def analytic_integrate(self, limits, name="analytic_integrate"):
        # TODO: get limits
        integral = self._call_analytic_integrate(limits, name=name)
        return integral

    def _numeric_integrate(self, limits):
        raise NotImplementedError

    def _call_numeric_integrate(self, limits, name):
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

    def _partial_integrate(self, value, limits, dims):
        raise NotImplementedError

    def _call_partial_integrate(self, value, limits, dims, name):
        try:
            return self._partial_integrate(value=value, limits=limits, dims=dims)
        except NotImplementedError:
            pass
        try:
            return self._partial_analytic_integrate(value=value, limits=limits)
        except NotImplementedError:
            pass

        return self._fallback_partial_integrate(value=value, limits=limits, dims=dims)

    def _fallback_partial_integrate(self, value, limits, dims):
        max_dims = self._analytic_integral.get_max_dims(out_of_dims=dims)
        if max_dims:
            def part_int(value):
                return self.partial_analytic_integrate(value=value, limits=limits,
                                                       dims=max_dims)

            dims = list(set(dims) - set(max_dims))
        else:
            part_int = self.unnormalized_prob

        integral_vals = zintegrate.auto_integrate(func=part_int, limits=limits, dims=dims,
                                                  value=value, dtype=self.dtype,
                                                  mc_sampler=self._integration.mc_sampler,
                                                  mc_options={"draws_per_dim":
                                                                  self._integration.draws_per_dim})
        return integral_vals

    def partial_integrate(self, value, limits, dims, name="partial_integrate"):
        return self._call_partial_integrate(value=value, limits=limits, dims=dims, name=name)

    def _partial_analytic_integrate(self, value, limits, dims):
        raise NotImplementedError

    def _call_partial_analytic_integrate(self, value, limits, dims, name):
        try:
            return self._partial_analytic_integrate(value=value, limits=limits, dims=dims)
        except NotImplementedError:
            pass
        return self._fallback_partial_analytic_integrate(value=value, limits=limits, dims=dims)

    def _fallback_partial_analytic_integrate(self, value, limits, dims):
        return self._analytic_integrals.integrate(value=value, limits=limits, dims=dims)

    def partial_analytic_integrate(self, value, limits, dims, name="partial_analytic_integrate"):
        """Partial integral over dims.

        Args:
            value ():
            dims (tuple(int)): The dims to integrate over

        Returns:
            Tensor:

        Raises:
            NotImplementedError: if the function is not implemented

        """
        # TODO: implement meaningful, how communicate integrated, not integrated vars?
        return self._call_analytic_integrate(value=value, limits=limits, dims=dims, name=name)

    def _partial_numeric_integrate(self, value, limits, dims):
        raise NotImplementedError

    def _call_partial_numeric_integrate(self, value, limits, dims, name):
        try:
            return self._partial_numeric_integrate(value=value, limits=limits, dims=dims)
        except NotImplementedError:
            pass
        return self._fallback_partial_numeric_integrate(value=value, limits=limits, dims=dims)

    def _fallback_partial_numeric_integrate(self, value, limits, dims):
        return zintegrate.auto_integrate(func=self.unnormalized_prob, limits=limits, dims=dims,
                                         value=value, dtype=self.dtype,
                                         mc_sampler=self._integration.mc_sampler,
                                         mc_options={
                                             "draws_per_dim": self._integration.draws_per_dim})

    def partial_numeric_integrate(self, value, limits, dims, name="partial_numeric_integrate"):
        return self._call_partial_numeric_integrate(value=value, limits=limits, dims=dims,
                                                    name=name)


class WrapDistribution(BasePDF):

    def __init__(self, distribution, name=None, **kwargs):
        # Check if subclass of distribution?
        name = name or distribution.name
        super(WrapDistribution, self).__init__(distribution=distribution, name=name, **kwargs)
        # self.tf_distribution = self.parameters['distribution']
        self.tf_distribution = distribution

    def _unnormalized_prob(self, value):
        return self.tf_distribution.prob(value=value, name="asdf")  # TODO name

    # TODO: register integral
    def _analytic_integrate(self, limits):
        lower, upper = limits  # TODO: limits
        upper = tf.cast(upper, dtype=tf.float64)
        lower = tf.cast(lower, dtype=tf.float64)
        integral = self.tf_distribution.cdf(upper, name="asdf2") - self.tf_distribution.cdf(lower,
                                                                                            name="asdf3")  # TODO name
        return integral


# TODO: remove below, play around while developing
if __name__ == "__main":
    import zfit

    mu_true = 1.4
    sigma_true = 1.8


    class TestGaussian(zfit.core.basepdf.BasePDF):
        def _func(self, value):
            return tf.exp(-(value - mu_true) ** 2 / sigma_true ** 2)  # non-normalized gaussian


    dist1 = TestGaussian()
    tf_gauss1 = tf.distributions.Normal(loc=mu_true, scale=sigma_true)
    wrapped = WrapDistribution(tf_gauss1)

    with tf.Session() as sess:
        res = sess.run(dist1.event_shape_tensor())
        print(res)
