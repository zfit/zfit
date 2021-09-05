#  Copyright (c) 2021 zfit
from typing import Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd

import zfit.z.numpy as znp

from .. import z
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitData, ZfitParameter, ZfitSpace
from ..settings import ztypes
from ..util import binning as binning_util
from ..util import convolution as convolution_util
from ..util import improved_sheather_jones as isj_util
from ..util import ztyping
from ..util.exception import OverdefinedError, ShapeIncompatibleError
from ..z.math import weighted_quantile
from .dist_tfp import WrapDistribution


@z.function(wraps='tensor')
def bandwidth_rule_of_thumb(data, weights, factor=None):
    if factor is None:
        factor = tf.constant(0.9)
    return min_std_or_iqr(data, weights) * tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.) * factor


@z.function(wraps='tensor')
def bandwidth_silverman(data, weights):
    return bandwidth_rule_of_thumb(data=data, weights=weights, factor=znp.array(0.9, dtype=ztypes.float))


@z.function(wraps='tensor')
def bandwidth_scott(data, weights):
    return bandwidth_rule_of_thumb(data=data, weights=weights, factor=znp.array(1.059, dtype=ztypes.float))


def bandwidth_isj(data, weights):
    return isj_util.calculate_bandwidth(data, num_grid_points=1024, binning_method='linear', weights=weights)


def bandwidth_adaptive_geomV1(data, func, weights):
    data = z.convert_to_tensor(data)
    if weights is not None:
        n = znp.sum(weights)
    else:
        n = tf.cast(tf.shape(data)[0], ztypes.float)
    probs = func(data)
    lambda_i = 1 / znp.sqrt(probs / z.math.reduce_geometric_mean(probs, weights=weights))

    return lambda_i * n ** (-1. / 5.) * min_std_or_iqr(data, weights)


def bandwidth_adaptive_zfitV1(data, func, weights):
    data = z.convert_to_tensor(data)
    probs = func(data)
    estimate = bandwidth_scott(data, weights=weights)
    factor = znp.sqrt(probs) / znp.mean(znp.sqrt(probs))
    return estimate / factor


def bandwidth_adaptive_stdV1(data, func, weights):
    data = z.convert_to_tensor(data)
    if weights is not None:
        n = znp.sum(weights)
    else:
        n = tf.cast(tf.shape(data)[0], ztypes.float)
    probs = func(data)
    divisor = min_std_or_iqr(data, weights)
    bandwidth = z.sqrt(divisor / probs)
    bandwidth *= tf.cast(n, ztypes.float) ** (-1. / 5.) * 1.059
    return bandwidth


def adaptive_factory(func, grid):
    if grid:
        def adaptive(constructor, data, **kwargs):
            kwargs.pop('name', None)
            kde_silverman = constructor(bandwidth='silverman', data=data,
                                        name=f"INTERNAL_for_adaptive_kde", **kwargs)
            grid = kde_silverman._grid
            weights = kde_silverman._grid_data
            return func(data=grid, func=kde_silverman.pdf, weights=weights * tf.cast(tf.shape(data)[0], ztypes.float))
    else:
        def adaptive(constructor, data, weights, **kwargs):
            kwargs.pop('name', None)
            kde_silverman = constructor(bandwidth='silverman', data=data,
                                        name=f"INTERNAL_for_adaptive_kde", **kwargs)
            return func(data=data, func=kde_silverman.pdf, weights=weights)
    return adaptive


_adaptive_geom_bandwidth_grid_KDEV1 = adaptive_factory(bandwidth_adaptive_geomV1, grid=True)
_adaptive_geom_bandwidth_KDEV1 = adaptive_factory(bandwidth_adaptive_geomV1, grid=False)

_adaptive_std_bandwidth_grid_KDEV1 = adaptive_factory(bandwidth_adaptive_stdV1, grid=True)
_adaptive_std_bandwidth_KDEV1 = adaptive_factory(bandwidth_adaptive_stdV1, grid=False)

_adaptive_zfit_bandwidth_grid_KDEV1 = adaptive_factory(bandwidth_adaptive_zfitV1, grid=True)
_adaptive_zfit_bandwidth_KDEV1 = adaptive_factory(bandwidth_adaptive_zfitV1, grid=False)


def _bandwidth_scott_KDEV1(data, weights, *_, **__):
    return bandwidth_scott(data, weights=weights, )


def _bandwidth_silverman_KDEV1(data, weights, *_, **__):
    return bandwidth_silverman(data, weights=weights, )


def _bandwidth_isj_KDEV1(data, weights, *_, **__):
    return bandwidth_isj(data, weights=weights)


@z.function(wraps='tensor')
def min_std_or_iqr(x, weights):
    # TODO: use weighted percentile
    if weights is not None:
        return znp.minimum(znp.sqrt(tf.nn.weighted_moments(x, axes=[0], frequency_weights=weights)[1]),
                           weighted_quantile(x, 0.75, weights=weights)[0]
                           - weighted_quantile(x, 0.25, weights=weights)[0])
    else:
        return znp.minimum(znp.std(x), (tfp.stats.percentile(x, 75) - tfp.stats.percentile(x, 25)))


@z.function(wraps='tensor')
def calc_kernel_probs(size, weights):
    if weights is not None:
        return weights / znp.sum(weights)
    else:
        return tf.broadcast_to(1 / size, shape=(tf.cast(size, tf.int32),))


class KDEHelperMixin:
    _bandwidth_methods = {
        'scott': _bandwidth_scott_KDEV1,
        'silverman': _bandwidth_silverman_KDEV1,
    }

    def _convert_init_data_weights_size(self, data, weights):
        self._original_data = data  # for copying
        if isinstance(data, ZfitData):
            if data.weights is not None:
                if weights is not None:
                    raise OverdefinedError("Cannot specify weights and use a `ZfitData` with weights.")
                else:
                    weights = data.weights

            if data.n_obs > 1:
                raise ShapeIncompatibleError(
                    f"KDE is 1 dimensional, but data {data} has {data.n_obs} observables.")
            data = z.unstack_x(data)
        shape_data = tf.shape(data)
        size = tf.cast(shape_data[0], ztypes.float)
        return data, size, weights

    def _convert_input_bandwidth(self, bandwidth, data, **kwargs):
        if bandwidth is None:
            bandwidth = 'silverman'
        # estimate bandwidth
        bandwidth_param = bandwidth
        if isinstance(bandwidth, str):
            bandwidth = self._bandwidth_methods.get(bandwidth)
            if bandwidth is None:
                raise ValueError(f"Cannot use {bandwidth} as a bandwidth method. Use a numerical value or one of"
                                 f" the defined methods: {list(self._bandwidth_methods.keys())}")
        if (not isinstance(bandwidth, ZfitParameter)) and callable(bandwidth):
            bandwidth = bandwidth(constructor=type(self), data=data, **kwargs)
        if bandwidth_param is None or bandwidth_param in (
                'adaptiveV1', 'adaptive', 'adaptive_zfit', 'adaptive_std', 'adaptive_geom'):
            bandwidth_param = -999
        else:
            bandwidth_param = bandwidth
        return bandwidth, bandwidth_param


class GaussianKDE1DimV1(KDEHelperMixin, WrapDistribution):
    _N_OBS = 1
    _bandwidth_methods = KDEHelperMixin._bandwidth_methods.copy()
    _bandwidth_methods.update({
        'adaptive': _adaptive_std_bandwidth_KDEV1,
        'isj': _bandwidth_isj_KDEV1
    })

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 bandwidth: Union[ztyping.ParamTypeInput, str] = None,
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 truncate: bool = False,
                 name: str = "GaussianKDE1DimV1"):
        r"""EXPERIMENTAL, `FEEDBACK WELCOME <https://github.com/zfit/zfit/issues/new?assignees=&labels=&template=other.md&title=>`_
        Exact, one dimensional, (truncated) Kernel Density Estimation with a Gaussian Kernel.

        This implementation features an exact implementation as is preferably used for smaller (max. ~ a few thousand
        points) data sets. For larger data sets, methods such as :py:class: that bin the dataset may be more efficient
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        .. math::

            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        where the kernel in this case is a (truncated) Gaussian

        .. math::
            K = \exp \Big(\frac{(x - x_i)^2}{\sigma^2}\Big)


        The bandwidth of the kernel can be estimated in different ways. It can either be a global bandwidth,
        corresponding to a single value, or a local bandwidth, each corresponding to one data point.

        **Usage**

        The KDE can be instantiated by using a numpy-like data sample, preferably a `zfit.Data` object. This
        will be used as the mu of the kernels. The bandwidth can either be given as a parameter (with length
        1 for a global one or length equal to the data size) - a rather advanced concept for methods such as
        cross validation - or determined from the data automatically, either through a simple method like
        scott or silverman rule of thumbs or through an iterative, adaptive method.

        Examples
        --------

        .. code-block:: python

            # generate some example kernels
            size = 150
            data = np.random.normal(size=size, loc=2, scale=3)

            limits = (-15, 5)
            obs = zfit.Space("obs1", limits=limits)
            kde_silverman = zfit.pdf.GaussianKDE1DimV1(data=data, obs=obs, bandwidth='silverman')

            # for a better smoothing of the kernels, use an adaptive approach
            kde = zfit.pdf.GaussianKDE1DimV1(data=data, obs=obs, bandwidth='adaptive')


        Args:
            data: 1-D Tensor-like. The positions of the `kernel`, the :math:`x_i`. Determines how many kernels will be created.
            bandwidth: Bandwidth of the kernel. Valid options are {'silverman', 'scott', 'adaptive'} or a numerical.
                If a numerical is given, it as to be broadcastable to the batch and event shape of the distribution.
                A scalar or a `zfit.Parameter` will simply broadcast to `data` for a 1-D distribution.

            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            truncate: If a truncated Gaussian kernel should be used with the limits given by the `obs` lower and
                upper limits. This can cause NaNs in case datapoints are outside of the limits.
            name: Name of the PDF
        """
        original_data = data
        data, size, weights = self._convert_init_data_weights_size(data, weights)

        bandwidth, bandwidth_param = self._convert_input_bandwidth(bandwidth=bandwidth, data=data, truncate=truncate,
                                                                   name=name, obs=obs, weights=weights)
        params = {'bandwidth': bandwidth_param}

        probs = calc_kernel_probs(size, weights)
        categorical = tfd.Categorical(probs=probs)  # no grad -> no need to recreate

        # create distribution factory
        if truncate:
            if not isinstance(obs, ZfitSpace):
                raise ValueError(f"`obs` has to be a `ZfitSpace` if `truncated` is True.")
            inside = obs.inside(data)
            all_inside = znp.all(inside)
            tf.debugging.assert_equal(all_inside, True, message="Not all data points are inside the limits but"
                                                                " a truncate kernel was chosen.")

            def kernel_factory():
                return tfp.distributions.TruncatedNormal(loc=self._data, scale=self._bandwidth,
                                                         low=self.space.rect_lower,
                                                         high=self.space.rect_upper)
        else:
            def kernel_factory():
                return tfp.distributions.Normal(loc=self._data, scale=self._bandwidth)

        def dist_kwargs():
            return dict(mixture_distribution=categorical,
                        components_distribution=kernel_factory())

        distribution = tfd.MixtureSameFamily

        super().__init__(obs=obs,
                         params=params,
                         dist_params={},
                         dist_kwargs=dist_kwargs,
                         distribution=distribution,
                         name=name)

        self._data_weights = weights
        self._bandwidth = bandwidth
        self._data = data
        self._original_data = original_data  # for copying
        self._truncate = truncate


class ExactKDE1Dim(KDEHelperMixin, WrapDistribution):
    _bandwidth_methods = KDEHelperMixin._bandwidth_methods.copy()
    _bandwidth_methods.update({
        'adaptive_geom': _adaptive_geom_bandwidth_KDEV1,
        'adaptive_std': _adaptive_std_bandwidth_KDEV1,
        'adaptive_zfit': _adaptive_zfit_bandwidth_KDEV1,
        'isj': _bandwidth_isj_KDEV1
    })

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel=tfd.Normal,
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "ExactKDE1DimV1"):
        r"""
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.
        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        Args:
            data: 1-D Tensor-like.
            bandwidth: Bandwidth of the kernel. Valid options are {'silverman', 'scott', 'adaptiveV1'} or a numerical.
                If a numerical is given, it as to be broadcastable to the batch and event shape of the distribution.
                A scalar or a `zfit.Parameter` will simply broadcast to `data` for a 1-D distribution.
            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            name: Name of the PDF
        """
        if kernel is None:
            kernel = tfd.Normal

        data, size, weights = self._convert_init_data_weights_size(data, weights)

        bandwidth, bandwidth_param = self._convert_input_bandwidth(bandwidth=bandwidth, data=data,
                                                                   name=name, obs=obs, weights=weights)

        original_data = data
        self._original_data = original_data  # for copying

        def components_distribution_generator(
                loc, scale):
            return tfd.Independent(kernel(loc=loc, scale=scale))

        self._data = data
        self._bandwidth = bandwidth
        self._kernel = kernel
        self._weights = weights

        probs = calc_kernel_probs(size, weights)

        mixture_distribution = tfd.Categorical(probs=probs)
        components_distribution = components_distribution_generator(loc=self._data, scale=self._bandwidth)

        def dist_kwargs():
            return dict(mixture_distribution=mixture_distribution,
                        components_distribution=components_distribution)

        distribution = tfd.MixtureSameFamily

        params = {'bandwidth': bandwidth_param}

        super().__init__(obs=obs,
                         params=params,
                         dist_params={},
                         dist_kwargs=dist_kwargs,
                         distribution=distribution,
                         name=name)


class GridKDE1Dim(KDEHelperMixin, WrapDistribution):
    _N_OBS = 1
    _bandwidth_methods = KDEHelperMixin._bandwidth_methods.copy()
    _bandwidth_methods.update({
        'adaptive_geom': _adaptive_geom_bandwidth_grid_KDEV1,
        'adaptive_zfit': _adaptive_zfit_bandwidth_grid_KDEV1,

        # 'adaptive_std': _adaptive_std_bandwidth_grid_KDEV1,
    })

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel=None,
                 num_grid_points=None,
                 binning_method=None,
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "GridKDE1DimV1"):
        r"""
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.
        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        Args:
            data: 1-D Tensor-like.
            bandwidth: Bandwidth of the kernel. Valid options are {'silverman', 'scott', 'adaptiveV1'} or a numerical.
                If a numerical is given, it as to be broadcastable to the batch and event shape of the distribution.
                A scalar or a `zfit.Parameter` will simply broadcast to `data` for a 1-D distribution.
            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            name: Name of the PDF
        """
        if kernel is None:
            kernel = tfd.Normal
        if binning_method is None:
            binning_method = 'linear'
        if num_grid_points is None:
            num_grid_points = 1024
        if bandwidth == 'isj':
            raise ValueError("isj not supported in GridKDE, use directly 'KDE1DimISJV1'")
        if bandwidth == 'adaptive_std':
            raise ValueError("adaptive_std not supported in GridKDE due to very bad results. This is maybe caused"
                             " by an issue regarding weights of the underlaying implementation.")

        data, size, weights = self._convert_init_data_weights_size(data, weights)

        original_data = data
        self._original_data = original_data  # for copying

        def components_distribution_generator(
                loc, scale):
            return tfd.Independent(kernel(loc=loc, scale=scale))

        if num_grid_points is not None:
            num_grid_points = tf.minimum(tf.cast(size, ztypes.int), tf.cast(num_grid_points, ztypes.int))
        self._num_grid_points = num_grid_points
        self._binning_method = binning_method
        self._data = data
        self._grid = binning_util.generate_1d_grid(self._data, num_grid_points=self._num_grid_points)

        bandwidth, bandwidth_param = self._convert_input_bandwidth(bandwidth=bandwidth, data=data,
                                                                   binning_method=binning_method,
                                                                   num_grid_points=num_grid_points,
                                                                   name=name, obs=obs, weights=weights)

        self._bandwidth = bandwidth
        self._kernel = kernel
        self._weights = weights

        self._grid_data = binning_util.bin_1d(self._binning_method, self._data, self._grid, self._weights)

        mixture_distribution = tfd.Categorical(probs=self._grid_data)
        components_distribution = components_distribution_generator(loc=self._grid, scale=self._bandwidth)

        def dist_kwargs():
            return dict(mixture_distribution=mixture_distribution,
                        components_distribution=components_distribution)

        distribution = tfd.MixtureSameFamily

        params = {'bandwidth': bandwidth_param}

        super().__init__(obs=obs,
                         params=params,
                         dist_params={},
                         dist_kwargs=dist_kwargs,
                         distribution=distribution,
                         name=name)


class KDE1DimFFTV1(KDEHelperMixin, BasePDF):
    _N_OBS = 1

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel=None,
                 support=None,
                 num_grid_points: Optional[int] = None,
                 binning_method=None,
                 fft_method=None,
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KDE1DimFFTV1"):
        r"""Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        It is computed by using a convolution of the data with the kernels evaluated at fixed grid points and then
        interpolating between this points to get an estimate for x.

        Args:
            data: 1-D Tensor-like.
            bandwidth: Bandwidth of the kernel. Valid options are {'silverman', 'scott', 'adaptiveV1'} or a numerical.
                If a numerical is given, it as to be broadcastable to the batch and event shape of the distribution.
                A scalar or a `zfit.Parameter` will simply broadcast to `data` for a 1-D distribution.
            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            name: Name of the PDF
        """
        if isinstance(bandwidth, ZfitParameter):
            raise TypeError(f"bandwidth cannot be a Parameter for the FFT KDE.")
        if num_grid_points is None:
            num_grid_points = 1024
        if binning_method is None:
            binning_method = 'linear'
        if fft_method is None:
            fft_method = 'conv1d'
        if kernel is None:
            kernel = tfd.Normal
        data, size, weights = self._convert_init_data_weights_size(data, weights)
        bandwidth, bandwidth_param = self._convert_input_bandwidth(bandwidth=bandwidth, data=data,
                                                                   name=name, obs=obs, weights=weights)
        num_grid_points = tf.minimum(tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int))
        self._num_grid_points = num_grid_points
        self._binning_method = binning_method
        self._fft_method = fft_method
        self._data = data

        self._bandwidth = bandwidth

        params = {'bandwidth': self._bandwidth}
        super().__init__(obs=obs, name=name, params=params)
        self._kernel = kernel
        self._weights = weights
        if support is None:
            area = znp.reshape(self.space.area(), ())
            if area is not None:
                support = area
        self._support = support
        self._grid = None
        self._grid_data = None

        self._grid = binning_util.generate_1d_grid(self._data, num_grid_points=self._num_grid_points)
        self._grid_data = binning_util.bin_1d(self._binning_method, self._data, self._grid, self._weights)
        self._grid_estimations = convolution_util.convolve_1d_data_with_kernel(
            self._kernel, self._bandwidth, self._grid_data, self._grid, self._support, self._fft_method)

    def _unnormalized_pdf(self, x):

        x = z.unstack_x(x)
        x_min = self._grid[0]
        x_max = self._grid[-1]

        value = tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_estimations)
        value.set_shape(x.shape)
        return value


class KDE1DimISJV1(KDEHelperMixin, BasePDF):
    _N_OBS = 1

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 num_grid_points: Optional[int] = None,
                 binning_method=None,
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KDE1DimISJV1"):
        r"""Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        .. math::

            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        The bandwidth is computed by using a trick described in a paper by Botev et al. that uses the fact,
        that the Kernel Density Estimation
        with a Gaussian Kernel is a solution to the Heat Equation.

        Args:
            data: 1-D Tensor-like.
            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            name: Name of the PDF
        """
        if num_grid_points is None:
            num_grid_points = 1024
        if binning_method is None:
            binning_method = 'linear'

        data, size, weights = self._convert_init_data_weights_size(data, weights)
        num_grid_points = tf.minimum(tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int))
        self._num_grid_points = num_grid_points
        self._binning_method = binning_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._weights = weights
        self._grid = None
        self._grid_data = None

        self._bandwidth, self._grid_estimations, self._grid = isj_util.calculate_bandwidth_and_density(
            self._data, self._num_grid_points, self._binning_method, self._weights)

        params = {}
        super().__init__(obs=obs, name=name, params=params)

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        x_min = self._grid[0]
        x_max = self._grid[-1]

        value = tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_estimations)
        value.set_shape(x.shape)
        return value
