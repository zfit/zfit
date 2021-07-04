#  Copyright (c) 2021 zfit
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd

import zfit.z.numpy as znp

from .. import z
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitData, ZfitSpace
from ..settings import ztypes
from ..util import binning as binning_util
from ..util import convolution as convolution_util
from ..util import improved_sheather_jones as isj_util
from ..util import ztyping
from ..util.exception import OverdefinedError, ShapeIncompatibleError
from .dist_tfp import WrapDistribution


def bandwidth_rule_of_thumb(data, factor=0.9):
    return min_std_or_iqr(data) * tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.) * factor


def bandwidth_silverman(data):
    return bandwidth_rule_of_thumb(data=data, factor=0.9)


def bandwidth_scott(data):
    return bandwidth_rule_of_thumb(data=data, factor=1.059)


def bandwidth_isj(data):
    return isj_util.calculate_bandwidth(data, num_grid_points=1024, binning_method='linear', weights=None)


def bandwidth_adaptiveV1(data, func):
    data = z.convert_to_tensor(data)
    bandwidth = z.sqrt(min_std_or_iqr(data) / func(data))
    bandwidth *= tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.) * 1.059
    return bandwidth


def _adaptive_bandwidth_KDEV1(constructor, obs, data, weights, name, truncate):
    kde_silverman = constructor(obs=obs, data=data, bandwidth='silverman', weights=weights,
                                name=f"INTERNAL_{name}", truncate=truncate)
    return bandwidth_adaptiveV1(data=data, func=kde_silverman.pdf)


def _bandwidth_scott_KDEV1(data, *_, **__):
    return bandwidth_scott(data)


def _bandwidth_silverman_KDEV1(data, *_, **__):
    return bandwidth_silverman(data)


def _bandwidth_isj_KDEV1(data, *_, **__):
    return bandwidth_isj(data)


@z.function(wraps='tensor')
def min_std_or_iqr(x):
    return znp.minimum(znp.std(x), (tfp.stats.percentile(x, 75) - tfp.stats.percentile(x, 25)))


class GaussianKDE1DimV1(WrapDistribution):
    _N_OBS = 1

    _bandwidth_methods = {
        'scott': _bandwidth_scott_KDEV1,
        'silverman': _bandwidth_silverman_KDEV1,
        'adaptiveV1': _adaptive_bandwidth_KDEV1,
        'adaptive': _adaptive_bandwidth_KDEV1,
        'isj': _bandwidth_isj_KDEV1
    }

    def __init__(self, obs: ztyping.ObsTypeInput, data: ztyping.ParamTypeInput,
                 bandwidth: Union[ztyping.ParamTypeInput, str] = None,
                 weights: Union[None, np.ndarray, tf.Tensor] = None, truncate: bool = False,
                 name: str = "GaussianKDE1DimV1"):
        r"""EXPERIMENTAL, `FEEDBACK WELCOME <https://github.com/zfit/zfit/issues/new?assignees=&labels=&template=other.md&title=>`_
        One dimensional, (truncated) Kernel Density Estimation with a Gaussian Kernel.

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
            kde_silverman = zfit.pdf.GaussianKDE1DimV1(data=data, obs=obs)

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
        if bandwidth is None:
            bandwidth = 'silverman'

        original_data = data

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

        # create fraction for the sum
        shape_data = tf.shape(data)
        size = tf.cast(shape_data[0], dtype=ztypes.float)
        if weights is not None:
            probs = weights / znp.sum(weights)
        else:
            probs = tf.broadcast_to(1 / size, shape=(tf.cast(size, tf.int32),))
        categorical = tfd.Categorical(probs=probs)  # no grad -> no need to recreate

        # estimate bandwidth
        bandwidth_param = bandwidth
        if isinstance(bandwidth, str):
            bw_method = self._bandwidth_methods.get(bandwidth)
            if bw_method is None:
                raise ValueError(f"Cannot use {bandwidth} as a bandwidth method. Use a numerical value or one of"
                                 f" the defined methods: {list(self._bandwidth_methods.keys())}")
            bandwidth = bw_method(constructor=type(self), obs=obs, data=data, weights=weights,
                                  name=f"INTERNAL_{name}", truncate=truncate)

        bandwidth_param = -999 if bandwidth_param in (
            'adaptiveV1', 'adaptive') else bandwidth  # TODO: multiparam for bandwidth?

        params = {'bandwidth': bandwidth_param}

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
        self._original_data = original_data
        self._truncate = truncate


class KDE1DimV1(WrapDistribution):
    _N_OBS = 1

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel=tfd.Normal,
                 support=None,
                 use_grid=False,
                 num_grid_points=1024,
                 binning_method='linear',
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KDE1DimV1"):
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

        def components_distribution_generator(
                loc, scale):
            return tfd.Independent(kernel(loc=loc, scale=scale))

        self._num_grid_points = tf.minimum(tf.cast(size, ztypes.int),
                                           tf.constant(num_grid_points, ztypes.int))
        self._binning_method = binning_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._bandwidth = tf.convert_to_tensor(bandwidth, ztypes.float)
        self._kernel = kernel
        self._weights = weights
        self._grid = None
        self._grid_data = None

        if use_grid:
            self._grid = binning_util.generate_1d_grid(self._data, num_grid_points=self._num_grid_points)
            self._grid_data = binning_util.bin_1d(self._binning_method, self._data, self._grid, self._weights)

            mixture_distribution = tfd.Categorical(probs=self._grid_data)
            components_distribution = components_distribution_generator(loc=self._grid, scale=self._bandwidth)

        else:

            if weights is not None:
                probs = weights / tf.reduce_sum(weights)
            else:
                probs = tf.broadcast_to(1 / size, shape=(tf.cast(size, ztypes.int),))

            mixture_distribution = tfd.Categorical(probs=probs)
            components_distribution = components_distribution_generator(loc=self._data, scale=self._bandwidth)

        def dist_kwargs():
            return dict(mixture_distribution=mixture_distribution,
                        components_distribution=components_distribution)

        distribution = tfd.MixtureSameFamily

        params = {'bandwidth': self._bandwidth}

        super().__init__(obs=obs,
                         params=params,
                         dist_params={},
                         dist_kwargs=dist_kwargs,
                         distribution=distribution,
                         name=name)


class KDE1DimFFTV1(BasePDF):
    _N_OBS = 1

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 kernel=tfd.Normal,
                 support=None,
                 num_grid_points=1024,
                 binning_method='linear',
                 fft_method='conv1d',
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KDE1DimFFTV1"):
        r"""
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.
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

        self._num_grid_points = tf.minimum(tf.cast(size, ztypes.int),
                                           tf.constant(num_grid_points, ztypes.int))
        self._binning_method = binning_method
        self._fft_method = fft_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._bandwidth = tf.convert_to_tensor(bandwidth, ztypes.float)
        self._kernel = kernel
        self._weights = weights
        self._support = support
        self._grid = None
        self._grid_data = None

        self._grid = binning_util.generate_1d_grid(self._data, num_grid_points=self._num_grid_points)
        self._grid_data = binning_util.bin_1d(self._binning_method, self._data, self._grid, self._weights)
        self._grid_estimations = convolution_util.convolve_1d_data_with_kernel(
            self._kernel, self._bandwidth, self._grid_data, self._grid, self._support, self._fft_method)

        params = {'bandwidth': self._bandwidth}
        super().__init__(obs=obs, name=name, params=params)

    def _unnormalized_pdf(self, x, norm_range=False):

        x = z.unstack_x(x)
        x_min = tf.reduce_min(self._grid)
        x_max = tf.reduce_max(self._grid)

        value = tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_estimations)
        value.set_shape(x.shape)
        return value


class KDE1DimISJV1(BasePDF):
    _N_OBS = 1

    def __init__(self,
                 obs: ztyping.ObsTypeInput,
                 data: ztyping.ParamTypeInput,
                 num_grid_points=1024,
                 binning_method='linear',
                 weights: Union[None, np.ndarray, tf.Tensor] = None,
                 name: str = "KDE1DimISJV1"):
        r"""Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        .. math::

            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        It is computed by using a trick described in a paper by Botev et al. that uses the fact,
        that the Kernel Density Estimation
        with a Gaussian Kernel is a solution to the Heat Equation.

        Args:
            data: 1-D Tensor-like.
            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            name: Name of the PDF
        """

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

        self._num_grid_points = tf.minimum(tf.cast(size, ztypes.int),
                                           tf.constant(num_grid_points, ztypes.int))
        self._binning_method = binning_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._weights = weights
        self._grid = None
        self._grid_data = None

        self._bandwidth, self._grid_estimations, self._grid = isj_util.calculate_bandwidth_and_density(
            self._data, self._num_grid_points, self._binning_method, self._weights)

        params = {}
        super().__init__(obs=obs, name=name, params=params)

    def _unnormalized_pdf(self, x, norm_range=False):

        x = z.unstack_x(x)
        x_min = tf.reduce_min(self._grid)
        x_max = tf.reduce_max(self._grid)

        return tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_estimations)
