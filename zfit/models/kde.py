#  Copyright (c) 2020 zfit
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd

from .dist_tfp import WrapDistribution
from .. import z, ztypes
from ..core.interfaces import ZfitData
from ..util import ztyping
from ..util.exception import OverdefinedError, ShapeIncompatibleError


def bandwidth_rule_of_thumb(data, factor=0.9):
    return tf.math.reduce_std(data) * tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.) * factor


def bandwidth_silverman(data):
    return bandwidth_rule_of_thumb(data=data, factor=0.9)


def bandwidth_scott(data):
    return bandwidth_rule_of_thumb(data=data, factor=1.059)


def bandwidth_adaptiveV1(data, func):
    from .. import run
    run.assert_executing_eagerly()
    data = z.convert_to_tensor(data)
    bandwidth = z.sqrt(tf.math.reduce_std(data) / func(data))
    bandwidth *= tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.) * 1.059
    return bandwidth


def _adaptive_bandwidth_KDEV1(constructor, obs, data, weights, name):
    kde_silverman = constructor(obs=obs, data=data, bandwidth='silverman', weights=weights,
                                name=f"INTERNAL_{name}")
    return bandwidth_adaptiveV1(data=data, func=kde_silverman.pdf)


def _bandwidth_scott_KDEV1(data, *_, **__):
    return bandwidth_scott(data)


def _bandwidth_silverman_KDEV1(data, *_, **__):
    return bandwidth_silverman(data)


class GaussianKDE1DimExactV1(WrapDistribution):
    _N_OBS = 1

    _bandwidth_methods = {
        'scott': _bandwidth_scott_KDEV1,
        'silverman': _bandwidth_silverman_KDEV1,
        'adaptiveV1': _adaptive_bandwidth_KDEV1
    }

    def __init__(self, obs: ztyping.ObsTypeInput, data: ztyping.ParamTypeInput,
                 bandwidth: ztyping.ParamTypeInput = None,
                 weights: Union[None, np.ndarray, tf.Tensor] = None, name: str = "GaussianKDE1DimV1"):
        r"""One dimensional Kernel Density Estimation with a Gaussian Kernel.

        Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        .. math::

            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        where the kernel in this case is a Gaussian

        .. math::
            K = \exp \Big(\frac{(x - x_i)^2}{\sigma^2}\Big)


        The bandwidth of the kernel can be estimated in different ways. It can either be a global bandwidth,
        corresponding to a single value, or a local bandwidth, each corresponding to one data point

        Args:
            data: 1-D Tensor-like. The positions of the `kernel`, the :math:`x_i`. Determines how many kernels will be created.
            bandwidth: Broadcastable to the batch and event shape of the distribution. A scalar will simply broadcast
                to `data` for a 1-D distribution.
            obs: Observables
            weights: Weights of each `data`, can be None or Tensor-like with shape compatible with `data`
            name: Name of the PDF
        """
        if bandwidth is None:
            bandwidth = 'silverman'

        if isinstance(data, ZfitData):
            if data.weights is not None:
                if weights is not None:
                    raise OverdefinedError("Cannot specify weights and use a `ZfitData` with weights.")
                else:
                    weights = data.weights

            if data.n_obs > 1:
                raise ShapeIncompatibleError(f"KDE is 1 dimensional, but data {data} has {data.n_obs} observables.")
            data = z.unstack_x(data)

        # create fraction for the sum
        shape_data = tf.shape(data)
        size = tf.cast(shape_data[0], dtype=ztypes.float)
        if weights is not None:
            probs = weights / tf.reduce_sum(weights)
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
                                  name=f"INTERNAL_{name}")

        bandwidth_param = -999 if bandwidth_param == 'adaptiveV1' else bandwidth  # TODO: multiparam for bandwidth?

        params = {'bandwidth': bandwidth_param}

        # create distribution factory
        def kernel_factory():
            return tfp.distributions.Normal(loc=self._data, scale=self._bandwidth)

        dist_kwargs = lambda: dict(mixture_distribution=categorical,
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
