#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Callable
from typing import Union, Optional

import pydantic

from ..serialization import Serializer, SpaceRepr

from typing import Literal

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd

import zfit.z.numpy as znp
from .dist_tfp import WrapDistribution
from .. import z
from ..core.basepdf import BasePDF
from ..core.interfaces import ZfitData, ZfitParameter, ZfitSpace
from ..core.serialmixin import SerializableMixin
from ..serialization.pdfrepr import BasePDFRepr
from ..settings import ztypes, run
from ..util import (
    binning as binning_util,
    convolution as convolution_util,
    improved_sheather_jones as isj_util,
    ztyping,
)
from ..util.exception import OverdefinedError, ShapeIncompatibleError
from ..util.ztyping import ExtendedInputType, NormInputType
from ..z.math import weighted_quantile


@z.function(wraps="tensor")
def bandwidth_rule_of_thumb(
    data: znp.array,
    weights: znp.array | None,
    factor: float | int | znp.array = None,
) -> znp.array:
    r"""Calculate the bandwidth of *data* using a rule of thumb.

    This calculates a global, single bandwidth for all kernels using a rule of thumb.
    |@doc:pdf.kde.bandwidth.explain_global| A global bandwidth
             is a single parameter that is shared amongst all kernels.
             While this is a fast and robust method,
             it is a rule of thumb approximation. Due to its global nature,
             it cannot take into account the different varying
             local densities. It uses notably the least amount of memory
             of all methods. |@docend:pdf.kde.bandwidth.explain_global|

    .. math::
      h = factor * min\left(\hat{\sigma}, \frac{IQR}{1.34}\right)\,n^{-\frac{1}{5}}

    Args:
        data: |@doc:pdf.kde.bandwidth.data| Data points to determine the bandwidth
               from. |@docend:pdf.kde.bandwidth.data|
        weights: |@doc:pdf.kde.bandwidth.weights| Weights of each event
               in *data*, can be None or Tensor-like with shape compatible
               with *data*. This will change the count of the events, whereas
               weight :math:`w_i` of :math:`x_i`. |@docend:pdf.kde.bandwidth.weights|
        factor (default: 0.9): Factor that scales the rule of thumb. Ofter used are 0.9 for
        silvermans rule of thumb or 1.059 for scotts rule of thumb.

    Returns:
        Estimated bandwidth
    """
    if factor is None:
        factor = tf.constant(0.9)
    return (
        min_std_or_iqr(data, weights)
        * tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.0)
        * factor
    )


@z.function(wraps="tensor")
def bandwidth_silverman(data, weights):
    r"""Calculate the bandwidth of *data* using silvermans rule of thumb.

    |@doc:pdf.kde.bandwidth.explain_global| A global bandwidth
             is a single parameter that is shared amongst all kernels.
             While this is a fast and robust method,
             it is a rule of thumb approximation. Due to its global nature,
             it cannot take into account the different varying
             local densities. It uses notably the least amount of memory
             of all methods. |@docend:pdf.kde.bandwidth.explain_global|

    .. math::
      h = 0.9 * min\left(\hat{\sigma}, \frac{IQR}{1.34}\right)\,n^{-\frac{1}{5}}

    Args:
        data: |@doc:pdf.kde.bandwidth.data| Data points to determine the bandwidth
               from. |@docend:pdf.kde.bandwidth.data|
        weights: |@doc:pdf.kde.bandwidth.weights| Weights of each event
               in *data*, can be None or Tensor-like with shape compatible
               with *data*. This will change the count of the events, whereas
               weight :math:`w_i` of :math:`x_i`. |@docend:pdf.kde.bandwidth.weights|

    Returns:
        Estimated bandwidth
    """
    return bandwidth_rule_of_thumb(
        data=data, weights=weights, factor=znp.array(0.9, dtype=ztypes.float)
    )


@z.function(wraps="tensor")
def bandwidth_scott(data, weights):
    r"""Calculate the bandwidth of *data* using silvermans rule of thumb.

    |@doc:pdf.kde.bandwidth.explain_global| A global bandwidth
             is a single parameter that is shared amongst all kernels.
             While this is a fast and robust method,
             it is a rule of thumb approximation. Due to its global nature,
             it cannot take into account the different varying
             local densities. It uses notably the least amount of memory
             of all methods. |@docend:pdf.kde.bandwidth.explain_global|

    .. math::
      h = 1.059 * min\left(\hat{\sigma}, \frac{IQR}{1.34}\right)\,n^{-\frac{1}{5}}

    Args:
        data: |@doc:pdf.kde.bandwidth.data| Data points to determine the bandwidth
               from. |@docend:pdf.kde.bandwidth.data|
        weights: |@doc:pdf.kde.bandwidth.weights| Weights of each event
               in *data*, can be None or Tensor-like with shape compatible
               with *data*. This will change the count of the events, whereas
               weight :math:`w_i` of :math:`x_i`. |@docend:pdf.kde.bandwidth.weights|

    Returns:
        Estimated bandwidth
    """
    return bandwidth_rule_of_thumb(
        data=data, weights=weights, factor=znp.array(1.059, dtype=ztypes.float)
    )


def bandwidth_isj(data, weights):
    r"""Calculate the bandwidth of *data* using the improved Sheather-Jones Algorithm.

    The ISJ is an adaptive kernel density estimator based on linear diffusion processes
     and an estimation for the optimal bandwidth :footcite:t:`Botev_2010`

    |@doc:pdf.kde.bandwidth.explain_global| A global bandwidth
             is a single parameter that is shared amongst all kernels.
             While this is a fast and robust method,
             it is a rule of thumb approximation. Due to its global nature,
             it cannot take into account the different varying
             local densities. It uses notably the least amount of memory
             of all methods. |@docend:pdf.kde.bandwidth.explain_global|


    .. footbibliography::

    Args:
        data: |@doc:pdf.kde.bandwidth.data| Data points to determine the bandwidth
               from. |@docend:pdf.kde.bandwidth.data|
        weights: |@doc:pdf.kde.bandwidth.weights| Weights of each event
               in *data*, can be None or Tensor-like with shape compatible
               with *data*. This will change the count of the events, whereas
               weight :math:`w_i` of :math:`x_i`. |@docend:pdf.kde.bandwidth.weights|

    Returns:
        Estimated bandwidth
    """
    return isj_util.calculate_bandwidth(
        data, num_grid_points=1024, binning_method="linear", weights=weights
    )


def bandwidth_adaptive_geomV1(data, func, weights):
    r"""Local, adaptive bandwidth estimation scaling by the geometric mean.

    The implementation follows Eq. 2 and 3 of :footcite:t:`7761150`. However,
    experimental results hint towards a non-optimal performance, which can be caused
    by a mistake in the implementation.

    |@doc:pdf.kde.bandwidth.explain_adaptive| Adaptive bandwidths are
             a way to reduce the dependence on the bandwidth parameter
             and are usually local bandwidths that take into account
             the local densities.
             Adaptive bandwidths are constructed by using an initial estimate
             of the local density in order to calculate a sensible bandwidth
             for each kernel. The initial estimator can be a kernel density
             estimation using a global bandwidth with a rule of thumb.
             The adaptive bandwidth h is obtained using this estimate, where
             usually

             .. math::

               h_{i} \propto f( x_{i} ) ^ {-1/2}

             Estimates can still differ in the overall scaling of this
             bandwidth. |@docend:pdf.kde.bandwidth.explain_adaptive|

    |@doc:pdf.kde.bandwidth.explain_local| A local bandwidth
             means that each kernel :math:`i` has a different bandwidth.
             In other words, given some data points with size n,
             we will need n bandwidth parameters.
             This is often more accurate than a global bandwidth,
             as it allows to have larger bandwiths in areas of smaller density,
             where, due to the small local sample size, we have less certainty
             over the true density while having a smaller bandwidth in denser
             populated areas.

             The accuracy comes at the cost of a longer pre-calculation to obtain
             the local bandwidth (there are different methods available), an
             increased runtime and - most importantly - a peak memory usage.

             This can be especially costly for a large number (> few thousand) of
             kernels and/or evaluating on large datasets (> 10'000). |@docend:pdf.kde.bandwidth.explain_local|


    .. footbibliography::

    Args:
        data: |@doc:pdf.kde.bandwidth.data| Data points to determine the bandwidth
               from. |@docend:pdf.kde.bandwidth.data|
        weights: |@doc:pdf.kde.bandwidth.weights| Weights of each event
               in *data*, can be None or Tensor-like with shape compatible
               with *data*. This will change the count of the events, whereas
               weight :math:`w_i` of :math:`x_i`. |@docend:pdf.kde.bandwidth.weights|

    Returns:
        Estimated bandwidth of size data
    """
    data = z.convert_to_tensor(data)
    if weights is not None:
        n = znp.sum(weights)
    else:
        n = tf.cast(tf.shape(data)[0], ztypes.float)
    probs = func(data)
    lambda_i = 1 / znp.sqrt(
        probs / z.math.reduce_geometric_mean(probs, weights=weights)
    )

    return lambda_i * n ** (-1.0 / 5.0) * min_std_or_iqr(data, weights)


def bandwidth_adaptive_zfitV1(data, func, weights) -> znp.array:
    r"""(Naive) Local, adaptive bandwidth estimation using a normalized scaling.

    This approach is an ad-hoc scaling. It works well but is not found in any paper.

    |@doc:pdf.kde.bandwidth.explain_adaptive| Adaptive bandwidths are
             a way to reduce the dependence on the bandwidth parameter
             and are usually local bandwidths that take into account
             the local densities.
             Adaptive bandwidths are constructed by using an initial estimate
             of the local density in order to calculate a sensible bandwidth
             for each kernel. The initial estimator can be a kernel density
             estimation using a global bandwidth with a rule of thumb.
             The adaptive bandwidth h is obtained using this estimate, where
             usually

             .. math::

               h_{i} \propto f( x_{i} ) ^ {-1/2}

             Estimates can still differ in the overall scaling of this
             bandwidth. |@docend:pdf.kde.bandwidth.explain_adaptive|

    |@doc:pdf.kde.bandwidth.explain_local| A local bandwidth
             means that each kernel :math:`i` has a different bandwidth.
             In other words, given some data points with size n,
             we will need n bandwidth parameters.
             This is often more accurate than a global bandwidth,
             as it allows to have larger bandwiths in areas of smaller density,
             where, due to the small local sample size, we have less certainty
             over the true density while having a smaller bandwidth in denser
             populated areas.

             The accuracy comes at the cost of a longer pre-calculation to obtain
             the local bandwidth (there are different methods available), an
             increased runtime and - most importantly - a peak memory usage.

             This can be especially costly for a large number (> few thousand) of
             kernels and/or evaluating on large datasets (> 10'000). |@docend:pdf.kde.bandwidth.explain_local|


    .. footbibliography::

    Args:
        data: |@doc:pdf.kde.bandwidth.data| Data points to determine the bandwidth
               from. |@docend:pdf.kde.bandwidth.data|
        weights: |@doc:pdf.kde.bandwidth.weights| Weights of each event
               in *data*, can be None or Tensor-like with shape compatible
               with *data*. This will change the count of the events, whereas
               weight :math:`w_i` of :math:`x_i`. |@docend:pdf.kde.bandwidth.weights|

    Returns:
        Estimated bandwidth array of same size as data
    """
    data = z.convert_to_tensor(data)
    probs = func(data)
    estimate = bandwidth_scott(data, weights=weights)
    factor = znp.sqrt(probs) / znp.mean(znp.sqrt(probs))
    return estimate / factor


def bandwidth_adaptive_stdV1(data, func, weights):
    r"""Local, adaptive bandwidth estimation scaling by the std of the data.

    The implementation follows Eq. 2 and 3 of :footcite:t:`Cranmer_2001`. However,
    experimental results hint towards a non-optimal performance, which can be caused
    by a mistake in the implementation.

    |@doc:pdf.kde.bandwidth.explain_adaptive| Adaptive bandwidths are
             a way to reduce the dependence on the bandwidth parameter
             and are usually local bandwidths that take into account
             the local densities.
             Adaptive bandwidths are constructed by using an initial estimate
             of the local density in order to calculate a sensible bandwidth
             for each kernel. The initial estimator can be a kernel density
             estimation using a global bandwidth with a rule of thumb.
             The adaptive bandwidth h is obtained using this estimate, where
             usually

             .. math::

               h_{i} \propto f( x_{i} ) ^ {-1/2}

             Estimates can still differ in the overall scaling of this
             bandwidth. |@docend:pdf.kde.bandwidth.explain_adaptive|

    |@doc:pdf.kde.bandwidth.explain_local| A local bandwidth
             means that each kernel :math:`i` has a different bandwidth.
             In other words, given some data points with size n,
             we will need n bandwidth parameters.
             This is often more accurate than a global bandwidth,
             as it allows to have larger bandwiths in areas of smaller density,
             where, due to the small local sample size, we have less certainty
             over the true density while having a smaller bandwidth in denser
             populated areas.

             The accuracy comes at the cost of a longer pre-calculation to obtain
             the local bandwidth (there are different methods available), an
             increased runtime and - most importantly - a peak memory usage.

             This can be especially costly for a large number (> few thousand) of
             kernels and/or evaluating on large datasets (> 10'000). |@docend:pdf.kde.bandwidth.explain_local|


    .. footbibliography::

    Args:
        data: |@doc:pdf.kde.bandwidth.data| Data points to determine the bandwidth
               from. |@docend:pdf.kde.bandwidth.data|
        weights: |@doc:pdf.kde.bandwidth.weights| Weights of each event
               in *data*, can be None or Tensor-like with shape compatible
               with *data*. This will change the count of the events, whereas
               weight :math:`w_i` of :math:`x_i`. |@docend:pdf.kde.bandwidth.weights|

    Returns:
        Estimated bandwidth array of same size as data
    """
    data = z.convert_to_tensor(data)
    if weights is not None:
        n = znp.sum(weights)
    else:
        n = tf.cast(tf.shape(data)[0], ztypes.float)
    probs = func(data)
    divisor = min_std_or_iqr(data, weights)
    bandwidth = z.sqrt(divisor / probs)
    bandwidth *= tf.cast(n, ztypes.float) ** (-1.0 / 5.0) * 1.059
    return bandwidth


def adaptive_factory(func, grid):
    if grid:

        def adaptive(constructor, data, **kwargs):
            kwargs.pop("name", None)
            kde_silverman = constructor(
                bandwidth="silverman",
                data=data,
                name="INTERNAL_for_adaptive_kde",
                **kwargs,
            )
            grid = kde_silverman._grid
            weights = kde_silverman._grid_data
            return func(
                data=grid,
                func=kde_silverman.pdf,
                weights=weights * tf.cast(tf.shape(data)[0], ztypes.float),
            )

    else:

        def adaptive(constructor, data, weights, **kwargs):
            kwargs.pop("name", None)
            kde_silverman = constructor(
                bandwidth="silverman",
                data=data,
                name="INTERNAL_for_adaptive_kde",
                **kwargs,
            )
            return func(data=data, func=kde_silverman.pdf, weights=weights)

    return adaptive


_adaptive_geom_bandwidth_grid_KDEV1 = adaptive_factory(
    bandwidth_adaptive_geomV1, grid=True
)
_adaptive_geom_bandwidth_KDEV1 = adaptive_factory(bandwidth_adaptive_geomV1, grid=False)

_adaptive_std_bandwidth_grid_KDEV1 = adaptive_factory(
    bandwidth_adaptive_stdV1, grid=True
)
_adaptive_std_bandwidth_KDEV1 = adaptive_factory(bandwidth_adaptive_stdV1, grid=False)

_adaptive_zfit_bandwidth_grid_KDEV1 = adaptive_factory(
    bandwidth_adaptive_zfitV1, grid=True
)
_adaptive_zfit_bandwidth_KDEV1 = adaptive_factory(bandwidth_adaptive_zfitV1, grid=False)


def _bandwidth_scott_KDEV1(data, weights, *_, **__):
    return bandwidth_scott(
        data,
        weights=weights,
    )


def _bandwidth_silverman_KDEV1(data, weights, *_, **__):
    return bandwidth_silverman(
        data,
        weights=weights,
    )


def _bandwidth_isj_KDEV1(data, weights, *_, **__):
    return bandwidth_isj(data, weights=weights)


def check_bw_grid_shapes(bandwidth, grid=None, n_grid=None):
    if run.executing_eagerly() and bw_is_arraylike(bandwidth, allow1d=False):
        n_grid = grid.shape[0] if grid is not None else n_grid
        if n_grid is None:
            raise ValueError("Either the grid or n_grid must be given.")
        if bandwidth.shape[0] != n_grid:
            raise ShapeIncompatibleError(
                "The bandwidth array must have the same length as the grid"
            )


@z.function(wraps="tensor")
def min_std_or_iqr(x, weights):
    if weights is not None:
        return znp.minimum(
            znp.sqrt(tf.nn.weighted_moments(x, axes=[0], frequency_weights=weights)[1]),
            weighted_quantile(x, 0.75, weights=weights)[0]
            - weighted_quantile(x, 0.25, weights=weights)[0],
        )
    else:
        return znp.minimum(
            znp.std(x), (tfp.stats.percentile(x, 75) - tfp.stats.percentile(x, 25))
        )


@z.function(wraps="tensor")
def calc_kernel_probs(size, weights):
    if weights is not None:
        return weights / znp.sum(weights)
    else:
        return tf.broadcast_to(1 / size, shape=(tf.cast(size, tf.int32),))


class KDEHelper:
    _bandwidth_methods = {
        "scott": _bandwidth_scott_KDEV1,
        "silverman": _bandwidth_silverman_KDEV1,
    }
    _default_padding = False
    _default_num_grid_points = 1024

    def _convert_init_data_weights_size(
        self, data, weights, padding, limits=None, bandwidth=None
    ):
        self._original_data = data  # for copying
        if isinstance(data, ZfitData):
            if data.weights is not None:
                if weights is not None:
                    raise OverdefinedError(
                        "Cannot specify weights and use a `ZfitData` with weights."
                    )
                else:
                    weights = data.weights

            if data.n_obs > 1:
                raise ShapeIncompatibleError(
                    f"KDE is 1 dimensional, but data {data} has {data.n_obs} observables."
                )
            data = z.unstack_x(data)

        if callable(padding):
            data, weights, bandwidth = padding(
                data=data, weights=weights, limits=limits, bandwidth=bandwidth
            )
        elif padding is not False:
            data, weights, bandwidth = padreflect_data_weights_1dim(
                data, weights=weights, mode=padding, limits=limits, bandwidth=bandwidth
            )
        shape_data = tf.shape(data)
        size = tf.cast(shape_data[0], ztypes.float)
        return data, size, weights, bandwidth

    def _convert_input_bandwidth(self, bandwidth, data, **kwargs):
        if bandwidth is None:
            bandwidth = "silverman"
        # estimate bandwidth
        bandwidth_param = bandwidth
        if isinstance(bandwidth, str):
            bandwidth = self._bandwidth_methods.get(bandwidth)
            if bandwidth is None:
                raise ValueError(
                    f"Cannot use {bandwidth} as a bandwidth method. Use a numerical value or one of"
                    f" the defined methods: {list(self._bandwidth_methods.keys())}"
                )
        if (not isinstance(bandwidth, ZfitParameter)) and callable(bandwidth):
            bandwidth = bandwidth(constructor=type(self), data=data, **kwargs)
        is_arraylike = bw_is_arraylike(bandwidth_param, allow1d=True)
        if (
            bandwidth_param is None
            or is_arraylike
            or bandwidth_param
            in (
                "adaptiveV1",
                "adaptive",
                "adaptive_zfit",
                "adaptive_std",
                "adaptive_geom",
            )
        ):
            bandwidth_param = -999999  # dummy value, not needed. Better solution?
        else:
            bandwidth_param = bandwidth
        if is_arraylike:
            bandwidth = znp.asarray(bandwidth)
        return bandwidth, bandwidth_param


def padreflect_data_weights_1dim(data, mode, weights=None, limits=None, bandwidth=None):
    if mode is False:
        return data, weights
    if mode is True:
        mode = znp.array(0.1)
    if not isinstance(mode, dict):
        mode = {"lowermirror": mode, "uppermirror": mode}
    for key in mode:
        if key not in ("lowermirror", "uppermirror"):
            raise ValueError(
                f"Key '{key}' is not a valid padding specification, use 'lowermirror' or 'uppermirror'"
                f" in order to mirror the data."
            )
    if limits is None:
        minimum = znp.min(data)
        maximum = znp.max(data)
    else:
        minimum = znp.array(limits[0][0])
        maximum = znp.array(limits[1][0])

    diff = maximum - minimum
    new_data = []
    new_weights = []
    bw_is_array = bw_is_arraylike(bandwidth, allow1d=False)
    if bw_is_array:
        new_bw = []

    lower = mode.get("lowermirror")
    if lower is not None:
        dx_lower = diff * lower
        lower_area = data < minimum + dx_lower
        lower_index = znp.where(lower_area)[0]
        lower_data = tf.gather(data, indices=lower_index)
        lower_data_mirrored = -lower_data + 2 * minimum
        new_data.append(lower_data_mirrored)
        if weights is not None:
            lower_weights = tf.gather(weights, indices=lower_index)
            new_weights.append(lower_weights)
        if bw_is_array:
            new_bw.append(tf.gather(bandwidth, indices=lower_index))
    new_data.append(data)
    new_weights.append(weights)
    if bw_is_array:
        new_bw.append(bandwidth)
    upper = mode.get("uppermirror")
    if upper is not None:
        dx_upper = diff * upper
        upper_area = data > maximum - dx_upper
        upper_index = znp.where(upper_area)[0]
        upper_data = tf.gather(data, indices=upper_index)
        upper_data_mirrored = -upper_data + 2 * maximum
        new_data.append(upper_data_mirrored)
        if weights is not None:
            upper_weights = tf.gather(weights, indices=upper_index)
            new_weights.append(upper_weights)
        if bw_is_array:
            new_bw.append(tf.gather(bandwidth, indices=upper_index))
    data = tf.concat(new_data, axis=0)

    if weights is not None:
        weights = tf.concat(new_weights, axis=0)
    if bw_is_array:
        bandwidth = tf.concat(new_bw, axis=0)
    return data, weights, bandwidth


class GaussianKDE1DimV1(KDEHelper, WrapDistribution):
    _N_OBS = 1
    _bandwidth_methods = KDEHelper._bandwidth_methods.copy()
    _bandwidth_methods.update(
        {"adaptive": _adaptive_std_bandwidth_KDEV1, "isj": _bandwidth_isj_KDEV1}
    )

    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        data: ztyping.ParamTypeInput,
        bandwidth: ztyping.ParamTypeInput | str = None,
        weights: None | np.ndarray | tf.Tensor = None,
        truncate: bool = False,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "GaussianKDE1DimV1",
    ):
        r"""EXPERIMENTAL, `FEEDBACK WELCOME.

        <https://github.com/zfit/zfit/issues/new?assignees=&labels=&template=other.md&title=>`_ Exact, one dimensional,
        (truncated) Kernel Density Estimation with a Gaussian Kernel.

        This implementation features an exact implementation as is preferably used for smaller (max. ~ a few thousand
        points) data sets. For larger data sets, methods such as :py:class:`~zfit.pdf.KDE1DimGrid`
        that bin the dataset may be more efficient
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
        will be used as the mean of the kernels. The bandwidth can either be given as a parameter (with length
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
            data: |@doc:pdf.kde.init.data| Data sample to approximate
             the density from. The points represent positions of the *kernel*,
             the :math:`x_i`. This is preferrably a ``ZfitData``, but can also
             be an array-like object.

             If the data has weights, they will be taken into account.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of
             :math:`K_i( x_i)`, resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.data|

            obs: |@doc:pdf.kde.init.obs| Observable space of the KDE.
             As with any other PDF, this will be used as the default *norm*, but
             does not define the domain of the PDF. Namely, this can be a smaller
             space than *data*, as long as the name of the observable match.
             Using a larger dataset is actually good practice avoiding
             bountary biases, see also :ref:`sec-boundary-bias-and-padding`. |@docend:pdf.kde.init.obs|
            bandwidth: Valid pre-defined options are {'silverman', 'scott', 'adaptive'}.
             |@doc:pdf.kde.init.bandwidth| Bandwidth of the kernel,
             often also denoted as :math:`h`. For a Gaussian kernel, this
             corresponds to *sigma*. This can be calculated using
             pre-defined options or by specifying a numerical value that is
             broadcastable to *data* -- a scalar or an array-like
             object with the same size as *data*.

             A scalar value is usually referred to as a global bandwidth while
             an array holds local bandwidths |@docend:pdf.kde.init.bandwidth|

            The bandwidth can also be a parameter, which should be used with caution. However,
            it allows to use it in cross-valitadion with a likelihood method.


            weights: |@doc:pdf.kde.init.weights| Weights of each event
             in *data*, can be None or Tensor-like with shape compatible
             with *data*. Instead of using this parameter, it is preferred
             to use a ``ZfitData`` as *data* that contains weights.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of :math:`K_i( x_i)`,
             resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.weights|
            truncate: If a truncated Gaussian kernel should be used with the limits given by the `obs` lower and
                upper limits. This can cause NaNs in case datapoints are outside the limits.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.name|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
        """
        original_data = data
        data, size, weights, _ = self._convert_init_data_weights_size(
            data,
            weights,
            padding=False,
            limits=None,
            bandwidth=bandwidth,
        )

        bandwidth, bandwidth_param = self._convert_input_bandwidth(
            bandwidth=bandwidth,
            data=data,
            truncate=truncate,
            name=name,
            obs=obs,
            weights=weights,
        )
        params = {"bandwidth": bandwidth_param}

        probs = calc_kernel_probs(size, weights)
        categorical = tfd.Categorical(probs=probs)  # no grad -> no need to recreate

        # create distribution factory
        if truncate:
            if not isinstance(obs, ZfitSpace):
                raise ValueError(
                    "`obs` has to be a `ZfitSpace` if `truncated` is True."
                )
            inside = obs.inside(data)
            all_inside = znp.all(inside)
            tf.debugging.assert_equal(
                all_inside,
                True,
                message="Not all data points are inside the limits but"
                " a truncate kernel was chosen.",
            )

            def kernel_factory():
                return tfp.distributions.TruncatedNormal(
                    loc=self._data,
                    scale=self._bandwidth,
                    low=self.space.rect_lower,
                    high=self.space.rect_upper,
                )

        else:

            def kernel_factory():
                return tfp.distributions.Normal(loc=self._data, scale=self._bandwidth)

        def dist_kwargs():
            return dict(
                mixture_distribution=categorical,
                components_distribution=kernel_factory(),
            )

        distribution = tfd.MixtureSameFamily

        super().__init__(
            obs=obs,
            params=params,
            dist_params={},
            dist_kwargs=dist_kwargs,
            distribution=distribution,
            extended=extended,
            norm=norm,
            name=name,
        )

        self._data_weights = weights
        self._bandwidth = bandwidth
        self._data = data
        self._original_data = original_data  # for copying
        self._truncate = truncate


class KDE1DimExact(KDEHelper, WrapDistribution, SerializableMixin):
    _bandwidth_methods = KDEHelper._bandwidth_methods.copy()
    _bandwidth_methods.update(
        {
            "adaptive_geom": _adaptive_geom_bandwidth_KDEV1,
            "adaptive_std": _adaptive_std_bandwidth_KDEV1,
            "adaptive_zfit": _adaptive_zfit_bandwidth_KDEV1,
            "isj": _bandwidth_isj_KDEV1,
        }
    )

    def __init__(
        self,
        data: ztyping.XTypeInput,
        *,
        obs: ztyping.ObsTypeInput | None = None,
        bandwidth: ztyping.ParamTypeInput | str | Callable | None = None,
        kernel: tfd.Distribution = None,
        padding: callable | str | bool | None = None,
        weights: np.ndarray | tf.Tensor | None = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = "ExactKDE1DimV1",
    ):
        r"""Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        For a more in-depth explanation, see also in the section about Kernel Density Estimation
        :ref:`section-exact-kdes`

        Given a sample *data* we want to estimate the

        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        This PDF features an exact implementation as is preferable for smaller (max. ~ a few thousand
        points) data sets. For larger data sets, methods such as :py:class:`~zfit.pdf.KDE1DimGrid`
        that bin the dataset may be more efficient
        Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        .. math::

            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        where the kernel in this case is a (truncated) Gaussian

        .. math::
            K = \exp \Big(\frac{(x - x_i)^2}{\sigma^2}\Big)


        The bandwidth of the kernel can be estimated in different ways. It can either be a global bandwidth,
        corresponding to a single value, or a local bandwidth, each corresponding to one data point.

        Args:
            data: |@doc:pdf.kde.init.data| Data sample to approximate
             the density from. The points represent positions of the *kernel*,
             the :math:`x_i`. This is preferrably a ``ZfitData``, but can also
             be an array-like object.

             If the data has weights, they will be taken into account.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of
             :math:`K_i( x_i)`, resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.data|

            obs: |@doc:pdf.kde.init.obs| Observable space of the KDE.
             As with any other PDF, this will be used as the default *norm*, but
             does not define the domain of the PDF. Namely, this can be a smaller
             space than *data*, as long as the name of the observable match.
             Using a larger dataset is actually good practice avoiding
             bountary biases, see also :ref:`sec-boundary-bias-and-padding`. |@docend:pdf.kde.init.obs|
            bandwidth: Valid pre-defined options are {'silverman', 'scott',
             'adaptive_zfit', 'adaptive_geom', 'adaptive_std', 'isj'}.
             |@doc:pdf.kde.init.bandwidth| Bandwidth of the kernel,
             often also denoted as :math:`h`. For a Gaussian kernel, this
             corresponds to *sigma*. This can be calculated using
             pre-defined options or by specifying a numerical value that is
             broadcastable to *data* -- a scalar or an array-like
             object with the same size as *data*.

             A scalar value is usually referred to as a global bandwidth while
             an array holds local bandwidths |@docend:pdf.kde.init.bandwidth|
             The bandwidth can also be a parameter, which should be used with caution. However,
             it allows to use it in cross-valitadion with a likelihood method.

            kernel: |@doc:pdf.kde.init.kernel| The kernel is the heart
             of the Kernel Density Estimation, which consists of the sum of
             kernels around each sample point. Therefore, a kernel should represent
             the distribution probability of a single data point as close as
             possible.

             The most widespread kernel is a Gaussian, or Normal, distribution. Due
             to the law of large numbers, the sum of many (arbitrary) random variables
             -- this is the case for most real world observable as they are the result of
             multiple consecutive random effects -- results in a Gaussian distribution.
             However, there are many cases where this assumption is not per-se true. In
             this cases an alternative kernel may offer a better choice.

             Valid choices are callables that return a
             :py:class:`~tensorflow_probability.distribution.Distribution`, such as all distributions
             that belong to the loc-scale family. |@docend:pdf.kde.init.kernel|

            padding: |@doc:pdf.kde.init.padding| KDEs have a peculiar
             weakness: the boundaries, as the outside has a zero density. This makes the KDE
             go down at the bountary as well, as the density approaches zero, no matter what the
             density inside the boundary was.

             There are two ways to circumvent this problem:

               - the best solution: providing a larger dataset than the default space the PDF is used in
               - mirroring the existing data at the boundaries, which is equivalent to a boundary condition
                 with a zero derivative. This is a padding technique and can improve the boundaries.
                 However, one important drawback of this method is to keep in mind that this will actually
                 alter the PDF *to look mirrored*. If the PDF is plotted in a larger range, this becomes
                 clear.

             Possible options are a number (default 0.1) that depicts the fraction of the overall space
             that defines the data mirrored on both sides. For example, for a space from 0 to 5, a value of
             0.3 means that all data in the region of 0 to 1.5 is taken, mirrored around 0 as well as
             all data from 3.5 to 5 and mirrored at 5. The new data will go from -1.5 to 6.5, so the
             KDE is also having a shape outside the desired range. Using it only for the range 0 to 5
             hides this.
             Using a dict, each side separately (or only a single one) can be mirrored, like ``{'lowermirror: 0.1}``
             or ``{'lowermirror: 0.2, 'uppermirror': 0.1}``.
             For more control, a callable that takes data and weights can also be used. |@docend:pdf.kde.init.padding|

            weights: |@doc:pdf.kde.init.weights| Weights of each event
             in *data*, can be None or Tensor-like with shape compatible
             with *data*. Instead of using this parameter, it is preferred
             to use a ``ZfitData`` as *data* that contains weights.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of :math:`K_i( x_i)`,
             resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.weights|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
        """
        original_init = {
            "data": data,
            "obs": obs,
            "bandwidth": bandwidth,
            "kernel": kernel,
            "padding": padding,
            "weights": weights,
            "extended": extended,
            "name": name,
        }

        if kernel is None:
            kernel = tfd.Normal

        if padding is None:
            padding = self._default_padding
        if obs is None:
            if not isinstance(data, ZfitData) or not data.space.has_limits:
                raise ValueError(
                    "obs can only be None if data is ZfitData with limits."
                )
            else:
                obs = data.space
        data, size, weights, bw = self._convert_init_data_weights_size(
            data, weights, padding=padding, limits=obs.limits, bandwidth=bandwidth
        )
        self._padding = padding
        bandwidth, bandwidth_param = self._convert_input_bandwidth(
            bandwidth=bandwidth,
            data=data,
            name=name,
            obs=obs,
            weights=weights,
            padding=False,
            kernel=kernel,
        )

        self._original_data = data  # for copying

        def components_distribution_generator(loc, scale):
            return tfd.Independent(kernel(loc=loc, scale=scale))

        self._data = data
        self._bandwidth = bandwidth
        self._kernel = kernel
        self._weights = weights

        probs = calc_kernel_probs(size, weights)

        mixture_distribution = tfd.Categorical(probs=probs)
        components_distribution = components_distribution_generator(
            loc=self._data, scale=self._bandwidth
        )

        def dist_kwargs():
            return dict(
                mixture_distribution=mixture_distribution,
                components_distribution=components_distribution,
            )

        distribution = tfd.MixtureSameFamily

        params = {"bandwidth": bandwidth_param}

        super().__init__(
            obs=obs,
            params=params,
            dist_params={},
            dist_kwargs=dist_kwargs,
            distribution=distribution,
            extended=extended,
            norm=norm,
            name=name,
        )
        self.hs3.original_init.update(original_init)


class KDE1DimExactRepr(BasePDFRepr):
    _implementation = KDE1DimExact
    hs3_type: Literal["KDE1DimExact"] = pydantic.Field("KDE1DimExact", alias="type")

    data: Union[np.ndarray, Serializer.types.DataTypeDiscriminated]
    obs: Optional[SpaceRepr] = None
    bandwidth: Optional[Union[str, float]] = None
    kernel: None = None
    padding: Optional[Union[bool, str]] = None
    weights: Optional[Union[np.ndarray, tf.Tensor]] = None
    name: Optional[str] = "KDE1DimExact"

    @pydantic.validator("kernel", pre=True)
    def validate_kernel(cls, v):
        if v is not None:
            if v != tfd.Normal:
                raise ValueError(
                    "Kernel must be None for KDE1DimExact to be serialized."
                )
            else:
                v = None
        return v

    @pydantic.root_validator(pre=True)
    def validate_all(cls, values):
        values = dict(values)
        if cls.orm_mode(values):
            for k, v in values["hs3"].original_init.items():
                values[k] = v
        return values


class KDE1DimGrid(KDEHelper, WrapDistribution, SerializableMixin):
    _N_OBS = 1
    _bandwidth_methods = KDEHelper._bandwidth_methods.copy()
    _bandwidth_methods.update(
        {
            "adaptive_geom": _adaptive_geom_bandwidth_grid_KDEV1,
            "adaptive_zfit": _adaptive_zfit_bandwidth_grid_KDEV1,
            # 'adaptive_std': _adaptive_std_bandwidth_grid_KDEV1,
        }
    )

    def __init__(
        self,
        data: ztyping.XTypeInput,
        *,
        bandwidth: ztyping.ParamTypeInput | str | Callable | None = None,
        kernel: tfd.Distribution = None,
        padding: callable | str | bool | None = None,
        num_grid_points: int | None = None,
        binning_method: str | None = None,
        obs: ztyping.ObsTypeInput | None = None,
        weights: np.ndarray | tf.Tensor | None = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "GridKDE1DimV1",
    ):
        r"""Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        For a more in-depth explanation, see also in the section about Kernel Density Estimation
        :ref:`sec-grid-kdes`

        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        Args:
            data: |@doc:pdf.kde.init.data| Data sample to approximate
             the density from. The points represent positions of the *kernel*,
             the :math:`x_i`. This is preferrably a ``ZfitData``, but can also
             be an array-like object.

             If the data has weights, they will be taken into account.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of
             :math:`K_i( x_i)`, resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.data|

            bandwidth: Valid pre-defined options are {'silverman', 'scott',
             'adaptive_zfit', 'adaptive_geom'}.
             |@doc:pdf.kde.init.bandwidth| Bandwidth of the kernel,
             often also denoted as :math:`h`. For a Gaussian kernel, this
             corresponds to *sigma*. This can be calculated using
             pre-defined options or by specifying a numerical value that is
             broadcastable to *data* -- a scalar or an array-like
             object with the same size as *data*.

             A scalar value is usually referred to as a global bandwidth while
             an array holds local bandwidths |@docend:pdf.kde.init.bandwidth|
            kernel: |@doc:pdf.kde.init.kernel| The kernel is the heart
             of the Kernel Density Estimation, which consists of the sum of
             kernels around each sample point. Therefore, a kernel should represent
             the distribution probability of a single data point as close as
             possible.

             The most widespread kernel is a Gaussian, or Normal, distribution. Due
             to the law of large numbers, the sum of many (arbitrary) random variables
             -- this is the case for most real world observable as they are the result of
             multiple consecutive random effects -- results in a Gaussian distribution.
             However, there are many cases where this assumption is not per-se true. In
             this cases an alternative kernel may offer a better choice.

             Valid choices are callables that return a
             :py:class:`~tensorflow_probability.distribution.Distribution`, such as all distributions
             that belong to the loc-scale family. |@docend:pdf.kde.init.kernel|
            padding: |@doc:pdf.kde.init.padding| KDEs have a peculiar
             weakness: the boundaries, as the outside has a zero density. This makes the KDE
             go down at the bountary as well, as the density approaches zero, no matter what the
             density inside the boundary was.

             There are two ways to circumvent this problem:

               - the best solution: providing a larger dataset than the default space the PDF is used in
               - mirroring the existing data at the boundaries, which is equivalent to a boundary condition
                 with a zero derivative. This is a padding technique and can improve the boundaries.
                 However, one important drawback of this method is to keep in mind that this will actually
                 alter the PDF *to look mirrored*. If the PDF is plotted in a larger range, this becomes
                 clear.

             Possible options are a number (default 0.1) that depicts the fraction of the overall space
             that defines the data mirrored on both sides. For example, for a space from 0 to 5, a value of
             0.3 means that all data in the region of 0 to 1.5 is taken, mirrored around 0 as well as
             all data from 3.5 to 5 and mirrored at 5. The new data will go from -1.5 to 6.5, so the
             KDE is also having a shape outside the desired range. Using it only for the range 0 to 5
             hides this.
             Using a dict, each side separately (or only a single one) can be mirrored, like ``{'lowermirror: 0.1}``
             or ``{'lowermirror: 0.2, 'uppermirror': 0.1}``.
             For more control, a callable that takes data and weights can also be used. |@docend:pdf.kde.init.padding|
            num_grid_points: |@doc:pdf.kde.init.num_grid_points| Number of points in
             the binning grid.

             The data will be binned using the *binning_method* in *num_grid_points*
             and this histogram grid will then be used as kernel points. This has the
             advantage to have a constant computational complexity independent of the data
             size.

             A number from 32 on can already yield good results, while the default is set
             to 1024, creating a fine grid. Lowering the number increases the performance
             at the cost of accuracy. |@docend:pdf.kde.init.num_grid_points|
            binning_method: |@doc:pdf.kde.init.binning_method| Method to be used for
             binning the data. Options are 'linear', 'simple'.

             The data can be binned in the usual way ('simple'), but this is less precise
             for KDEs, where we are interested in the shape of the histogram and smoothing
             it. Therefore, a better suited method, 'linear', is available.

             In normal binnig, each event (or weight) falls into the bin within the bin edges,
             while the neighbouring bins get zero counts from this event.
             In linear binning, the event is split between two bins, proportional to its
             closeness to each bin.

             The 'linear' method provides superior performance, most notably in small (~32)
             grids. |@docend:pdf.kde.init.binning_method|
            obs: |@doc:pdf.kde.init.obs| Observable space of the KDE.
             As with any other PDF, this will be used as the default *norm*, but
             does not define the domain of the PDF. Namely, this can be a smaller
             space than *data*, as long as the name of the observable match.
             Using a larger dataset is actually good practice avoiding
             bountary biases, see also :ref:`sec-boundary-bias-and-padding`. |@docend:pdf.kde.init.obs|
            weights: |@doc:pdf.kde.init.weights| Weights of each event
             in *data*, can be None or Tensor-like with shape compatible
             with *data*. Instead of using this parameter, it is preferred
             to use a ``ZfitData`` as *data* that contains weights.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of :math:`K_i( x_i)`,
             resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.weights|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
            extended: |@doc:model.init.extended| Whether the PDF is extended
                or not. If True, the PDF can be integrated over the full space
                and the integral will be 1. If False, the integral will be the
                number of events in the dataset. |@docend:model.init.extended|
        """
        original_init = {
            "data": data,
            "bandwidth": bandwidth,
            "kernel": kernel,
            "binning_method": binning_method,
            "num_grid_points": num_grid_points,
            "obs": obs,
            "weights": weights,
            "padding": padding,
            "name": name,
            "extended": extended,
            "norm": norm,
        }
        if kernel is None:
            kernel = tfd.Normal
        if binning_method is None:
            binning_method = "linear"
        if num_grid_points is None:
            num_grid_points = self._default_num_grid_points

        if isinstance(
            bandwidth, str
        ):  # numpy arrays cannot be compared with equal-> "use any, all"
            if bandwidth == "isj":
                raise ValueError(
                    "isj not supported in GridKDE, use directly 'KDE1DimISJ'"
                )
            if bandwidth == "adaptive_std":
                raise ValueError(
                    "adaptive_std not supported in GridKDE due to very bad results. This is maybe caused"
                    " by an issue regarding weights of the underlaying implementation."
                )

        if padding is None:
            padding = self._default_padding
        if obs is None:
            if not isinstance(data, ZfitData) or not data.space.has_limits:
                raise ValueError(
                    "obs can only be None if data is ZfitData with limits."
                )
            else:
                obs = data.space
        data, size, weights, _ = self._convert_init_data_weights_size(
            data, weights, padding=padding, limits=obs.limits, bandwidth=bandwidth
        )
        self._padding = padding

        self._original_data = data  # for copying

        def components_distribution_generator(loc, scale):
            return tfd.Independent(kernel(loc=loc, scale=scale))

        if num_grid_points is not None:
            num_grid_points = tf.minimum(
                tf.cast(size, ztypes.int), tf.cast(num_grid_points, ztypes.int)
            )
        self._num_grid_points = num_grid_points
        self._binning_method = binning_method
        self._data = data
        self._grid = binning_util.generate_1d_grid(
            self._data, num_grid_points=self._num_grid_points
        )

        bandwidth, bandwidth_param = self._convert_input_bandwidth(
            bandwidth=bandwidth,
            data=data,
            binning_method=binning_method,
            num_grid_points=num_grid_points,
            padding=False,
            kernel=kernel,
            name=name,
            obs=obs,
            weights=weights,
        )

        self._bandwidth = bandwidth
        self._kernel = kernel
        self._weights = weights

        self._grid_data = binning_util.bin_1d(
            self._binning_method, self._data, self._grid, self._weights
        )

        mixture_distribution = tfd.Categorical(probs=self._grid_data)

        check_bw_grid_shapes(self._bandwidth, self._grid)

        components_distribution = components_distribution_generator(
            loc=self._grid, scale=self._bandwidth
        )

        def dist_kwargs():
            return dict(
                mixture_distribution=mixture_distribution,
                components_distribution=components_distribution,
            )

        distribution = tfd.MixtureSameFamily

        params = {"bandwidth": bandwidth_param}

        super().__init__(
            obs=obs,
            params=params,
            dist_params={},
            dist_kwargs=dist_kwargs,
            distribution=distribution,
            extended=extended,
            norm=norm,
            name=name,
        )
        self.hs3.original_init.update(original_init)


def bw_is_arraylike(bw, allow1d):
    return (
        hasattr(bw, "shape")
        and bw.shape
        and len(bw.shape) > 0
        and (bw.shape[0] is None or (bw.shape[0] > 1 or allow1d))
    )


class KDE1DimGridRepr(BasePDFRepr):
    _implementation = KDE1DimGrid
    hs3_type: Literal["KDE1DimGrid"] = pydantic.Field("KDE1DimGrid", alias="type")

    data: Union[np.ndarray, Serializer.types.DataTypeDiscriminated]
    obs: Optional[SpaceRepr] = None
    bandwidth: Optional[Union[str, float]] = None
    num_grid_points: Optional[int] = None
    binning_method: Optional[str] = None
    kernel: None = None
    padding: Optional[Union[bool, str]] = None
    weights: Optional[Union[np.ndarray, tf.Tensor]] = None
    name: Optional[str] = "GridKDE1DimV1"

    @pydantic.validator("kernel", pre=True)
    def validate_kernel(cls, v):
        if v is not None:
            if v != tfd.Normal:
                raise ValueError(
                    "Kernel must be None for GridKDE1DimV1 to be serialized."
                )
            else:
                v = None
        return v

    @pydantic.root_validator(pre=True)
    def validate_all(cls, values):
        values = dict(values)
        if cls.orm_mode(values):
            for k, v in values["hs3"].original_init.items():
                values[k] = v
        return values


class KDE1DimFFT(KDEHelper, BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        data: ztyping.XTypeInput,
        *,
        obs: ztyping.ObsTypeInput | None = None,
        bandwidth: ztyping.ParamTypeInput | str | Callable | None = None,
        kernel: tfd.Distribution = None,
        num_grid_points: int | None = None,
        binning_method: str | None = None,
        support=None,
        fft_method: str | None = None,
        padding: callable | str | bool | None = None,
        weights: np.ndarray | tf.Tensor | None = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "KDE1DimFFT",
    ):
        r"""Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        For a more in-depth explanation, see also in the section about Kernel Density Estimation
        :ref:`sec-fft-kdes`

        .. math::
            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        It is computed by using a convolution of the data with the kernels evaluated at fixed grid points and then
        interpolating between this points to get an estimate for x.

        Args:
            data: |@doc:pdf.kde.init.data| Data sample to approximate
             the density from. The points represent positions of the *kernel*,
             the :math:`x_i`. This is preferrably a ``ZfitData``, but can also
             be an array-like object.

             If the data has weights, they will be taken into account.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of
             :math:`K_i( x_i)`, resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.data|
            bandwidth: |@doc:pdf.kde.init.bandwidth| Bandwidth of the kernel,
             often also denoted as :math:`h`. For a Gaussian kernel, this
             corresponds to *sigma*. This can be calculated using
             pre-defined options or by specifying a numerical value that is
             broadcastable to *data* -- a scalar or an array-like
             object with the same size as *data*.

             A scalar value is usually referred to as a global bandwidth while
             an array holds local bandwidths |@docend:pdf.kde.init.bandwidth|
            kernel: |@doc:pdf.kde.init.kernel| The kernel is the heart
             of the Kernel Density Estimation, which consists of the sum of
             kernels around each sample point. Therefore, a kernel should represent
             the distribution probability of a single data point as close as
             possible.

             The most widespread kernel is a Gaussian, or Normal, distribution. Due
             to the law of large numbers, the sum of many (arbitrary) random variables
             -- this is the case for most real world observable as they are the result of
             multiple consecutive random effects -- results in a Gaussian distribution.
             However, there are many cases where this assumption is not per-se true. In
             this cases an alternative kernel may offer a better choice.

             Valid choices are callables that return a
             :py:class:`~tensorflow_probability.distribution.Distribution`, such as all distributions
             that belong to the loc-scale family. |@docend:pdf.kde.init.kernel|
            padding: |@doc:pdf.kde.init.padding| KDEs have a peculiar
             weakness: the boundaries, as the outside has a zero density. This makes the KDE
             go down at the bountary as well, as the density approaches zero, no matter what the
             density inside the boundary was.

             There are two ways to circumvent this problem:

               - the best solution: providing a larger dataset than the default space the PDF is used in
               - mirroring the existing data at the boundaries, which is equivalent to a boundary condition
                 with a zero derivative. This is a padding technique and can improve the boundaries.
                 However, one important drawback of this method is to keep in mind that this will actually
                 alter the PDF *to look mirrored*. If the PDF is plotted in a larger range, this becomes
                 clear.

             Possible options are a number (default 0.1) that depicts the fraction of the overall space
             that defines the data mirrored on both sides. For example, for a space from 0 to 5, a value of
             0.3 means that all data in the region of 0 to 1.5 is taken, mirrored around 0 as well as
             all data from 3.5 to 5 and mirrored at 5. The new data will go from -1.5 to 6.5, so the
             KDE is also having a shape outside the desired range. Using it only for the range 0 to 5
             hides this.
             Using a dict, each side separately (or only a single one) can be mirrored, like ``{'lowermirror: 0.1}``
             or ``{'lowermirror: 0.2, 'uppermirror': 0.1}``.
             For more control, a callable that takes data and weights can also be used. |@docend:pdf.kde.init.padding|
            num_grid_points: |@doc:pdf.kde.init.num_grid_points| Number of points in
             the binning grid.

             The data will be binned using the *binning_method* in *num_grid_points*
             and this histogram grid will then be used as kernel points. This has the
             advantage to have a constant computational complexity independent of the data
             size.

             A number from 32 on can already yield good results, while the default is set
             to 1024, creating a fine grid. Lowering the number increases the performance
             at the cost of accuracy. |@docend:pdf.kde.init.num_grid_points|
            binning_method: |@doc:pdf.kde.init.binning_method| Method to be used for
             binning the data. Options are 'linear', 'simple'.

             The data can be binned in the usual way ('simple'), but this is less precise
             for KDEs, where we are interested in the shape of the histogram and smoothing
             it. Therefore, a better suited method, 'linear', is available.

             In normal binnig, each event (or weight) falls into the bin within the bin edges,
             while the neighbouring bins get zero counts from this event.
             In linear binning, the event is split between two bins, proportional to its
             closeness to each bin.

             The 'linear' method provides superior performance, most notably in small (~32)
             grids. |@docend:pdf.kde.init.binning_method|
            support:
            fft_method:
            obs: |@doc:pdf.kde.init.obs| Observable space of the KDE.
             As with any other PDF, this will be used as the default *norm*, but
             does not define the domain of the PDF. Namely, this can be a smaller
             space than *data*, as long as the name of the observable match.
             Using a larger dataset is actually good practice avoiding
             bountary biases, see also :ref:`sec-boundary-bias-and-padding`. |@docend:pdf.kde.init.obs|
            weights: |@doc:pdf.kde.init.weights| Weights of each event
             in *data*, can be None or Tensor-like with shape compatible
             with *data*. Instead of using this parameter, it is preferred
             to use a ``ZfitData`` as *data* that contains weights.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of :math:`K_i( x_i)`,
             resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.weights|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
            extended: |@doc:model.init.extended||@docend:model.init.extended|
        """
        original_init = {
            "data": data,
            "bandwidth": bandwidth,
            "num_grid_points": num_grid_points,
            "binning_method": binning_method,
            "support": support,
            "fft_method": fft_method,
            "obs": obs,
            "weights": weights,
            "extended": extended,
            "norm": norm,
            "name": name,
        }
        if isinstance(bandwidth, ZfitParameter):
            raise TypeError("bandwidth cannot be a Parameter for the FFT KDE.")
        if num_grid_points is None:
            num_grid_points = self._default_num_grid_points
        if binning_method is None:
            binning_method = "linear"
        if fft_method is None:
            fft_method = "conv1d"
        if kernel is None:
            kernel = tfd.Normal

        if padding is None:
            padding = self._default_padding
        if obs is None:
            if not isinstance(data, ZfitData) or not data.space.has_limits:
                raise ValueError(
                    "obs can only be None if data is ZfitData with limits."
                )
            else:
                obs = data.space
        data, size, weights, _ = self._convert_init_data_weights_size(
            data, weights, padding=padding, limits=obs.limits, bandwidth=bandwidth
        )
        self._padding = padding

        bandwidth, bandwidth_param = self._convert_input_bandwidth(
            bandwidth=bandwidth,
            data=data,
            padding=False,
            kernel=kernel,
            support=support,
            fft_method=fft_method,
            name=name,
            obs=obs,
            weights=weights,
        )
        num_grid_points = tf.minimum(
            tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int)
        )
        check_bw_grid_shapes(bandwidth, n_grid=num_grid_points)
        self._num_grid_points = num_grid_points
        self._binning_method = binning_method
        self._fft_method = fft_method
        self._data = data

        self._bandwidth = bandwidth

        params = {"bandwidth": self._bandwidth}
        super().__init__(
            obs=obs, name=name, params=params, extended=extended, norm=norm
        )
        self._kernel = kernel
        self._weights = weights
        if support is None:
            area = znp.reshape(self.space.area(), ())
            if area is not None:
                support = area * 1.2
        self._support = support
        self._grid = None
        self._grid_data = None

        self._grid = binning_util.generate_1d_grid(
            self._data, num_grid_points=self._num_grid_points
        )
        self._grid_data = binning_util.bin_1d(
            self._binning_method, self._data, self._grid, self._weights
        )
        self._grid_estimations = convolution_util.convolve_1d_data_with_kernel(
            self._kernel,
            self._bandwidth,
            self._grid_data,
            self._grid,
            self._support,
            self._fft_method,
        )
        self.hs3.original_init.update(original_init)

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        x_min = self._grid[0]
        x_max = self._grid[-1]

        value = tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_estimations)
        value.set_shape(x.shape)
        return value


class KDE1DimFFTRepr(BasePDFRepr):
    _implementation = KDE1DimFFT
    hs3_type: Literal["KDE1DimFFT"] = pydantic.Field("KDE1DimFFT", alias="type")

    data: Union[np.ndarray, Serializer.types.DataTypeDiscriminated]
    obs: Optional[SpaceRepr] = None
    bandwidth: Optional[Union[str, float]] = None
    num_grid_points: Optional[int] = None
    binning_method: Optional[str] = None
    kernel: None = None
    support: Optional[float] = None
    fft_method: Optional[str] = None
    padding: Optional[Union[bool, str]] = None
    weights: Optional[Union[np.ndarray, tf.Tensor]] = None
    name: Optional[str] = "KDE1DimFFT"

    @pydantic.validator("kernel", pre=True)
    def validate_kernel(cls, v):
        if v is not None:
            if v != tfd.Normal:
                raise ValueError(
                    "Kernel must be None for GridKDE1DimV1 to be serialized."
                )
            else:
                v = None
        return v

    @pydantic.root_validator(pre=True)
    def validate_all(cls, values):
        values = dict(values)
        if cls.orm_mode(values):
            for k, v in values["hs3"].original_init.items():
                values[k] = v
        return values


class KDE1DimISJ(KDEHelper, BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        data: ztyping.XTypeInput,
        *,
        obs: ztyping.ObsTypeInput | None = None,
        padding: callable | str | bool | None = None,
        num_grid_points: int | None = None,
        binning_method: str | None = None,
        weights: np.ndarray | tf.Tensor | None = None,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "KDE1DimISJ",
    ):
        r"""Kernel Density Estimation is a non-parametric method to approximate the density of given points.

        For a more in-depth explanation, see also in the section about Kernel Density Estimation
        :ref:`sec-isj-kde`

        .. math::

            f_h(x) =  \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x-x_i}{h}\Big)

        The bandwidth is computed by using a trick described in a paper by Botev et al. that uses the fact,
        that the Kernel Density Estimation
        with a Gaussian Kernel is a solution to the Heat Equation.

        Args:
            data: |@doc:pdf.kde.init.data| Data sample to approximate
             the density from. The points represent positions of the *kernel*,
             the :math:`x_i`. This is preferrably a ``ZfitData``, but can also
             be an array-like object.

             If the data has weights, they will be taken into account.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of
             :math:`K_i( x_i)`, resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.data|
            padding: |@doc:pdf.kde.init.padding| KDEs have a peculiar
             weakness: the boundaries, as the outside has a zero density. This makes the KDE
             go down at the bountary as well, as the density approaches zero, no matter what the
             density inside the boundary was.

             There are two ways to circumvent this problem:

               - the best solution: providing a larger dataset than the default space the PDF is used in
               - mirroring the existing data at the boundaries, which is equivalent to a boundary condition
                 with a zero derivative. This is a padding technique and can improve the boundaries.
                 However, one important drawback of this method is to keep in mind that this will actually
                 alter the PDF *to look mirrored*. If the PDF is plotted in a larger range, this becomes
                 clear.

             Possible options are a number (default 0.1) that depicts the fraction of the overall space
             that defines the data mirrored on both sides. For example, for a space from 0 to 5, a value of
             0.3 means that all data in the region of 0 to 1.5 is taken, mirrored around 0 as well as
             all data from 3.5 to 5 and mirrored at 5. The new data will go from -1.5 to 6.5, so the
             KDE is also having a shape outside the desired range. Using it only for the range 0 to 5
             hides this.
             Using a dict, each side separately (or only a single one) can be mirrored, like ``{'lowermirror: 0.1}``
             or ``{'lowermirror: 0.2, 'uppermirror': 0.1}``.
             For more control, a callable that takes data and weights can also be used. |@docend:pdf.kde.init.padding|
            num_grid_points: |@doc:pdf.kde.init.num_grid_points| Number of points in
             the binning grid.

             The data will be binned using the *binning_method* in *num_grid_points*
             and this histogram grid will then be used as kernel points. This has the
             advantage to have a constant computational complexity independent of the data
             size.

             A number from 32 on can already yield good results, while the default is set
             to 1024, creating a fine grid. Lowering the number increases the performance
             at the cost of accuracy. |@docend:pdf.kde.init.num_grid_points|
            binning_method: |@doc:pdf.kde.init.binning_method| Method to be used for
             binning the data. Options are 'linear', 'simple'.

             The data can be binned in the usual way ('simple'), but this is less precise
             for KDEs, where we are interested in the shape of the histogram and smoothing
             it. Therefore, a better suited method, 'linear', is available.

             In normal binnig, each event (or weight) falls into the bin within the bin edges,
             while the neighbouring bins get zero counts from this event.
             In linear binning, the event is split between two bins, proportional to its
             closeness to each bin.

             The 'linear' method provides superior performance, most notably in small (~32)
             grids. |@docend:pdf.kde.init.binning_method|
            obs: |@doc:pdf.kde.init.obs| Observable space of the KDE.
             As with any other PDF, this will be used as the default *norm*, but
             does not define the domain of the PDF. Namely, this can be a smaller
             space than *data*, as long as the name of the observable match.
             Using a larger dataset is actually good practice avoiding
             bountary biases, see also :ref:`sec-boundary-bias-and-padding`. |@docend:pdf.kde.init.obs|
            weights: |@doc:pdf.kde.init.weights| Weights of each event
             in *data*, can be None or Tensor-like with shape compatible
             with *data*. Instead of using this parameter, it is preferred
             to use a ``ZfitData`` as *data* that contains weights.
             This will change the count of the events, whereas
             weight :math:`w_i` of :math:`x_i` will scale the value of :math:`K_i( x_i)`,
             resulting in a factor of :math:`\frac{w_i}{\sum w_i} `.

             If no weights are given, each kernel will be scaled by the same
             constant :math:`\frac{1}{n_{data}}`. |@docend:pdf.kde.init.weights|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:model.init.name| Human-readable name
               or label of
               the PDF for better identification.
               Has no programmatical functional purpose as identification. |@docend:model.init.name|
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
        """
        original_init = {
            "data": data,
            "weights": weights,
            "num_grid_points": num_grid_points,
            "binning_method": binning_method,
            "obs": obs,
            "padding": padding,
            "name": name,
            "norm": norm,
            "extended": extended,
        }
        if num_grid_points is None:
            num_grid_points = self._default_num_grid_points
        if binning_method is None:
            binning_method = "linear"
        if padding is None:
            padding = self._default_padding
        if obs is None:
            if not isinstance(data, ZfitData) or not data.space.has_limits:
                raise ValueError(
                    "obs can only be None if data is ZfitData with limits."
                )
            else:
                obs = data.space
        data, size, weights, _ = self._convert_init_data_weights_size(
            data, weights, padding=padding, limits=obs.limits, bandwidth=None
        )
        self._padding = padding

        num_grid_points = tf.minimum(
            tf.cast(size, ztypes.int), tf.constant(num_grid_points, ztypes.int)
        )
        self._num_grid_points = num_grid_points
        self._binning_method = binning_method
        self._data = tf.convert_to_tensor(data, ztypes.float)
        self._weights = weights
        self._grid = None
        self._grid_data = None

        (
            self._bandwidth,
            self._grid_estimations,
            self._grid,
        ) = isj_util.calculate_bandwidth_and_density(
            self._data, self._num_grid_points, self._binning_method, self._weights
        )

        params = {}
        super().__init__(
            obs=obs, name=name, params=params, extended=extended, norm=norm
        )
        self.hs3.original_init.update(original_init)

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        x_min = self._grid[0]
        x_max = self._grid[-1]

        value = tfp.math.interp_regular_1d_grid(x, x_min, x_max, self._grid_estimations)
        value.set_shape(x.shape)
        return value


class KDE1DimISJRepr(BasePDFRepr):
    _implementation = KDE1DimISJ
    hs3_type: Literal["KDE1DimISJ"] = pydantic.Field("KDE1DimISJ", alias="type")

    data: Union[np.ndarray, Serializer.types.DataTypeDiscriminated]
    obs: Optional[SpaceRepr] = None
    bandwidth: Optional[Union[str, float]] = None
    num_grid_points: Optional[int] = None
    binning_method: Optional[str] = None
    kernel: None = None
    padding: Optional[Union[bool, str]] = None
    weights: Optional[Union[np.ndarray, tf.Tensor]] = None
    name: Optional[str] = "KDE1DimISJ"

    @pydantic.validator("kernel", pre=True)
    def validate_kernel(cls, v):
        if v is not None:
            if v != tfd.Normal:
                raise ValueError(
                    "Kernel must be None for GridKDE1DimV1 to be serialized."
                )
            else:
                v = None
        return v

    @pydantic.root_validator(pre=True)
    def validate_all(cls, values):
        values = dict(values)
        if cls.orm_mode(values):
            for k, v in values["hs3"].original_init.items():
                values[k] = v
        return values
