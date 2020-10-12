#  Copyright (c) 2020 zfit
from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from .functor import BaseFunctor
from ..core.basepdf import BasePDF
from ..util import exception, ztyping
from ..util.exception import WorkInProgressError, ShapeIncompatibleError


class FFTConv1DV1(BaseFunctor):
    def __init__(self, func: BasePDF, kernel: BasePDF,
                 limits_func: ztyping.ObsTypeInput = None, limits_kernel: ztyping.ObsTypeInput = None,
                 obs: ztyping.ObsTypeInput = None, interpolation: Optional[str] = None,
                 npoints: int = None, name: str = "FFTConv1DV1"):
        """Numerical Convolution pdf of *func* convoluted with *kernel*.

        Args:
            func (:py:class:`zfit.pdf.BasePDF`): PDF  with `pdf` method that takes x and returns the function value.
                Here x is a `Data` with the obs and limits of *limits*.
            kernel (:py:class:`zfit.pdf.BasePDF`): PDF with `pdf` method that takes x acting as the kernel.
                Here x is a `Data` with the obs and limits of *limits*.
            limits_func (:py:class:`zfit.Space`): Limits for the numerical integration.
            obs (:py:class:`zfit.Space`): Observables of the class
            npoints (int): Number of points to evaluate the kernel and pdf at
            name (str): Human readable name of the pdf
        """
        valid_interpolations = ('spline', 'linear')

        obs = func.space if obs is None else obs
        super().__init__(obs=obs, pdfs=[func, kernel], params={}, name=name)
        stretch_limits_func = False
        # stretch_limits_kernel = False
        if limits_func is None:
            limits_func = func.space
            stretch_limits_func = True
        if limits_kernel is None:
            # stretch_limits_kernel = True
            limits_kernel = kernel.space
        limits_func = self._check_input_limits(limits=limits_func)
        limits_kernel = self._check_input_limits(limits=limits_kernel)
        if limits_func.n_limits == 0:
            raise exception.LimitsNotSpecifiedError("obs have to have limits to define where to integrate over.")
        if limits_func.n_limits > 1:
            raise WorkInProgressError("Multiple Limits not implemented")

        if func.n_obs != kernel.n_obs:
            raise ShapeIncompatibleError("Func and Kernel need to have (currently) the same number of obs,"
                                         f" currently are func: {func.n_obs} and kernel: {kernel.n_obs}")
        if not func.n_obs == limits_func.n_obs == limits_kernel.n_obs:
            raise ShapeIncompatibleError("Func and Kernel limits need to have (currently) the same number of obs,"
                                         f" are {limits_func.n_obs} and {limits_kernel.n_obs} with the func n_obs"
                                         f" {func.n_obs}")

        if self.n_obs > 3:
            raise WorkInProgressError("More than 3 dimensional convolutions are currently not supported.")

        if interpolation is None:
            interpolation = 'spline' if self.n_obs == 1 else 'linear'

        if interpolation not in valid_interpolations:
            raise ValueError(f"`interpolation` {interpolation} not known. Has to be one "
                             f"of the following: {valid_interpolations}")
        self._interpolation = interpolation

        if npoints is None:
            npoints_scaling = 100
            npoints = tf.cast(limits_kernel.rect_area() / limits_func.rect_area() * npoints_scaling, tf.int32)[0]
            npoints = max(npoints, npoints_scaling)
            tf.assert_less(npoints, tf.cast(1e6, tf.int32),
                           message="Number of points automatically calculated to be used for the FFT"
                                   " based convolution exceeds 1e6. If you want to use this number - "
                                   "or an even higher value - use explicitly the `npoints` argument.")
        x_kernels = []
        x_funcs = []

        limit_func = limits_func.with_obs(obs)
        lower, upper = limit_func.rect_limits
        if stretch_limits_func:
            areas = upper - lower
            areas *= 0.2  # add to limits this fraction
            lower -= areas
            upper += areas
        x_funcs = tf.linspace(lower, upper, npoints)
        x_kernels = x_funcs - (lower + upper) / 2

        # for ob in self.obs:
        #     limit_func = limits_func.with_obs(ob)
        #     x_func_min, x_func_max = limit_func.limit1d
        #     if stretch_limits_func:
        #         add_to_limit = limit_func.rect_area()[0] * 0.2
        #         x_func_min -= add_to_limit
        #         x_func_max += add_to_limit
        #     x_func = tf.linspace(x_func_min, x_func_max, npoints)
        #     x_shifted = x_func - (x_func_max + x_func_min) / 2
        #     # x_kernel = limits_kernel.filter(x_shifted)
        #     x_kernel = x_shifted
        #     x_kernels.append(x_kernel)
        #     x_funcs.append(x_func)

        x_kernel = - tf.transpose(tf.meshgrid(*tf.unstack(x_kernels, axis=-1),
                                              indexing='ij'))
        x_func = - tf.transpose(tf.meshgrid(*tf.unstack(x_funcs, axis=-1),
                                            indexing='ij'))
        self._xfunc_lower = lower
        self._xfunc_upper = upper
        self._npoints = npoints
        self._xfunc = x_func
        self._xkernel = x_kernel

    # @z.function
    def _unnormalized_pdf(self, x):
        # x = z.unstack_x(x)
        y_func = self.pdfs[0].pdf(self._xfunc)
        y_kernel = self.pdfs[1].pdf(self._xkernel)

        # conv = tf.nn.conv1d(
        #     input=tf.reshape(y_func, (1, -1, 1)),
        #     filters=tf.reshape(y_kernel, (-1, 1, 1)),
        #     stride=1,
        #     padding='SAME',
        #     data_format='NWC'
        # )
        npoints = self._npoints
        obs_dims = [npoints] * self.n_obs
        new_shape = (1, *obs_dims, 1)
        conv = tf.nn.convolution(
            input=tf.reshape(y_func, new_shape),
            filters=tf.reshape(y_kernel, (*obs_dims, 1, 1)),
            strides=1,
            padding='SAME',
            # data_format='NWC'
        )

        # conv = tf.reshape(conv, (-1,))
        # prob = tfp.math.interp_regular_1d_grid(x=x,
        #                                        x_ref_min=self._x_func_min,
        #                                        x_ref_max=self._x_func_max,
        #                                        y_ref=conv)
        train_points = tf.reshape(self._xfunc, (1, -1, self.n_obs))
        query_points = tf.expand_dims(x.value(), axis=0)
        if self.interpolation == 'spline':
            conv_points = tf.reshape(conv, (1, -1, 1))
            prob = tfa.image.interpolate_spline(train_points=train_points,
                                                train_values=conv_points,
                                                query_points=query_points,
                                                order=3)
            prob = prob[0, ..., 0]
        elif self.interpolation == 'linear':
            prob = tfp.math.batch_interp_regular_nd_grid(x=query_points[0],  # they are inverted due to the convolution
                                                         x_ref_min=self._xfunc_lower,
                                                         x_ref_max=self._xfunc_upper,
                                                         y_ref=tf.reverse(conv[0, ..., 0], axis=[0]),
                                                         axis=-self.n_obs)
            prob = prob[0]

        return prob

    @property
    def interpolation(self):
        return self._interpolation
