#  Copyright (c) 2020 zfit
from typing import Optional, Union

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from .functor import BaseFunctor
from .. import exception
from ..core.data import add_samples
from ..core.interfaces import ZfitPDF
from ..core.space import supports
from ..util import ztyping
from ..util.exception import WorkInProgressError, ShapeIncompatibleError


class FFTConvPDFV1(BaseFunctor):

    def __init__(self,
                 func: ZfitPDF,
                 kernel: ZfitPDF,
                 n: int = None,
                 limits_func: Union[ztyping.LimitsType, float] = None,
                 limits_kernel: ztyping.LimitsType = None,
                 interpolation: Optional[str] = None,
                 obs: ztyping.ObsTypeInput = None,
                 name: str = "FFTConvV1"):
        """*EXPERIMENTAL* Numerical Convolution pdf of `func` convoluted with `kernel` using FFT



        Args:
            func: PDF  with `pdf` method that takes x and returns the function value.
                Here x is a `Data` with the obs and limits of *limits*.
            kernel: PDF with `pdf` method that takes x acting as the kernel.
                Here x is a `Data` with the obs and limits of *limits*.
            n: Number of points _per dimension_ to evaluate the kernel and pdf at.
                The higher the number of points, the more accurate the convolution at the cost
                of computing time. If `None`, a heuristic is used (default to 100 in 1 dimension).
            limits_func: Specify in which limits the `func` should
                be evaluated for the convolution:
                - If `None`, the limits from the `func` are used and extended by a
                 default value (relative 0.2).
                - If float: the fraction of the limit do be extended. 0 means no extension,
                  1 would extend the limits to each side by the same size resulting in
                  a tripled size (for 1 dimension).
                  As an example, the limits (1, 5) with a `limits_func` of 0.5 would result in
                  effective limits of (-1, 7), as 0.5 * (5 - 1) = 2 has been added to each side.
                - If a space with limits is used, this is taken as the range.
            limits_kernel: the limits of the kernel. Usually not needed to change and automatically
                taken from the kernel.
            interpolation: Specify the method that is used for interpolation. Available methods are:
                - 'linear': this is the default for any convolution > 1 dimensional. It is a
                  fast, linear interpolation between the evaluated points and approximates the
                  function reasonably well in case of high number of points and a smooth response.
                - 'spline' or `f'spline{order}'`: a spline interpolation with polynomials. If
                  the order is not specified, a default is used. To specify the order, an integer
                  should be followed the word 'spline' as e.g. in 'spline3' to use a spline of
                  order three.
                  This method is considerably more computationally intensive as it requires to solve
                  a system of equations. When using 1000+ points this can affect the runtime critical.
                  However, it provides better solutions that are smooth even with less points
                  than for a linear interpolation.
            obs: Observables of the class. If not specified, automatically taken from `func`
            name: Human readable name of the PDF
        """
        valid_interpolations = ('spline', 'linear')

        obs = func.space if obs is None else obs
        super().__init__(obs=obs, pdfs=[func, kernel], params={}, name=name)

        if self.n_obs > 3:
            raise WorkInProgressError("More than 3 dimensional convolutions are currently not supported.")

        if interpolation is None:
            interpolation = 'spline' if self.n_obs == 1 else 'linear'

        spline_order = 3  # default
        if ':' in interpolation:
            interpolation, spline_order = interpolation.split(':')
            spline_order = int(spline_order)
        if interpolation not in valid_interpolations:
            raise ValueError(f"`interpolation` {interpolation} not known. Has to be one "
                             f"of the following: {valid_interpolations}")
        self._interpolation = interpolation
        self._spline_order = spline_order

        if n is None:
            npoints_scaling = 101
            # npoints_scaling = 5  # HACK
            n = tf.cast(limits_kernel.rect_area() / limits_func.rect_area() * npoints_scaling, tf.int32)[0]
            n = max(n, npoints_scaling)
            tf.assert_less(n - 1,  # so that for three dimension it's 999'999, not 10^6
                           tf.cast(1e6, tf.int32),
                           message="Number of points automatically calculated to be used for the FFT"
                                   " based convolution exceeds 1e6. If you want to use this number - "
                                   "or an even higher value - use explicitly the `n` argument.")
        if not n % 2:
            n += 1  # make it odd to have a unique shifting when using "same" in the convolution

        # get function limits
        if limits_func is None:
            limits_func = func.space
        limits_func = self._check_input_limits(limits=limits_func)
        if limits_func.n_limits == 0:
            raise exception.LimitsNotSpecifiedError("obs have to have limits to define where to integrate over.")
        if limits_func.n_limits > 1:
            raise WorkInProgressError("Multiple Limits not implemented")

        # get kernel limits
        if limits_kernel is None:
            limits_kernel = kernel.space
        limits_kernel = self._check_input_limits(limits=limits_kernel)

        if limits_kernel.n_limits == 0:
            raise exception.LimitsNotSpecifiedError("obs have to have limits to define where to integrate over.")
        if limits_kernel.n_limits > 1:
            raise WorkInProgressError("Multiple Limits not implemented")

        if func.n_obs != kernel.n_obs:
            raise ShapeIncompatibleError("Func and Kernel need to have (currently) the same number of obs,"
                                         f" currently are func: {func.n_obs} and kernel: {kernel.n_obs}")
        if not func.n_obs == limits_func.n_obs == limits_kernel.n_obs:
            raise ShapeIncompatibleError("Func and Kernel limits need to have (currently) the same number of obs,"
                                         f" are {limits_func.n_obs} and {limits_kernel.n_obs} with the func n_obs"
                                         f" {func.n_obs}")

        limits_func = limits_func.with_obs(obs)
        limits_kernel = limits_kernel.with_obs(obs)
        lower_func, upper_func = limits_func.rect_limits
        lower_kernel, upper_kernel = limits_kernel.rect_limits
        lower_sample = lower_func + upper_kernel
        upper_sample = upper_func + upper_kernel
        buffer = (upper_sample - lower_sample) / n * 2  # 2 * 2 buffer on each side
        # due to rounding up on the kernel side we can loose on a single side 2 bins: one because we take the
        # larger bin in the kernel (e.g. instead of kernel size to the right is 2 bins it's 3 (because the
        # edge would be 1.99 -> 2.98 instead) which increases our function histogram in size -> decreases our
        # buffer zone by one. Furthermore, by that the kernel grew by 1 -> we need 1 more in the buffer => 2 more
        lower_sample -= buffer
        upper_sample += buffer

        binwidth = (upper_sample - lower_sample) / n
        nbins_kernel = tf.cast(limits_kernel.rect_area() / limits_func.rect_area() * n + 1.,
                               dtype=tf.int32)  # casting is floor

        kernel_length = nbins_kernel * binwidth
        overflow = (kernel_length - (upper_kernel - lower_kernel)) / 2.
        lower_kernel -= overflow
        upper_kernel += overflow

        self._limits_kernel = lower_kernel, upper_kernel
        self._nbins_kernel = nbins_kernel

        self._xfunc_lower = lower_sample
        self._xfunc_upper = upper_sample
        self._npoints = n

    # @z.function
    def _unnormalized_pdf(self, x):
        # TODO: done up to here
        lower, upper = self._xfunc_lower, self._xfunc_upper
        x_funcs = tf.linspace(lower, upper, self._npoints)
        # x_kernels = self._limits_kernel.filter(x_funcs)
        x_kernels = x_funcs - (lower + upper) / 2

        # if self.n_obs == 2:
        #     xk1, xk2 = tf.unstack(x_kernels, axis=-1)
        #     xf1, xf2 = tf.unstack(x_funcs, axis=-1)
        #     x_kernel = xk1 * xk2[None, :]
        #     x_func = xf1 * xf2[None, :]
        #
        # else:
        x_kernel = tf.transpose(tf.meshgrid(*tf.unstack(x_kernels, axis=-1),
                                            indexing='ij'))

        x_func = tf.transpose(tf.meshgrid(*tf.unstack(x_funcs, axis=-1),
                                          indexing='ij'))

        y_func = self.pdfs[0].pdf(x_func)
        y_kernel = self.pdfs[1].pdf(x_kernel)

        npoints = self._npoints
        obs_dims = [npoints] * self.n_obs
        # HACK TODO: playing around
        y_kernel = tf.reshape(y_kernel, obs_dims)

        y_func = tf.reshape(y_func, obs_dims)
        # if self.n_obs > 1:
        #     y_kernel = tf.linalg.adjoint(y_kernel)

        # if self.n_obs == 1:
        # y_kernel = tf.reverse(y_kernel, axis=range(self.n_obs))
        if self.n_obs == 1:
            y_kernel = y_kernel[::-1]
        elif self.n_obs == 2:
            y_kernel = y_kernel[::-1, ::-1]
        # if self.n_obs == 2:
        #     y_kernel = tf.linalg.adjoint(y_kernel)
        new_shape = (1, *obs_dims, 1)
        if self.n_obs == 1 or True:  # HACK

            conv = tf.nn.convolution(
                input=tf.reshape(y_func, new_shape),
                filters=tf.reshape(y_kernel, (*obs_dims, 1, 1)),
                strides=1,
                padding='SAME',
            )

        # elif self.n_obs == 2:
        #     conv_fft = tf.signal.rfft2d(y_kernel) * tf.signal.rfft2d(y_func)
        #     conv = tf.signal.irfft2d(conv_fft)

        train_points = tf.reshape(x_func, (1, -1, self.n_obs))
        query_points = tf.expand_dims(x.value(), axis=0)
        if self.interpolation == 'spline':
            conv_points = tf.reshape(conv, (1, -1, 1))
            prob = tfa.image.interpolate_spline(train_points=train_points,
                                                train_values=conv_points,
                                                query_points=query_points,
                                                order=self._spline_order)
            prob = prob[0, ..., 0]
        elif self.interpolation == 'linear':
            prob = tfp.math.batch_interp_regular_nd_grid(x=query_points[0],
                                                         x_ref_min=self._xfunc_lower,
                                                         x_ref_max=self._xfunc_upper,
                                                         y_ref=conv[0, ..., 0],
                                                         # y_ref=tf.reverse(conv[0, ..., 0], axis=[0]),
                                                         axis=-self.n_obs)
            prob = prob[0]

        return prob

    @property
    def interpolation(self):
        return self._interpolation

    @supports()
    def _sample(self, n, limits):
        sample_func = self.pdfs[0].sample(n=n, limits=limits)
        # TODO: maybe use calculated kernel limits?
        sample_kernel = self.pdfs[1].sample(n=n)  # no limits! it's the kernel, around 0
        sample = add_samples(sample_func, sample_kernel, obs=limits)
        return sample
