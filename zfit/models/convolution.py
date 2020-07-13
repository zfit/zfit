#  Copyright (c) 2020 zfit
import tensorflow as tf
import tensorflow_addons as tfa

from .functor import BaseFunctor
from .. import z
from ..core.basepdf import BasePDF
from ..util import exception, ztyping
from ..util.exception import WorkInProgressError, ShapeIncompatibleError


class FFTConv1DV1(BaseFunctor):
    def __init__(self, func: BasePDF, kernel: BasePDF,
                 limits_func: ztyping.ObsTypeInput = None, limits_kernel: ztyping.ObsTypeInput = None,
                 obs: ztyping.ObsTypeInput = None,
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

        if npoints is None:
            npoints_scaling = 100
            npoints = tf.cast(limits_kernel.rect_area() / limits_func.rect_area() * npoints_scaling, tf.int32)[0]
            npoints = max(npoints, npoints_scaling)
            tf.assert_less(npoints, tf.cast(1e6, tf.int32),
                           message="Number of points automatically calculated to be used for the FFT"
                                   " based convolution exceeds 1e6. If you want to use this number - "
                                   "or an even higher value - use explicitly the `npoints` argument.")
        x_func_min, x_func_max = limits_func.limit1d
        if stretch_limits_func:
            add_to_limit = limits_func.rect_area()[0] * 0.2
            x_func_min -= add_to_limit
            x_func_max += add_to_limit
        x_func = tf.linspace(x_func_min, x_func_max, npoints)
        x_shifted = x_func - (x_func_max + x_func_min) / 2
        x_kernel = limits_kernel.filter(x_shifted)
        self._x_func_min = x_func_min
        self._x_func_max = x_func_max
        self._x_func = x_func
        self._x_kernel = -x_kernel

    # @z.function
    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        y_func = self.pdfs[0].pdf(self._x_func)
        y_kernel = self.pdfs[1].pdf(self._x_kernel)

        conv = tf.nn.conv1d(
            input=tf.reshape(y_func, (1, -1, 1)),
            filters=tf.reshape(y_kernel, (-1, 1, 1)),
            stride=1,
            padding='SAME',
            data_format='NWC'
        )
        # conv = tf.reshape(conv, (-1,))
        # prob = tfp.math.interp_regular_1d_grid(x=x,
        #                                        x_ref_min=self._x_func_min,
        #                                        x_ref_max=self._x_func_max,
        #                                        y_ref=conv)
        prob = tfa.image.interpolate_spline(train_points=tf.reshape(self._x_func, (1, -1, 1)),
                                            train_values=conv,
                                            query_points=tf.reshape(x, (1, -1, 1)),
                                            order=4)
        prob = tf.reshape(prob, (-1,))

        return prob
