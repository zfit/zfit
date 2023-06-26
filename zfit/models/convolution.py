#  Copyright (c) 2023 zfit
from __future__ import annotations

from typing import Optional, Union

import pydantic
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Literal

import zfit.z.numpy as znp
from .functor import BaseFunctor
from .. import exception, z
from ..core.data import Data, sum_samples
from ..core.interfaces import ZfitPDF
from ..core.sample import accept_reject_sample
from ..core.serialmixin import SerializableMixin
from ..core.space import supports
from ..serialization import Serializer, SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..util import ztyping
from ..util.exception import ShapeIncompatibleError, WorkInProgressError
from ..util.ztyping import ExtendedInputType, NormInputType
from ..z.interpolate_spline import interpolate_spline

LimitsTypeInput = Optional[Union[ztyping.LimitsType, float]]


class FFTConvPDFV1(BaseFunctor, SerializableMixin):
    def __init__(
        self,
        func: ZfitPDF,
        kernel: ZfitPDF,
        n: int | None = None,
        limits_func: LimitsTypeInput | None = None,
        limits_kernel: ztyping.LimitsType | None = None,
        interpolation: str | None = None,
        obs: ztyping.ObsTypeInput | None = None,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str = "FFTConvV1",
    ):
        r"""*EXPERIMENTAL* Numerical Convolution pdf of `func` convoluted with `kernel` using FFT

        CURRENTLY ONLY 1 DIMENSIONAL!

        EXPERIMENTAL: Feedback is very welcome! Performance, which parameters to tune, which fail etc.

        TL;DR technical details:

          - FFT-like technique: discretization of function. Number of bins splits the kernel into `n` bins
            and uses the same binwidth for the func while extending it by the kernel space. Internally,
            `tf.nn.convolution` is used.
          - Then interpolation by either linear or spline function
          - The kernel is assumed to be "small enough" outside of it's `space` and points there won't be
            evaluated.

        The convolution of two (normalized) functions is defined as

        .. math::

            (f * g)(t) \triangleq\ \int_{-\infty}^\infty f(\tau) g(t - \tau) \, d\tau

        It defines the "smearing" of `func` by a `kernel`. This is when an element in `func` is
        randomly added to an element of `kernel`. While the sampling (the addition of elements) is rather
        simple to do computationally, the calculation of the convolutional PDF (if there is no analytic
        solution available) is not, as it requires:

            - an integral from -inf to inf
            - an integral *for every point of x that is requested*

        This can be solved with a few tricks. Instead of integrating to infinity, it is usually sufficient to
        integrate from a point where the function is "small enough".

        If the functions are arbitrary and with conditional dependencies,
        there is no way around an integral and another PDF has to be used. If the two functions are
        uncorrelated, a simplified version can be done by a discretization of the space (followed by a
        Fast Fourier Transfrom, after which the convolution becomes a simple multiplication) and a
        discrete convolution can be performed.

        An interpolation of the discrete convolution for the requested points `x` is performed afterwards.


        Args:
            func: PDF  with `pdf` method that takes x and returns the function value.
                Here x is a `Data` with the obs and limits of *limits*.
            kernel: PDF with `pdf` method that takes x acting as the kernel.
                Here x is a `Data` with the obs and limits of *limits*.
            n: Number of points *per dimension* to evaluate the kernel and pdf at.
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
                - 'spline' or `f'spline:{order}'`: a spline interpolation with polynomials. If
                  the order is not specified, a default is used. To specify the order, 'spline'
                  should be followed  an integer, separated by a colon as e.g. in 'spline:3'
                  to use a spline of order three.
                  This method is considerably more computationally expensive as it requires to solve
                  a system of equations. When using 1000+ points this can affect the runtime critical.
                  However, it provides better solution, a curve that is smooth even with less points
                  than for a linear interpolation.

            obs: Observables of the class. If not specified, automatically taken from `func`
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: Human readable name of the PDF
        """
        from zfit import run

        run.assert_executing_eagerly()
        valid_interpolations = ("spline", "linear")
        original_init = {
            "func": func,
            "kernel": kernel,
            "n": n,
            "limits_func": limits_func,
            "limits_kernel": limits_kernel,
            "interpolation": interpolation,
            "obs": obs,
            "extended": extended,
            "norm": norm,
            "name": name,
        }

        obs = func.space if obs is None else obs
        super().__init__(
            obs=obs,
            pdfs=[func, kernel],
            params={},
            name=name,
            extended=extended,
            norm=norm,
        )
        self.hs3.original_init.update(original_init)

        if self.n_obs > 1:
            raise WorkInProgressError(
                "More than 1 dimensional convolutions are currently not supported."
                " If you need that, please open an issue on github."
            )

        if self.n_obs > 3:  # due to tf.nn.convolution not supporting more
            raise WorkInProgressError(
                "More than 3 dimensional convolutions are currently not supported."
            )

        if interpolation is None:
            interpolation = "linear"  # "spline" if self.n_obs == 1 else "linear"  # TODO: spline is very inefficient, why?

        spline_order = 3  # default
        if ":" in interpolation:
            interpolation, spline_order = interpolation.split(":")
            spline_order = int(spline_order)
        if interpolation not in valid_interpolations:
            raise ValueError(
                f"`interpolation` {interpolation} not known. Has to be one "
                f"of the following: {valid_interpolations}"
            )
        self._conv_interpolation = interpolation
        self._conv_spline_order = spline_order

        # get function limits
        if limits_func is None:
            limits_func = func.space
        limits_func = self._check_input_limits(limits=limits_func)
        if limits_func.n_limits == 0:
            raise exception.LimitsNotSpecifiedError(
                "obs have to have limits to define where to integrate over."
            )
        if limits_func.n_limits > 1:
            raise WorkInProgressError("Multiple Limits not implemented")

        # get kernel limits
        if limits_kernel is None:
            limits_kernel = kernel.space
        limits_kernel = self._check_input_limits(limits=limits_kernel)

        if limits_kernel.n_limits == 0:
            raise exception.LimitsNotSpecifiedError(
                "obs have to have limits to define where to integrate over."
            )
        if limits_kernel.n_limits > 1:
            raise WorkInProgressError("Multiple Limits not implemented")

        if func.n_obs != kernel.n_obs:
            raise ShapeIncompatibleError(
                "Func and Kernel need to have (currently) the same number of obs,"
                f" currently are func: {func.n_obs} and kernel: {kernel.n_obs}"
            )
        if not func.n_obs == limits_func.n_obs == limits_kernel.n_obs:
            raise ShapeIncompatibleError(
                "Func and Kernel limits need to have (currently) the same number of obs,"
                f" are {limits_func.n_obs} and {limits_kernel.n_obs} with the func n_obs"
                f" {func.n_obs}"
            )

        limits_func = limits_func.with_obs(obs)
        limits_kernel = limits_kernel.with_obs(obs)

        self._initargs = {"limits_kernel": limits_kernel, "limits_func": limits_func}

        if n is None:
            n = 51  # default kernel points
        if not n % 2:
            n += 1  # make it odd to have a unique shifting when using "same" in the convolution

        lower_func, upper_func = limits_func.rect_limits
        lower_kernel, upper_kernel = limits_kernel.rect_limits
        lower_sample = lower_func + lower_kernel
        upper_sample = upper_func + upper_kernel

        # TODO: what if kernel area is larger?
        if limits_kernel.rect_area() > limits_func.rect_area():
            raise WorkInProgressError(
                "Currently, only kernels that are smaller than the func are supported."
                "Simply switch the two should resolve the problem."
            )

        # get the finest resolution. Find the dimensions with the largest kernel-space to func-space ratio
        # We take the binwidth of the kernel as the overall binwidth and need to have the same binning in
        # the function as well
        area_ratios = (upper_sample - lower_sample) / (
            limits_kernel.rect_upper - limits_kernel.rect_lower
        )
        nbins_func_exact_max = znp.max(area_ratios * n)
        nbins_func = znp.ceil(
            nbins_func_exact_max
        )  # plus one and floor is like ceiling (we want more bins) with the
        # guarantee that we add one bin (e.g. if we hit exactly the boundaries, we add one.
        nbins_kernel = n
        # n = max(n, npoints_scaling)
        # TODO: below needed if we try to estimate the number of points
        # tf.assert_less(
        #     n - 1,  # so that for three dimension it's 999'999, not 10^6
        #     tf.cast(1e6, tf.int32),
        #     message="Number of points automatically calculated to be used for the FFT"
        #     " based convolution exceeds 1e6. If you want to use this number - "
        #     "or an even higher value - use explicitly the `n` argument.",
        # )

        binwidth = (upper_kernel - lower_kernel) / nbins_kernel
        to_extend = (
            binwidth * nbins_func - (upper_sample - lower_sample)
        ) / 2  # how much we need to extend the func_limits
        # on each side in order to match the binwidth of the kernel
        lower_sample -= to_extend
        upper_sample += to_extend

        lower_valid = lower_sample + lower_kernel
        upper_valid = upper_sample + upper_kernel

        self._conv_limits = {
            "kernel": (lower_kernel, upper_kernel),
            "valid": (lower_valid, upper_valid),
            "func": (lower_sample, upper_sample),
            "nbins_kernel": nbins_kernel,
            "nbins_func": nbins_func,
        }

    @z.function(wraps="model_convolution")
    def _unnormalized_pdf(self, x):
        lower_func, upper_func = self._conv_limits["func"]
        nbins_func = self._conv_limits["nbins_func"]
        x_funcs = tf.linspace(lower_func, upper_func, tf.cast(nbins_func, tf.int32))

        lower_kernel, upper_kernel = self._conv_limits["kernel"]
        nbins_kernel = self._conv_limits["nbins_kernel"]
        x_kernels = tf.linspace(
            lower_kernel, upper_kernel, tf.cast(nbins_kernel, tf.int32)
        )

        x_func = tf.meshgrid(*tf.unstack(x_funcs, axis=-1), indexing="ij")
        x_func = znp.transpose(x_func)
        x_func_flatish = znp.reshape(x_func, (-1, self.n_obs))
        data_func = Data.from_tensor(tensor=x_func_flatish, obs=self.obs)

        x_kernel = tf.meshgrid(*tf.unstack(x_kernels, axis=-1), indexing="ij")
        x_kernel = znp.transpose(x_kernel)
        x_kernel_flatish = znp.reshape(x_kernel, (-1, self.n_obs))
        data_kernel = Data.from_tensor(tensor=x_kernel_flatish, obs=self.obs)

        y_func = self.pdfs[0].pdf(data_func, norm=False)
        y_kernel = self.pdfs[1].pdf(data_kernel, norm=False)

        func_dims = [nbins_func] * self.n_obs
        kernel_dims = [nbins_kernel] * self.n_obs

        y_func = znp.reshape(y_func, func_dims)
        y_kernel = znp.reshape(y_kernel, kernel_dims)

        # flip the kernel to use the cross-correlation called `convolution function from TF
        # convolution = cross-correlation with flipped kernel
        # this introduces a shift and has to be corrected when interpolating/x_func
        # because the convolution has to be independent of the kernes **limits**
        # We assume they are symmetric when doing the FFT, so shift them back.
        y_kernel = tf.reverse(y_kernel, axis=range(self.n_obs))
        kernel_shift = (upper_kernel + lower_kernel) / 2
        x_func += kernel_shift
        lower_func += kernel_shift
        upper_func += kernel_shift

        # make rectangular grid
        y_func_rect = znp.reshape(y_func, func_dims)
        y_kernel_rect = znp.reshape(y_kernel, kernel_dims)

        # needed for multi dims?
        # if self.n_obs == 2:
        #     y_kernel_rect = tf.linalg.adjoint(y_kernel_rect)

        # get correct shape for tf.nn.convolution
        y_func_rect_conv = znp.reshape(y_func_rect, (1, *func_dims, 1))
        y_kernel_rect_conv = znp.reshape(y_kernel_rect, (*kernel_dims, 1, 1))

        conv = tf.nn.convolution(
            input=y_func_rect_conv,
            filters=y_kernel_rect_conv,
            strides=1,
            padding="SAME",
        )

        # needed for multidims?
        # if self.n_obs == 2:
        #     conv = tf.linalg.adjoint(conv[0, ..., 0])[None, ..., None]
        # conv = scipy.signal.convolve(
        #     y_func_rect,
        #     y_kernel_rect,
        #     mode='same'
        # )[None, ..., None]
        train_points = znp.expand_dims(x_func, axis=0)
        query_points = znp.expand_dims(x.value(), axis=0)
        if self.conv_interpolation == "spline":
            conv_points = znp.reshape(conv, (1, -1, 1))
            prob = interpolate_spline(
                train_points=train_points,
                train_values=conv_points,
                query_points=query_points,
                order=self._conv_spline_order,
            )
            prob = prob[0, ..., 0]
        elif self.conv_interpolation == "linear":
            prob = tfp.math.batch_interp_regular_1d_grid(
                x=query_points[0, ..., 0],
                x_ref_min=lower_func[..., 0],
                x_ref_max=upper_func[..., 0],
                y_ref=conv[0, ..., 0],
                # y_ref=tf.reverse(conv[0, ..., 0], axis=[0]),
                axis=0,
            )
            prob = prob[0]

        return prob

    @property
    def conv_interpolation(self):
        return self._conv_interpolation

    @supports()
    def _sample(self, n, limits):
        # this is a custom implementation of sampling. Since the kernel and func are not correlated,
        # we can simply sample from both and add them. This is "trivial" compared to accept reject sampling
        # However, one large pitfall is that we cannot simply request n events from each pdf and then add them.
        # The kernel can move points out of the limits that we want, since it's "smearing" it.
        # E.g. with x sampled between limits, x + xkernel can be outside of limits.
        # Therefore we need to (repeatedly) sample from a) the func in the range of limits +
        # the kernel limits; to extend limits in the upper direction with the kernel upper limits and vice versa.
        # Therefore we (ab)use the accept reject sample: it samples until it is full. Everything has a constant
        # probability to be accepted.
        # This is maybe not the most efficient way to do and a more specialized (meaning taking care of less
        # special cases such as `accept_reject_sample` does) can be more efficient. However, sampling is not
        # supposed to be the bottleneck anyway.
        func = self.pdfs[0]
        kernel = self.pdfs[1]

        sample_and_weights = AddingSampleAndWeights(
            func=func,
            kernel=kernel,
            limits_func=self._initargs["limits_func"],
            limits_kernel=self._initargs["limits_kernel"],
        )

        return accept_reject_sample(
            lambda x: tf.ones(
                shape=tf.shape(x.value())[0], dtype=self.dtype
            ),  # use inside?
            # all the points are inside
            n=n,
            limits=limits,
            sample_and_weights_factory=lambda: sample_and_weights,
            dtype=self.dtype,
            prob_max=1.0,
            efficiency_estimation=0.9,
        )


class FFTConvPDFV1Repr(BasePDFRepr):
    _implementation = FFTConvPDFV1
    hs3_type: Literal["FFTConvPDFV1"] = pydantic.Field("FFTConvPDFV1", alias="type")
    func: Serializer.types.PDFTypeDiscriminated

    kernel: Serializer.types.PDFTypeDiscriminated
    n: Optional[int] = None
    limits_func: Optional[SpaceRepr] = None
    limits_kernel: Optional[SpaceRepr] = None
    interpolation: Optional[str] = None
    obs: Optional[SpaceRepr] = None

    @pydantic.root_validator(pre=True)
    def validate_all(cls, values):
        values = dict(values)
        if cls.orm_mode(values):
            for k, v in values["hs3"].original_init.items():
                values[k] = v
        return values


class AddingSampleAndWeights:
    def __init__(self, func, kernel, limits_func, limits_kernel) -> None:
        super().__init__()
        self.func = func
        self.kernel = kernel
        self.limits_func = limits_func
        self.limits_kernel = limits_kernel

    def __call__(self, n_to_produce: int | tf.Tensor, limits, dtype):
        kernel_lower, kernel_upper = self.kernel.space.rect_limits
        sample_lower, sample_upper = limits.rect_limits

        sample_ext_lower = sample_lower + kernel_lower
        sample_ext_upper = sample_upper + kernel_upper
        sample_ext_space = limits.with_limits((sample_ext_lower, sample_ext_upper))

        sample_func = self.func.sample(n=n_to_produce, limits=sample_ext_space)
        sample_kernel = self.kernel.sample(
            n=n_to_produce, limits=self.limits_kernel
        )  # no limits! it's the kernel, around 0
        sample = sum_samples(sample_func, sample_kernel, obs=limits, shuffle=True)
        sample = limits.filter(sample)
        n_drawn = tf.shape(sample)[0]
        from zfit import run

        if run.numeric_checks:
            tf.debugging.assert_positive(
                n_drawn,
                "Could not draw any samples. Check the limits of the func and kernel.",
            )
        thresholds_unscaled = tf.ones(shape=(n_drawn,), dtype=sample.dtype)
        weights = thresholds_unscaled  # also "ones_like"
        weights_max = z.constant(1.0)
        return sample, thresholds_unscaled * 0.5, weights, weights_max, n_drawn
