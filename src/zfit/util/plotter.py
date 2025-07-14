#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing
from collections.abc import Callable, Mapping

import numpy as np

from zfit._interfaces import ZfitBinnedData, ZfitData, ZfitPDF, ZfitUnbinnedData

from ..core.space import convert_to_space
from . import ztyping
from .checks import RuntimeDependency
from .warnings import warn_experimental_feature

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    plt = RuntimeDependency("plt", error_msg=str(error))

try:
    import mplhep
except ImportError as error:
    mplhep = RuntimeDependency("mplhep", error_msg=str(error))


def plot_sumpdf_components_pdfV1(
    model,
    *,
    plotfunc: Callable | None = None,
    scale=1,
    ax=None,
    linestyle=None,
    plotkwargs: Mapping[str, object] | None = None,
    extended: bool | None = None,
):
    """Plot the components of a sum pdf.

    Args:
        model: A zfit SumPDF.
        plotfunc: A plotting function that takes the `ax` to plot on x, y, and additional arguments.
        scale: An overall scale factor to apply to the components.
        ax: A matplotlib Axes object to plot on.
        linestyle: A linestyle to use for the components. Default is "--".
        plotkwargs: Additional keyword arguments to pass to the plotting function.
        extended: If True, plot extended components. If None, uses the model's extended state.
    """
    import zfit  # noqa: PLC0415

    if not isinstance(model, zfit.pdf.SumPDF):
        msg = f"model must be a ZfitPDF, not a {type(model)}. Model is {model}."
        raise ValueError(msg)
    if linestyle is None:
        linestyle = "--"
    if plotkwargs is None:
        plotkwargs = {}
    if extended is None:
        extended = model.is_extended

    plotfunc = plot_model_pdf if plotfunc is None else plotfunc

    # Check if the SumPDF is automatically extended
    is_auto_extended = hasattr(model, "_automatically_extended") and model._automatically_extended

    # For automatically extended SumPDFs, we need to handle components differently
    if extended and is_auto_extended:
        for mod in model.pdfs:
            plotfunc(mod, scale=scale, ax=ax, linestyle=linestyle, extended=True, plotkwargs=plotkwargs)
    else:
        if extended:
            scale *= model.get_yield()
        # For non-extended or manually extended SumPDFs, use fractions
        # Force components to be non-extended and scale by fractions
        for mod, frac in zip(model.pdfs, model.params.values(), strict=True):
            plotfunc(mod, scale=frac * scale, ax=ax, linestyle=linestyle, extended=False, plotkwargs=plotkwargs)
    return ax


def plot_model_pdf(
    model: ZfitPDF,
    *,
    plotfunc: Callable | None = None,
    extended: bool | None = None,
    obs: ztyping.ObsTypeInput = None,
    scale: float | int | None = None,
    ax: plt.Axes | None = None,
    num: int | None = None,
    full: bool | None = None,
    linestyle=None,
    plotkwargs=None,
):
    """Plot the 1 dimensional density of a model, possibly scaled by the yield if extended.

    Args:
        model: An unbinned ZfitPDF.
        plotfunc: A plotting function that takes the ``ax`` to plot on, and  x, y, and additional arguments. Default is ``ax.plot``.
        extended: If True, plot the extended pdf. If False, plot the pdf.
        obs: The observable to plot the pdf for. If None, the model's space is used.
        scale: An overall scale factor to apply to the pdf.
        ax: A matplotlib Axes object to plot on.
        num: The number of points to evaluate the pdf at. Default is 300.
        full: If True, set the x and y labels and the legend. Default is True.
        linestyle: A linestyle to use for the pdf.
        plotkwargs: Additional keyword arguments to pass to the plotting function.

    Returns:
    """
    import zfit.z.numpy as znp  # noqa: PLC0415

    if not isinstance(model, ZfitPDF):
        msg = f"model must be a ZfitPDF, not a {type(model)}. Model is {model}."
        raise TypeError(msg)

    if scale is None:
        scale = 1
    if num is None:
        num = 300
    if full is None:
        full = True
    if plotkwargs is None:
        plotkwargs = {}

    if obs is None:
        obs = model.space
        if not obs.has_limits:
            msg = "Observables must have limits to be plotted. Either provide the limits with `obs` or use a model that has limits."
            raise ValueError(msg)
    else:
        obs = convert_to_space(obs)
        if not obs.has_limits:
            obs = model.space.with_obs(obs)
        if not obs.has_limits:
            msg = "Observables must have limits to be plotted. Either provide the limits with `obs` or use a model that has limits."
            raise ValueError(msg)

    if obs.n_obs != 1:
        msg = "obs must be 1D to be plotted."
        raise ValueError(msg)
    if model.space.n_obs != 1:
        if obs is None:
            msg = "1D space must be provided for multi-dimensional models to provide a 1D projection."
            raise ValueError(msg)
        model = model.create_projection_pdf(obs=obs, label=model.label)
    lower, upper = obs.v1.limits
    x = znp.linspace(lower, upper, num=num)
    y = model.ext_pdf(x) if extended else model.pdf(x)
    y *= scale
    if ax is None:
        ax = plt.gca()
    elif not isinstance(ax, plt.Axes):
        msg = "ax must be a matplotlib Axes object"
        raise ValueError(msg)
    plotfunc = ax.plot if plotfunc is None else plotfunc
    if "label" not in plotkwargs and full:
        plotkwargs["label"] = model.label
    plotfunc(x, y, linestyle=linestyle, **plotkwargs)
    if full:
        ax.set_xlabel(obs.label)
        ylabel = "Probability density" if not extended else "Extended probability density"
        ax.set_ylabel(ylabel)
        plt.legend()
    return ax


def assert_initialized(func):
    def wrapper(self, *args, **kwargs):
        if self.pdf is None:
            msg = "PDFPlotter is not initialized with a PDF."
            raise ValueError(msg)
        return func(self, *args, **kwargs)

    return wrapper


class ZfitPDFPlotter:
    @warn_experimental_feature
    @assert_initialized
    def plotpdf(
        self,
        data: ZfitData | None = None,
        *,
        depth: int | None = None,
        density: bool | None = None,
        plotfunc: Callable | None = None,
        extended: bool | None = None,
        obs: ztyping.ObsTypeInput = None,
        scale: float | int | None = None,
        ax: plt.Axes | None = None,
        num: int | None = None,
        full: bool | None = None,
        linestyle=None,
        plotkwargs: Mapping[str, object] | None = None,
        histplotkwargs: Mapping[str, object] | None = None,
    ):
        """Plot the 1 dimensional density of the PDF, possibly scaled by the yield if extended.

        This is the main plotting method for PDFs in zfit. It provides a quick way to visualize
        the probability density function.

        Examples:
            Basic usage::

                # Plot a simple PDF
                pdf.plot.plotpdf()

                # Plot extended PDF (scaled by yield)
                pdf.plot.plotpdf(extended=True)

                # Custom styling
                pdf.plot.plotpdf(color='red', linestyle='--', label='My PDF')

            For composite PDFs like SumPDF::

                # Plot the sum
                sumpdf.plot.plotpdf()

                # Plot components
                sumpdf.plot.comp.plotpdf(linestyle='--')

        Args:
            data: An optional `ZfitData` object to plot alongside the PDF. If provided, the PDF will be scaled
                to match the data's normalization (either count or density). If `data` is unbinned, it will be
                binned automatically for plotting with 50 bins. To provide a custom binning,
                use ``pdf.plot(data=data.to_binned(...), ...)``.
            depth: The depth to plot if the PDF is made up of a Sum pdf with components.
            density: If True, the data will be plotted as a density histogram. If False, it will be plotted.
                as a count histogram. If `data` is provided and `extended` is True, `density` defaults to False.
                If `data` is provided and `extended` is False, `density` defaults to True.
            extended: If True, plot the extended pdf (multiplied by the yield). If False, plot the
                normalized pdf. If None, uses the PDF's `is_extended` property.
            obs: The observable to plot the pdf for. If None, the model's space is used.
            scale: An overall scale factor to apply to the pdf. Useful for plotting multiple PDFs
                with different normalizations.
            ax: A matplotlib Axes object to plot on. If None, uses the current axes (plt.gca()).
            num: The number of points to evaluate the pdf at. Default is 300.
            full: If True, set the x and y labels and the legend. Default is True.
            linestyle: A linestyle to use for the pdf (e.g., '-', '--', '-.', ':').
            plotfunc: A plotting function that takes the `ax` to plot on, and x, y, and additional arguments.
                Default is `ax.plot`.
            plotkwargs: Additional keyword arguments to pass to the plotting function (e.g., color,
                alpha, linewidth, label).
            histplotkwargs: Additional keyword arguments to pass to `mplhep.histplot` when plotting data.

        Returns:
            matplotlib.axes.Axes: The matplotlib Axes object used for plotting.

        See Also:
            zfit.plot.plot_model_pdf: The underlying plotting function.
            SumPDF.plot.comp.plotpdf: For plotting components of composite PDFs.
        """
        extended = self._preprocess_args_extended(extended)
        if depth is None:
            depth = 1
        if scale is None:
            scale = 1
        if data is None:
            if density is not None:
                msg = "Density argument is only supported when data is provided."
                raise ValueError(msg)
            if histplotkwargs is not None:
                msg = "histplotkwargs argument is only supported when data is provided."
                raise ValueError(msg)
        else:
            if density is None:
                density = not extended
            normalize = not extended
            ax, newscale = self._plot_scale_data(
                data, density=density, normalize=normalize, ax=ax, histplotkwargs=histplotkwargs
            )
            scale *= newscale

        return self._plotpdf(
            depth=depth,
            plotfunc=plotfunc,
            extended=extended,
            obs=obs,
            scale=scale,
            ax=ax,
            num=num,
            full=full,
            linestyle=linestyle,
            plotkwargs=plotkwargs,
        )

    def _plotpdf(self, **kwargs):
        raise NotImplementedError

    @property
    def comp(self):
        return None

    def __call__(self, data=None, **kwargs):
        return self.plotpdf(data=data, **kwargs)

    def _plot_scale_data(
        self, data: ZfitData, density=None, normalize=None, ax=None, histplotkwargs=None
    ) -> (plt.Axes, float):
        """Plots the scaled data.

        This method plots the given data with optional density and normalization.

        Args
        ----------
        data: The ZfitBinnedData object to be plotted.

        density: If True, the plot will show the density of the data. Defaults to False.

        normalize: If True, the plot will normalize the data. Defaults to True.

        ax: The matplotlib axes object to plot on. If None, a new axes object will be created. Defaults to None.



        Notes
        -----
        The plot will be scaled based on the provided normalization.
        The density of the data will be displayed if the density parameter is set to True.
        """
        import zfit.z.numpy as znp  # noqa: PLC0415

        if histplotkwargs is None:
            histplotkwargs = {}
        if density is None:
            density = False
        if normalize is None:
            normalize = True
        if ax is None:
            ax = plt.gca()
        elif not isinstance(ax, plt.Axes):
            msg = "ax must be a matplotlib Axes object"
            raise ValueError(msg)

        if not isinstance(data, ZfitBinnedData) and isinstance(data, ZfitUnbinnedData):
            data = data.to_binned(50)
        values = data.values()
        binwidths = np.prod(data.binning.widths, axis=0)
        edges = data.binning.edges
        errors = None
        if (variances := data.variances()) is not None:
            errors = variances**0.5

        scale = 1
        nvals = None
        if density or normalize:
            nvals = znp.sum(values)
        if density:
            values /= binwidths
            if errors is not None:
                errors /= binwidths
        else:
            scale *= np.mean(binwidths)  # converting the PDF density to counts
        if normalize:
            values /= nvals
            if errors is not None:
                errors /= nvals
        # plot values
        mplhep.histplot((values, edges), yerr=errors, **histplotkwargs, label=data.label, ax=ax)
        return ax, scale

    def _preprocess_args_extended(self, extended):
        if extended is None:
            extended = self.pdf.is_extended
        if extended and not self.pdf.is_extended:
            msg = "Provided extended as argument for plotting, but pdf is not extended."
            raise ValueError(msg)
        return extended


class PDFPlotter(ZfitPDFPlotter):
    def __init__(
        self,
        pdf: ZfitPDF | None,
        pdfplotter: Callable | None = None,
        componentplotter: ZfitPDFPlotter = None,
        defaults: Mapping[str, object] | None = None,
    ):
        self.defaults = {} if defaults is None else defaults
        self.pdf = pdf
        if pdfplotter is not None and not callable(pdfplotter):
            msg = f"pdfplotter must be a callable, is {type(pdfplotter)}."
            raise TypeError(msg)
        self._pdfplotter = plot_model_pdf if pdfplotter is None else pdfplotter
        if componentplotter is not None and not isinstance(componentplotter, ZfitPDFPlotter):
            msg = f"componentplotter must be a ZfitPDFPlotter, is {type(componentplotter)}."
            raise TypeError(msg)
        self._componentplotter = componentplotter

    def _plotpdf(self, depth: int | None = None, **kwargs):
        if depth is None:
            depth = 1
        kwargs |= self.defaults
        ax = plot_model_pdf(self.pdf, **kwargs)
        _ = kwargs.pop("ax", None)
        if kwargs.get("linestyle") is None:
            kwargs["linestyle"] = ":"
        if depth and self.comp is not None:
            return self.comp(depth=depth - 1, ax=ax, **kwargs)
        return ax

    @property
    @assert_initialized
    def comp(self):
        return self._componentplotter


class SumCompPlotter(ZfitPDFPlotter):
    def __init__(
        self,
        pdf: ZfitPDF | None,
        *args,
        **kwargs,
    ):
        if pdf is not None and not isinstance(pdf, ZfitPDF):
            msg = f"pdf must be a ZfitPDF, is {type(pdf)}."
            raise TypeError(msg)
        self.pdf = pdf
        super().__init__(*args, **kwargs)

    def _plotpdf(self, data=None, *, depth: int | None = None, **kwargs):  # noqa: ARG002
        import zfit  # noqa: PLC0415

        if not isinstance(pdf := self.pdf, zfit.pdf.SumPDF):  # we can relax this later with duck typing
            msg = f"pdf must be a SumPDF, is {type(pdf)}."
            raise TypeError(msg)
        assert isinstance(pdf, zfit.pdf.SumPDF), "pdf must be a SumPDF"
        scale = kwargs.pop("scale", None)
        if scale is None:
            scale = 1
        if depth is None:
            depth = 0
        if depth < 0:
            ax = kwargs.get("ax")
            if ax is None:
                msg = (
                    "ax is None. Either there is an issue with the depth argument or an internal error. "
                    "Make sure `depth` is at least 0, if that's the case, please open a bug report "
                    "with zfit."
                )
                raise RuntimeError(msg)
            return ax
        if kwargs.pop("extended", False):
            scale *= pdf.get_yield()
        kwargs["extended"] = False  # we manually scale the components, this should always hold
        assert len(pdf.pdfs) > 0, "INTERNAL ERROR: pdfs cannot be empty"
        values = pdf.params.values()
        assert len(values) > 0, "INTERNAL ERROR: values cannot be empty"
        for mod, frac in zip(pdf.pdfs, values, strict=True):
            ax = mod.plot.plotpdf(scale=frac * scale, **kwargs, depth=depth - 1)
        return ax
