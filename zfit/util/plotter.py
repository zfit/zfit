#  Copyright (c) 2024 zfit
from __future__ import annotations

from typing import Callable, Mapping, Optional

from ..core.interfaces import ZfitPDF
from ..core.space import convert_to_space
from . import ztyping
from .checks import RuntimeDependency
from .exception import WorkInProgressError
from .warnings import warn_experimental_feature

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    plt = RuntimeDependency("plt", error_msg=str(error))


def plot_sumpdf_components_pdfV1(
    model,
    *,
    plotfunc: Optional[Callable] = None,
    scale=1,
    ax=None,
    linestyle=None,
    plotkwargs: Mapping[str, object] | None = None,
):
    """Plot the components of a sum pdf.

    Args:
        model: A zfit SumPDF.
        plotfunc: A plotting function that takes the `ax` to plot on x, y, and additional arguments.
        scale: An overall scale factor to apply to the components.
        ax: A matplotlib Axes object to plot on.
        linestyle: A linestyle to use for the components. Default is "--".
        plotkwargs: Additional keyword arguments to pass to the plotting function.
    """
    import zfit

    if not isinstance(model, zfit.pdf.SumPDF):
        msg = f"model must be a ZfitPDF, not a {type(model)}. Model is {model}."
        raise ValueError(msg)
    if linestyle is None:
        linestyle = "--"
    if plotkwargs is None:
        plotkwargs = {}
    plotfunc = plot_model_pdfV1 if plotfunc is None else plotfunc
    for mod, frac in zip(model.pdfs, model.params.values()):
        plotfunc(mod, scale=frac * scale, ax=ax, linestyle=linestyle, plotkwargs=plotkwargs)
    return ax


def plot_model_pdfV1(
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
    import zfit.z.numpy as znp

    if not isinstance(model, ZfitPDF):
        msg = f"model must be a ZfitPDF, not a {type(model)}. Model is {model}."
        raise TypeError(msg)

    if extended is None:
        extended = model.is_extended

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
        """Plot the 1 dimensional density of the PDF, possibly scaled by the yield if extended.

        Args:
            plotfunc: A plotting function that takes the `ax` to plot on, and x, y, and additional arguments. Default is `ax.plot`.
            extended: If True, plot the extended pdf. If False, plot the pdf.
            obs: The observable to plot the pdf for. If None, the model's space is used.
            scale: An overall scale factor to apply to the pdf.
            ax: A matplotlib Axes object to plot on.
            num: The number of points to evaluate the pdf at. Default is 300.
            full: If True, set the x and y labels and the legend. Default is True.
            linestyle: A linestyle to use for the pdf.
            plotkwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            The matplotlib Axes object.
        """
        return self._plotpdf(
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
        self._pdfplotter = plot_model_pdfV1 if pdfplotter is None else pdfplotter
        if componentplotter is not None and not isinstance(componentplotter, ZfitPDFPlotter):
            msg = f"componentplotter must be a ZfitPDFPlotter, is {type(componentplotter)}."
            raise TypeError(msg)
        self._componentplotter = componentplotter

    def __call__(self):
        msg = "This is not yet implemented. Use the methods instead."
        raise WorkInProgressError(msg)

    def _plotpdf(self, **kwargs):
        kwargs |= self.defaults
        return plot_model_pdfV1(self.pdf, **kwargs)

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

    def _plotpdf(self, **kwargs):
        import zfit

        if not isinstance(pdf := self.pdf, zfit.pdf.SumPDF):  # we can relax this later with duck typing
            msg = f"pdf must be a SumPDF, is {type(pdf)}."
            raise TypeError(msg)
        scale = kwargs.pop("scale")
        if scale is None:
            scale = 1
        for mod, frac in zip(pdf.pdfs, pdf.params.values()):
            ax = mod.plot.plotpdf(scale=frac * scale, **kwargs)
        return ax
