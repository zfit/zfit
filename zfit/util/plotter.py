#  Copyright (c) 2024 zfit
from __future__ import annotations

from typing import Callable, Mapping, Optional

from .. import convert_to_space
from ..core.interfaces import ZfitPDF
from . import ztyping
from .checks import RuntimeDependency

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    plt = RuntimeDependency("plt", error_msg=str(error))


def plot_sumpdf_components_pdf(
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
    plotfunc = plot_model_pdf if plotfunc is None else plotfunc
    for mod, frac in zip(model.pdfs, model.params.values()):
        plotfunc(mod, scale=frac * scale, ax=ax, linestyle=linestyle, plotkwargs=plotkwargs)
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
    plotfunc(x, y, label=model.label, linestyle=linestyle, **plotkwargs)
    if full:
        ax.set_xlabel(obs.label)
        ylabel = "Probability density" if not extended else "Probability"
        ax.set_ylabel(ylabel)
    plt.legend()
    return ax
