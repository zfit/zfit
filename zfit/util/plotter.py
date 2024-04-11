#  Copyright (c) 2024 zfit
from __future__ import annotations

from .. import convert_to_space
from .checks import RuntimeDependency

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    plt = RuntimeDependency("plt", error_msg=str(error))


def plot_comp_model_pdf(model, *, extended=None, obs=None, scale=1, ax=None, num=300, full=True):
    for mod, frac in zip(model.pdfs, model.params.values()):
        plot_model_pdf(mod, scale=frac * scale, ax=ax, num=num, full=full, extended=extended, obs=obs)
    plot_model_pdf(model, scale=scale, ax=ax, num=num, full=full, extended=extended, obs=obs)
    return ax


def plot_model_pdf(model, *, extended=None, obs=None, scale=1, ax=None, num=300, full=True):
    import zfit.z.numpy as znp

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
    ax.plot(x, y, label=model.label)
    if full:
        ax.set_xlabel(obs.label)
        ylabel = "Probability density" if not extended else "Probability"
        ax.set_ylabel(ylabel)
    plt.legend()
    return ax
