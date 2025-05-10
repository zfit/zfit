#  Copyright (c) 2025 zfit
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import zfit
from zfit import Parameter

# Setup
here = Path(__file__).absolute().parent
OUTPATH = Path(here / "../images/_generated/pdfs").absolute().resolve()
OUTPATH.mkdir(parents=True, exist_ok=True)
print(f"Saving plots to {OUTPATH}")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


@dataclass
class PDFConfig:
    """Configuration for plotting a PDF."""

    pdf_class: type
    param_name: str
    param_values: list[float]
    title: str
    filename: str
    x_range: tuple[float, float]
    fixed_params: dict[str, float] = field(default_factory=dict)
    label_fn: Callable[[float], str] = None


def save_plot(filename):
    """Save the current plot."""
    plt.tight_layout()
    plt.savefig(OUTPATH / filename, dpi=100, bbox_inches="tight")
    plt.close()


def plot_pdf(config, obs=None):
    """Generic function to plot a PDF with parameter variation."""
    if obs is None:
        obs = zfit.Space("x", limits=config.x_range)

    plt.figure()
    fixed_params = {name: Parameter(name, value) for name, value in config.fixed_params.items()}

    for value in config.param_values:
        param = Parameter(config.param_name, value)
        params = {config.param_name: param, **fixed_params}
        pdf = config.pdf_class(**params, obs=obs)

        x = np.linspace(config.x_range[0], config.x_range[1], 1000)
        y = pdf.pdf(x)

        label = config.label_fn(value) if config.label_fn else f"{config.param_name} = {value}"
        plt.plot(x, y, label=label)

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(config.title)
    plt.legend()
    save_plot(config.filename)


def create_configs(pdf_class, base_name, base_title, x_range, configs_data):
    """Create standard configs for a PDF."""
    return [
        PDFConfig(
            pdf_class=pdf_class,
            param_name=param_name,
            param_values=values,
            title=f"{base_title} with different {label or param_name} values",
            filename=f"{base_name}_{param_name}.png",
            x_range=x_range,
            fixed_params=fixed_params,
            label_fn=label_fn,
        )
        for param_name, values, fixed_params, label, label_fn in configs_data
    ]


def plot_multiple_configs(configs, obs=None):
    """Plot multiple parameter variations for a PDF."""
    for config in configs:
        plot_pdf(config, obs)


def handle_error(func_name, error, filename=None):
    """Handle errors with dummy plots."""
    print(f"Error in {func_name}: {error}")
    if filename:
        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            f"Plot not available: {func_name}\nCheck documentation for correct usage",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title(f"{func_name} plot")
        save_plot(filename)


# ========================
# Basic PDFs
# ========================


def plot_gaussian():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.Gauss,
        "gauss",
        "Gaussian PDF",
        (-5, 5),
        [
            ("mu", [-3, -1.5, 0, 1.5, 3], {"sigma": 1.0}, r"\mu", lambda v: rf"\mu = {v}"),
            ("sigma", [0.3, 0.7, 1.0, 1.5, 2.5], {"mu": 0.0}, r"\sigma", lambda v: rf"\sigma = {v}"),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_exponential():
    obs = zfit.Space("x", limits=(0, 5))
    config = PDFConfig(
        pdf_class=zfit.pdf.Exponential,
        param_name="lam",
        param_values=[0.3, 0.6, 1.0, 1.5, 2.5],
        title=r"Exponential PDF with different \lambda values",
        filename="exponential_lambda.png",
        x_range=(0, 5),
        label_fn=lambda v: rf"\lambda = {v}",
    )
    plot_pdf(config, obs)


def plot_uniform():
    """Plot Uniform PDF with different ranges."""
    obs = zfit.Space("x", limits=(-5, 5))

    plt.figure()
    ranges = [(-4, 4), (-3, 3), (-2, 2), (-1, 3), (0, 4)]

    for low, high in ranges:
        uniform = zfit.pdf.Uniform(low=Parameter("low", low), high=Parameter("high", high), obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = uniform.pdf(x)
        plt.plot(x, y, label=f"Range: [{low}, {high}]")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Uniform PDF with different ranges")
    plt.legend()
    save_plot("uniform_range.png")


def plot_cauchy():
    obs = zfit.Space("x", limits=(-10, 10))
    configs = create_configs(
        zfit.pdf.Cauchy,
        "cauchy",
        "Cauchy PDF",
        (-10, 10),
        [
            ("m", [-3, -1.5, 0, 1.5, 3], {"gamma": 1.0}, "m", None),
            ("gamma", [0.3, 0.7, 1.0, 1.5, 2.5], {"m": 0.0}, r"\gamma", lambda v: rf"\gamma = {v}"),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_voigt():
    obs = zfit.Space("x", limits=(-10, 10))
    configs = create_configs(
        zfit.pdf.Voigt,
        "voigt",
        "Voigt PDF",
        (-10, 10),
        [
            ("sigma", [0.3, 0.7, 1.0, 1.5, 2.5], {"m": 0.0, "gamma": 1.0}, r"\sigma", lambda v: rf"\sigma = {v}"),
            ("gamma", [0.3, 0.7, 1.0, 1.5, 2.5], {"m": 0.0, "sigma": 1.0}, r"\gamma", lambda v: rf"\gamma = {v}"),
            ("m", [-3, -1.5, 0, 1.5, 3], {"sigma": 1.0, "gamma": 1.0}, "m", None),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_crystalball():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.CrystalBall,
        "crystalball",
        "CrystalBall PDF",
        (-5, 5),
        [
            (
                "alpha",
                [0.3, 0.7, 1.0, 1.5, 2.5],
                {"mu": 0.0, "sigma": 1.0, "n": 2.0},
                r"\alpha",
                lambda v: rf"\alpha = {v}",
            ),
            ("n", [1.0, 1.5, 2.0, 3.5, 5.0], {"mu": 0.0, "sigma": 1.0, "alpha": 1.0}, "n", None),
            (
                "mu",
                [-1.5, -0.75, 0.0, 0.75, 1.5],
                {"sigma": 1.0, "alpha": 1.0, "n": 2.0},
                r"\mu",
                lambda v: rf"\mu = {v}",
            ),
            (
                "sigma",
                [0.5, 0.75, 1.0, 1.25, 1.5],
                {"mu": 0.0, "alpha": 1.0, "n": 2.0},
                r"\sigma",
                lambda v: rf"\sigma = {v}",
            ),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_lognormal():
    obs = zfit.Space("x", limits=(0, 10))
    configs = create_configs(
        zfit.pdf.LogNormal,
        "lognormal",
        "LogNormal PDF",
        (0.1, 10),
        [
            ("mu", [-0.8, -0.4, 0.0, 0.4, 0.8], {"sigma": 0.5}, r"\mu", lambda v: rf"\mu = {v}"),
            ("sigma", [0.2, 0.35, 0.5, 0.75, 1.0], {"mu": 0.0}, r"\sigma", lambda v: rf"\sigma = {v}"),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_bifurgauss():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.BifurGauss,
        "bifurgauss",
        "BifurGauss PDF",
        (-5, 5),
        [
            ("mu", [-1.5, -0.75, 0.0, 0.75, 1.5], {"sigmal": 1.0, "sigmar": 1.0}, r"\mu", lambda v: rf"\mu = {v}"),
            (
                "sigmal",
                [0.5, 0.75, 1.0, 1.25, 1.5],
                {"mu": 0.0, "sigmar": 1.0},
                r"\sigma_left",
                lambda v: rf"\sigma_left = {v}",
            ),
            (
                "sigmar",
                [0.5, 0.75, 1.0, 1.25, 1.5],
                {"mu": 0.0, "sigmal": 1.0},
                r"\sigma_right",
                lambda v: rf"\sigma_right = {v}",
            ),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_poisson():
    """Plot Poisson PDF with different lambda values."""
    obs = zfit.Space("x", limits=(0, 20))
    plt.figure()

    x = np.arange(0, 20)
    for lambda_val in [1.0, 3.0, 5.0, 7.0, 10.0]:
        poisson = zfit.pdf.Poisson(lam=Parameter("lambda", lambda_val), obs=obs)
        y = poisson.pdf(x)
        plt.step(x, y, where="mid", label=rf"\lambda = {lambda_val}")

    plt.xlabel("x")
    plt.ylabel("Probability mass")
    plt.title(r"Poisson PDF with different \lambda values")
    plt.legend()
    save_plot("poisson_lambda.png")


def plot_qgauss():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.QGauss,
        "qgauss",
        "QGauss PDF",
        (-5, 5),
        [
            ("mu", [-1.5, -0.75, 0.0, 0.75, 1.5], {"sigma": 1.0, "q": 1.5}, r"\mu", lambda v: rf"\mu = {v}"),
            ("sigma", [0.5, 0.75, 1.0, 1.25, 1.5], {"mu": 0.0, "q": 1.5}, r"\sigma", lambda v: rf"\sigma = {v}"),
            ("q", [1.1, 1.3, 1.5, 1.7, 2.0], {"mu": 0.0, "sigma": 1.0}, "q", None),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_johnsonsu():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.JohnsonSU,
        "johnsonsu",
        "JohnsonSU PDF",
        (-5, 5),
        [
            (
                "mu",
                [-1.5, -0.75, 0.0, 0.75, 1.5],
                {"lambd": 1.0, "gamma": 1.0, "delta": 1.0},
                r"\mu",
                lambda v: rf"\mu = {v}",
            ),
            (
                "gamma",
                [0.0, 0.5, 1.0, 1.5, 2.0],
                {"mu": 0.0, "lambd": 1.0, "delta": 1.0},
                r"\gamma",
                lambda v: rf"\gamma = {v}",
            ),
            (
                "delta",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "lambd": 1.0, "gamma": 1.0},
                r"\delta",
                lambda v: rf"\delta = {v}",
            ),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_generalizedgauss():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.GeneralizedGauss,
        "generalizedgauss",
        "GeneralizedGauss PDF",
        (-5, 5),
        [
            ("mu", [-1.5, -0.75, 0.0, 0.75, 1.5], {"sigma": 1.0, "beta": 2.0}, r"\mu", lambda v: rf"\mu = {v}"),
            ("sigma", [0.5, 0.75, 1.0, 1.5, 2.0], {"mu": 0.0, "beta": 2.0}, r"\sigma", lambda v: rf"\sigma = {v}"),
            ("beta", [1.0, 1.5, 2.0, 3.0, 4.0], {"mu": 0.0, "sigma": 1.0}, r"\beta", lambda v: rf"\beta = {v}"),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_truncatedgauss():
    """Plot TruncatedGauss PDF with different parameters."""
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different mu values
    configs = create_configs(
        zfit.pdf.TruncatedGauss,
        "truncatedgauss",
        "TruncatedGauss PDF",
        (-5, 5),
        [
            (
                "mu",
                [-1.5, -0.75, 0.0, 0.75, 1.5],
                {"sigma": 1.0, "low": -2.0, "high": 2.0},
                r"\mu",
                lambda v: rf"\mu = {v}",
            ),
            (
                "sigma",
                [0.5, 0.75, 1.0, 1.25, 1.5],
                {"mu": 0.0, "low": -2.0, "high": 2.0},
                r"\sigma",
                lambda v: rf"\sigma = {v}",
            ),
        ],
    )
    plot_multiple_configs(configs, obs)

    # Plot with different ranges
    plt.figure()
    ranges = [(-1.0, 1.0), (-2.0, 2.0), (-0.5, 2.0), (-1.5, 0.5), (-0.5, 1.5)]
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)

    for low_val, high_val in ranges:
        truncgauss = zfit.pdf.TruncatedGauss(
            mu=mu, sigma=sigma, low=Parameter("low", low_val), high=Parameter("high", high_val), obs=obs
        )
        x = np.linspace(-5, 5, 1000)
        y = truncgauss.pdf(x)
        plt.plot(x, y, label=f"Range: [{low_val}, {high_val}]")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("TruncatedGauss PDF with different truncation ranges")
    plt.legend()
    save_plot("truncatedgauss_range.png")


def plot_chisquared():
    obs = zfit.Space("x", limits=(0, 20))
    config = PDFConfig(
        pdf_class=zfit.pdf.ChiSquared,
        param_name="ndof",
        param_values=[1, 2, 3, 4, 5],
        title="ChiSquared PDF with different ndof values",
        filename="chisquared_ndof.png",
        x_range=(0.1, 20),
    )
    plot_pdf(config, obs)


def plot_studentt():
    obs = zfit.Space("x", limits=(-5, 5))
    config = PDFConfig(
        pdf_class=zfit.pdf.StudentT,
        param_name="ndof",
        param_values=[1, 2, 3, 5, 10],
        title="StudentT PDF with different ndof values",
        filename="studentt_ndof.png",
        x_range=(-5, 5),
        fixed_params={"mu": 0.0, "sigma": 1.0},
    )
    plot_pdf(config, obs)


def plot_gamma():
    obs = zfit.Space("x", limits=(0, 10))
    configs = create_configs(
        zfit.pdf.Gamma,
        "gamma",
        "Gamma PDF",
        (0.1, 10),
        [
            ("gamma", [1.0, 1.5, 2.0, 3.5, 5.0], {"mu": 0.0, "beta": 1.0}, r"\gamma", lambda v: rf"\gamma = {v}"),
            ("beta", [0.5, 0.75, 1.0, 1.5, 2.0], {"mu": 0.0, "gamma": 2.0}, r"\beta", lambda v: rf"\beta = {v}"),
        ],
    )
    plot_multiple_configs(configs, obs)


# ========================
# Polynomial PDFs
# ========================


def plot_polynomial(pdf_class, name, x_range, degree_values=None):
    """Generic function to plot polynomial PDFs."""
    if degree_values is None:
        degree_values = [2, 3, 5, 6]
    try:
        # Create the observable
        obs = zfit.Space("x", limits=x_range)

        # Plot with different degrees
        plt.figure()
        for degree in degree_values:
            coeffs = [Parameter(f"c{i}", 1.0 if i == 0 else 0.3) for i in range(degree + 1)]
            poly = pdf_class(obs=obs, coeffs=coeffs)
            x = np.linspace(x_range[0], x_range[1], 1000)
            y = poly.pdf(x)
            plt.plot(x, y, label=f"Degree {degree}")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title(f"{name.capitalize()} PDF with different degrees")
        plt.legend()
        save_plot(f"{name}_degree.png")

        # Plot with different coefficient patterns
        plt.figure()
        patterns = [
            [1.0, 0.0, 0.0, 0.0],  # Constant
            [1.0, 0.5, 0.0, 0.0],  # Linear
            [1.0, 0.0, 0.5, 0.0],  # Quadratic
            [1.0, 0.0, 0.0, 0.5],  # Cubic
        ]

        for pattern in patterns:
            coeffs = [Parameter(f"c{j}", val) for j, val in enumerate(pattern)]
            poly = pdf_class(obs=obs, coeffs=coeffs)
            x = np.linspace(x_range[0], x_range[1], 1000)
            y = poly.pdf(x)
            plt.plot(x, y, label=f"Pattern: {pattern}")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title(f"{name.capitalize()} PDF with different coefficient patterns")
        plt.legend()
        save_plot(f"{name}_patterns.png")
    except Exception as e:
        handle_error(f"{name} polynomial", e, f"{name}_degree.png")


def plot_bernstein():
    """Plot Bernstein PDF with different degrees and coefficient patterns."""
    try:
        obs = zfit.Space("x", limits=(0, 1))

        # Plot with different degrees
        plt.figure()
        for degree in [2, 3, 5, 6]:
            # Create coefficients with a simple pattern
            coeffs = []
            for i in range(degree + 1):
                val = 1.0 - abs(i - degree / 2) / (degree / 2)  # Special pattern for Bernstein
                coeffs.append(Parameter(f"c{i}", val))

            bernstein = zfit.pdf.Bernstein(obs=obs, coeffs=coeffs)
            x = np.linspace(0, 1, 1000)
            y = bernstein.pdf(x)
            plt.plot(x, y, label=f"Degree {degree}")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("Bernstein PDF with different degrees")
        plt.legend()
        save_plot("bernstein_degree.png")

        # Plot with different coefficient patterns
        plt.figure()
        patterns = [
            [1.0, 0.2, 0.2, 1.0],  # U-shape
            [0.2, 1.0, 1.0, 0.2],  # Inverted U-shape
            [0.2, 0.5, 1.0, 0.2],  # Increasing then decreasing
            [0.5, 0.5, 0.5, 0.5],  # Flat
            [1.0, 0.7, 0.4, 0.1],  # Decreasing
        ]

        for pattern in patterns:
            coeffs = [Parameter(f"c{j}", val) for j, val in enumerate(pattern)]
            bernstein = zfit.pdf.Bernstein(obs=obs, coeffs=coeffs)
            x = np.linspace(0, 1, 1000)
            y = bernstein.pdf(x)
            plt.plot(x, y, label=f"Pattern: {pattern}")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("Bernstein PDF with different coefficient patterns")
        plt.legend()
        save_plot("bernstein_patterns.png")
    except Exception as e:
        handle_error("Bernstein", e, "bernstein_degree.png")


def plot_chebyshev():
    plot_polynomial(zfit.pdf.Chebyshev, "chebyshev", (-1, 1), [2, 3, 5, 6, 7, 8])


def plot_legendre():
    plot_polynomial(zfit.pdf.Legendre, "legendre", (-1, 1), [2, 3, 5, 6, 7, 8])


def plot_chebyshev2():
    plot_polynomial(zfit.pdf.Chebyshev2, "chebyshev2", (-1, 1), [2, 3, 5, 6, 7, 8])


def plot_hermite():
    plot_polynomial(zfit.pdf.Hermite, "hermite", (-5, 5))


def plot_laguerre():
    plot_polynomial(zfit.pdf.Laguerre, "laguerre", (0, 10))


def plot_recursivepolynomial():
    try:
        plot_polynomial(zfit.pdf.RecursivePolynomial, "recursivepolynomial", (-1, 1))
    except Exception:
        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "RecursivePolynomial plots not available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("RecursivePolynomial PDF plots")
        save_plot("recursivepolynomial_degree.png")
        save_plot("recursivepolynomial_patterns.png")


# ========================
# Physics PDFs
# ========================


def plot_doublecb():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.DoubleCB,
        "doublecb",
        "DoubleCB PDF",
        (-5, 5),
        [
            (
                "alphal",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "sigma": 1.0, "nl": 2.0, "alphar": 1.0, "nr": 2.0},
                r"\alpha_L",
                lambda v: rf"\alpha_L = {v}",
            ),
            (
                "alphar",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "sigma": 1.0, "nl": 2.0, "alphal": 1.0, "nr": 2.0},
                r"\alpha_R",
                lambda v: rf"\alpha_R = {v}",
            ),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_gaussexptail():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.GaussExpTail,
        "gaussexptail",
        "GaussExpTail PDF",
        (-5, 5),
        [
            ("alpha", [0.5, 0.75, 1.0, 1.5, 2.0], {"mu": 0.0, "sigma": 1.0}, r"\alpha", lambda v: rf"\alpha = {v}"),
            ("sigma", [0.5, 0.75, 1.0, 1.5, 2.0], {"mu": 0.0, "alpha": 1.0}, r"\sigma", lambda v: rf"\sigma = {v}"),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_generalizedcb():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.GeneralizedCB,
        "generalizedcb",
        "GeneralizedCB PDF",
        (-5, 5),
        [
            (
                "alphal",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "sigmal": 1.0, "nl": 2.0, "sigmar": 1.0, "alphar": 1.0, "nr": 2.0},
                r"\alpha_L",
                lambda v: rf"\alpha_L = {v}",
            ),
            (
                "nl",
                [1.0, 1.5, 2.0, 3.5, 5.0],
                {"mu": 0.0, "sigmal": 1.0, "alphal": 1.0, "sigmar": 1.0, "alphar": 1.0, "nr": 2.0},
                "nL",
                lambda v: f"nL = {v}",
            ),
            (
                "alphar",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "sigmal": 1.0, "alphal": 1.0, "nl": 2.0, "sigmar": 1.0, "nr": 2.0},
                r"\alpha_R",
                lambda v: rf"\alpha_R = {v}",
            ),
        ],
    )
    plot_multiple_configs(configs, obs)


def plot_generalizedgaussexptail():
    obs = zfit.Space("x", limits=(-5, 5))
    configs = create_configs(
        zfit.pdf.GeneralizedGaussExpTail,
        "generalizedgaussexptail",
        "GeneralizedGaussExpTail PDF",
        (-5, 5),
        [
            (
                "alphal",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "sigmal": 1.0, "sigmar": 1.0, "alphar": 1.0},
                r"\alpha_L",
                lambda v: rf"\alpha_L = {v}",
            ),
            (
                "alphar",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "sigmal": 1.0, "alphal": 1.0, "sigmar": 1.0},
                r"\alpha_R",
                lambda v: rf"\alpha_R = {v}",
            ),
            (
                "sigmal",
                [0.5, 0.75, 1.0, 1.5, 2.0],
                {"mu": 0.0, "alphal": 1.0, "sigmar": 1.0, "alphar": 1.0},
                r"\sigma_L",
                lambda v: rf"\sigma_L = {v}",
            ),
        ],
    )
    plot_multiple_configs(configs, obs)


# ========================
# Binned PDFs
# ========================


def create_dummy_plot(title, filename):
    """Create a simple dummy plot when a complex plot can't be created."""
    plt.figure(figsize=(10, 6))
    plt.text(
        0.5,
        0.5,
        f"Plot not available: {title}\nCheck documentation for correct usage",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(title)
    save_plot(filename)


def plot_histogrampdf():
    """Plot histogram PDFs with different shapes."""
    try:
        import hist

        from zfit._data.binneddatav1 import BinnedData

        # Create the observable
        zfit.Space("x", limits=(0, 10))

        plt.figure()

        # Create histograms with different shapes
        x_values = np.linspace(0, 10, 100)
        bins = np.linspace(0, 10, 20)

        # Gaussian-like histogram
        gaussian_hist = np.exp(-0.5 * ((x_values - 5) / 1.5) ** 2)
        values, _ = np.histogram(
            np.random.choice(x_values, size=10000, p=gaussian_hist / np.sum(gaussian_hist)), bins=bins
        )

        h1 = hist.Hist(hist.axis.Regular(19, 0, 10, name="x"))
        h1[:] = values
        binned_data = BinnedData.from_hist(h1)
        hist_pdf = zfit.pdf.HistogramPDF(binned_data)

        # Bimodal histogram
        bimodal_hist = np.exp(-0.5 * ((x_values - 3) / 1.0) ** 2) + np.exp(-0.5 * ((x_values - 7) / 1.0) ** 2)
        values2, _ = np.histogram(
            np.random.choice(x_values, size=10000, p=bimodal_hist / np.sum(bimodal_hist)), bins=bins
        )

        h2 = hist.Hist(hist.axis.Regular(19, 0, 10, name="x"))
        h2[:] = values2
        binned_data2 = BinnedData.from_hist(h2)
        hist_pdf2 = zfit.pdf.HistogramPDF(binned_data2)

        # Uniform histogram
        uniform_hist = np.ones_like(x_values)
        values3, _ = np.histogram(
            np.random.choice(x_values, size=10000, p=uniform_hist / np.sum(uniform_hist)), bins=bins
        )

        h3 = hist.Hist(hist.axis.Regular(19, 0, 10, name="x"))
        h3[:] = values3
        binned_data3 = BinnedData.from_hist(h3)
        hist_pdf3 = zfit.pdf.HistogramPDF(binned_data3)

        # Plot all PDFs
        x = np.linspace(0, 10, 1000)
        plt.plot(x, hist_pdf.pdf(x), label="Gaussian-like histogram")
        plt.plot(x, hist_pdf2.pdf(x), label="Bimodal histogram")
        plt.plot(x, hist_pdf3.pdf(x), label="Uniform histogram")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("HistogramPDF with different shapes")
        plt.legend()
        save_plot("histogrampdf_shapes.png")
    except Exception as e:
        handle_error("HistogramPDF", e, "histogrampdf_shapes.png")


def plot_binwisescalemodifier():
    """Plot BinwiseScaleModifier with different scale patterns."""
    try:
        import hist

        from zfit._data.binneddatav1 import BinnedData

        # Create the observable
        zfit.Space("x", limits=(0, 10))

        plt.figure()

        # Create a base histogram
        x_values = np.linspace(0, 10, 100)
        gaussian_hist = np.exp(-0.5 * ((x_values - 5) / 1.5) ** 2)

        # Create bins and values
        bins = np.linspace(0, 10, 20)
        values, _ = np.histogram(
            np.random.choice(x_values, size=10000, p=gaussian_hist / np.sum(gaussian_hist)), bins=bins
        )

        h1 = hist.Hist(hist.axis.Regular(19, 0, 10, name="x"))
        h1[:] = values
        binned_data = BinnedData.from_hist(h1)
        hist_pdf = zfit.pdf.HistogramPDF(binned_data)

        # Plot the original PDF
        x = np.linspace(0, 10, 1000)
        y = hist_pdf.pdf(x)
        plt.plot(x, y, label="Original histogram")

        # Apply scale modifiers
        center_scaled_pdf = zfit.pdf.BinwiseScaleModifier(hist_pdf, True)
        y_center = center_scaled_pdf.pdf(x)
        plt.plot(x, y_center, label="Modified PDF 1")

        tail_scaled_pdf = zfit.pdf.BinwiseScaleModifier(hist_pdf, True)
        y_tail = tail_scaled_pdf.pdf(x)
        plt.plot(x, y_tail, label="Modified PDF 2")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("BinwiseScaleModifier with different scale patterns")
        plt.legend()
        save_plot("binwisescalemodifier_patterns.png")
    except Exception as e:
        handle_error("BinwiseScaleModifier", e, "binwisescalemodifier_patterns.png")


def plot_binnedfromunbinnedpdf():
    """Create a dummy plot for BinnedFromUnbinnedPDF."""
    create_dummy_plot("BinnedFromUnbinnedPDF comparison", "binnedfromunbinnedpdf_comparison.png")


def plot_splinemorphingpdf():
    """Plot SplineMorphingPDF with different parameter values."""
    try:
        from zfit.data import BinnedData

        # Create the observable
        zfit.Space("x", limits=(-5, 5))

        plt.figure()

        # Create template histograms with different means
        bins = np.linspace(-5, 5, 30)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        means = [-2.0, 0.0, 2.0]
        templates = []
        hist_pdfs = []

        for mean in means:
            # Create a Gaussian distribution
            values = np.exp(-0.5 * ((bin_centers - mean) / 1.0) ** 2)
            templates.append(values)

            # Create binned data
            binned_space = zfit.Space("x", binning=zfit.binned.RegularBinning(len(bin_centers), -5, 5, name="x"))
            binned_data = BinnedData.from_tensor(space=binned_space, values=values)

            # Create histogram PDF
            hist_pdf = zfit.pdf.HistogramPDF(binned_data)
            hist_pdfs.append(hist_pdf)

        # Create morphing parameter and PDF
        morph_param = Parameter("morph", 0.0, -2.0, 2.0)
        spline_pdf = zfit.pdf.SplineMorphingPDF(morph_param, hist_pdfs)

        # Plot templates
        x = np.linspace(-5, 5, 1000)
        for i, mean in enumerate(means):
            plt.plot(
                bin_centers,
                templates[i] / np.sum(templates[i]) * (bins[1] - bins[0]) * len(bins),
                "o",
                label=f"Template (mean={mean})",
            )

        # Plot morphed PDFs
        morph_values = [-1.5, -0.5, 0.5, 1.5]
        for morph_val in morph_values:
            morph_param.set_value(morph_val)
            y = spline_pdf.pdf(x)
            plt.plot(x, y, label=f"Morphed (param={morph_val})")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("SplineMorphingPDF with different parameter values")
        plt.legend()
        save_plot("splinemorphingpdf_morphing.png")
    except Exception as e:
        handle_error("SplineMorphingPDF", e, "splinemorphingpdf_morphing.png")


def plot_binnedsumpdf():
    """Plot BinnedSumPDF with different component fractions."""
    try:
        from zfit._data.binneddatav1 import BinnedData

        # Create the observable
        zfit.Space("x", limits=(0, 10))

        plt.figure()

        # Create bins and components
        bins = np.linspace(0, 10, 20)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        binned_space = zfit.Space("x", binning=zfit.binned.RegularBinning(len(bin_centers), 0, 10, name="x"))

        # Gaussian component
        gaussian_values = np.exp(-0.5 * ((bin_centers - 3) / 1.0) ** 2)
        gaussian_data = BinnedData.from_tensor(space=binned_space, values=gaussian_values)
        gaussian_hist = zfit.pdf.HistogramPDF(gaussian_data)

        # Exponential component
        exponential_values = np.exp(-bin_centers / 3)
        exponential_data = BinnedData.from_tensor(space=binned_space, values=exponential_values)
        exponential_hist = zfit.pdf.HistogramPDF(exponential_data)

        # Uniform component
        uniform_values = np.ones_like(bin_centers)
        uniform_data = BinnedData.from_tensor(space=binned_space, values=uniform_values)
        uniform_hist = zfit.pdf.HistogramPDF(uniform_data)

        # Create fractions and sum PDF
        frac1 = Parameter("frac1", 0.6)
        frac2 = Parameter("frac2", 0.3)
        binned_sum = zfit.pdf.BinnedSumPDF([gaussian_hist, exponential_hist, uniform_hist], fracs=[frac1, frac2])

        # Plot components and sum
        x = np.linspace(0, 10, 1000)
        plt.plot(x, gaussian_hist.pdf(x), label="Gaussian component")
        plt.plot(x, exponential_hist.pdf(x), label="Exponential component")
        plt.plot(x, uniform_hist.pdf(x), label="Uniform component")
        plt.plot(x, binned_sum.pdf(x), label="Binned Sum PDF", linewidth=2)

        # Plot with different fractions
        frac_sets = [(0.8, 0.1), (0.4, 0.4), (0.2, 0.7)]
        for f1, f2 in frac_sets:
            frac1.set_value(f1)
            frac2.set_value(f2)
            y_sum_alt = binned_sum.pdf(x)
            plt.plot(x, y_sum_alt, linestyle="--", label=f"Sum with fracs=({f1:.1f}, {f2:.1f}, {1 - f1 - f2:.1f})")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("BinnedSumPDF with different component fractions")
        plt.legend()
        save_plot("binnedsumpdf_fractions.png")
    except Exception as e:
        handle_error("BinnedSumPDF", e, "binnedsumpdf_fractions.png")


def plot_splinepdf():
    """Plot SplinePDF with different shapes."""
    try:
        from zfit._data.binneddatav1 import BinnedData

        # Create the observable
        obs = zfit.Space("x", limits=(0, 10))

        plt.figure()

        # Define shapes
        x_points = np.array([0, 2, 4, 6, 8, 10])
        y_gauss = np.array([0.05, 0.1, 0.4, 0.4, 0.1, 0.05])
        y_increasing = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        y_bimodal = np.array([0.05, 0.3, 0.1, 0.1, 0.3, 0.05])

        # Create binned data
        binned_space = zfit.Space("x", binning=zfit.binned.RegularBinning(len(x_points) - 1, 0, 10, name="x"))
        gauss_data = BinnedData.from_tensor(space=binned_space, values=y_gauss[:-1])
        increasing_data = BinnedData.from_tensor(space=binned_space, values=y_increasing[:-1])
        bimodal_data = BinnedData.from_tensor(space=binned_space, values=y_bimodal[:-1])

        # Create histogram PDFs
        gauss_hist = zfit.pdf.HistogramPDF(gauss_data)
        increasing_hist = zfit.pdf.HistogramPDF(increasing_data)
        bimodal_hist = zfit.pdf.HistogramPDF(bimodal_data)

        # Create spline PDFs
        spline_gauss = zfit.pdf.SplinePDF(gauss_hist, obs=obs)
        spline_increasing = zfit.pdf.SplinePDF(increasing_hist, obs=obs)
        spline_bimodal = zfit.pdf.SplinePDF(bimodal_hist, obs=obs)

        # Plot splines
        x = np.linspace(0, 10, 1000)

        plt.plot(x_points, y_gauss, "o", label="Gaussian-like points")
        plt.plot(x, spline_gauss.pdf(x), label="Gaussian-like spline")

        plt.plot(x_points, y_increasing, "s", label="Increasing points")
        plt.plot(x, spline_increasing.pdf(x), label="Increasing spline")

        plt.plot(x_points, y_bimodal, "^", label="Bimodal points")
        plt.plot(x, spline_bimodal.pdf(x), label="Bimodal spline")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("SplinePDF with different shapes")
        plt.legend()
        save_plot("splinepdf_shapes.png")
    except Exception as e:
        handle_error("SplinePDF", e, "splinepdf_shapes.png")


def plot_unbinnedfromibinnedpdf():
    """Plot UnbinnedFromBinnedPDF comparison."""
    try:
        from zfit._data.binneddatav1 import BinnedData

        # Create the observable
        zfit.Space("x", limits=(0, 10))

        plt.figure()

        # Create bins and components
        bins = np.linspace(0, 10, 20)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        binned_space = zfit.Space("x", binning=zfit.binned.RegularBinning(len(bin_centers), 0, 10, name="x"))

        # Bimodal shape
        values = np.exp(-0.5 * ((bin_centers - 3) / 1.0) ** 2) + np.exp(-0.5 * ((bin_centers - 7) / 1.0) ** 2)
        binned_data = BinnedData.from_tensor(space=binned_space, values=values)
        hist_pdf = zfit.pdf.HistogramPDF(binned_data)
        unbinned_pdf = zfit.pdf.UnbinnedFromBinnedPDF(hist_pdf)

        # Exponential shape
        values2 = np.exp(-bin_centers / 3)
        binned_data2 = BinnedData.from_tensor(space=binned_space, values=values2)
        hist_pdf2 = zfit.pdf.HistogramPDF(binned_data2)
        unbinned_pdf2 = zfit.pdf.UnbinnedFromBinnedPDF(hist_pdf2)

        # Plot PDFs
        x = np.linspace(0, 10, 1000)
        plt.plot(x, hist_pdf.pdf(x), label="Original histogram PDF", linestyle="--")
        plt.plot(x, unbinned_pdf.pdf(x), label="Unbinned from binned PDF")
        plt.plot(x, hist_pdf2.pdf(x), label="Original exponential histogram", linestyle="--")
        plt.plot(x, unbinned_pdf2.pdf(x), label="Unbinned from exponential")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("UnbinnedFromBinnedPDF comparison")
        plt.legend()
        save_plot("unbinnedfromibinnedpdf_comparison.png")
    except Exception as e:
        handle_error("UnbinnedFromBinnedPDF", e, "unbinnedfromibinnedpdf_comparison.png")


# ========================
# Composed PDFs
# ========================


def plot_sumpdf():
    """Plot SumPDF with different component fractions."""
    try:
        obs = zfit.Space("x", limits=(-5, 5))

        plt.figure()

        # Create components
        mu1 = Parameter("mu1", -1.5)
        sigma1 = Parameter("sigma1", 0.5)
        gauss1 = zfit.pdf.Gauss(mu=mu1, sigma=sigma1, obs=obs)

        mu2 = Parameter("mu2", 1.5)
        sigma2 = Parameter("sigma2", 0.5)
        gauss2 = zfit.pdf.Gauss(mu=mu2, sigma=sigma2, obs=obs)

        lambda_param = Parameter("lambda", 0.5)
        exp_pdf = zfit.pdf.Exponential(lambda_param, obs=obs)

        # Create fractions and sum PDF
        frac1 = Parameter("frac1", 0.6)
        frac2 = Parameter("frac2", 0.3)
        sum_pdf = zfit.pdf.SumPDF([gauss1, gauss2, exp_pdf], fracs=[frac1, frac2])

        # Plot components and sum
        x = np.linspace(-5, 5, 1000)
        plt.plot(x, gauss1.pdf(x), label="Gaussian 1")
        plt.plot(x, gauss2.pdf(x), label="Gaussian 2")
        plt.plot(x, exp_pdf.pdf(x), label="Exponential")
        plt.plot(x, sum_pdf.pdf(x), label="Sum PDF", linewidth=2)

        # Plot with different fractions
        frac_sets = [(0.8, 0.1), (0.4, 0.4), (0.2, 0.7)]
        for f1, f2 in frac_sets:
            frac1.set_value(f1)
            frac2.set_value(f2)
            y_sum_alt = sum_pdf.pdf(x)
            plt.plot(x, y_sum_alt, linestyle="--", label=f"Sum with fracs=({f1:.1f}, {f2:.1f}, {1 - f1 - f2:.1f})")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("SumPDF with different component fractions")
        plt.legend()
        save_plot("sumpdf_fractions.png")
    except Exception as e:
        handle_error("SumPDF", e, "sumpdf_fractions.png")


def plot_productpdf():
    """Plot ProductPDF as 2D PDF."""
    try:
        obs_x = zfit.Space("x", limits=(-5, 5))
        obs_y = zfit.Space("y", limits=(-5, 5))

        # Create Gaussian PDFs for x and y
        mu_x = Parameter("mu_x", 0.0)
        sigma_x = Parameter("sigma_x", 1.0)
        gauss_x = zfit.pdf.Gauss(mu=mu_x, sigma=sigma_x, obs=obs_x)

        mu_y = Parameter("mu_y", 0.0)
        sigma_y = Parameter("sigma_y", 1.0)
        gauss_y = zfit.pdf.Gauss(mu=mu_y, sigma=sigma_y, obs=obs_y)

        # Create product PDF
        prod_pdf = zfit.pdf.ProductPDF([gauss_x, gauss_y])

        # Plot symmetric 2D Gaussian
        plt.figure()
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.flatten(), Y.flatten()])
        Z_prod = prod_pdf.pdf(points).numpy().reshape(X.shape)

        plt.contourf(X, Y, Z_prod, levels=20, cmap="viridis")
        plt.colorbar(label="Probability density")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("ProductPDF: 2D Gaussian")
        plt.axis("equal")
        save_plot("productpdf_2d_gaussian.png")

        # Plot asymmetric 2D Gaussian
        plt.figure()
        sigma_x.set_value(0.5)
        sigma_y.set_value(2.0)
        Z_prod_asym = prod_pdf.pdf(points).numpy().reshape(X.shape)

        plt.contourf(X, Y, Z_prod_asym, levels=20, cmap="viridis")
        plt.colorbar(label="Probability density")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("ProductPDF: Asymmetric 2D Gaussian")
        plt.axis("equal")
        save_plot("productpdf_asymmetric.png")
    except Exception as e:
        handle_error("ProductPDF", e, "productpdf_2d_gaussian.png")


def plot_productpdf_1d():
    """Plot ProductPDF as multiplication of PDFs in the same dimension."""
    try:
        obs = zfit.Space("x", limits=(-5, 5))

        plt.figure()

        # Create Gaussian
        mu = Parameter("mu", 0.0)
        sigma = Parameter("sigma", 1.0)
        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

        # Create exponential
        lambda_param = Parameter("lambda", 0.5)
        expo = zfit.pdf.Exponential(lambda_=lambda_param, obs=obs)

        # Create product
        prod_pdf = zfit.pdf.ProductPDF([gauss, expo])

        # Plot all PDFs
        x = np.linspace(-5, 5, 1000)
        plt.plot(x, gauss.pdf(x), label="Gaussian PDF")
        plt.plot(x, expo.pdf(x), label="Exponential PDF")
        plt.plot(x, prod_pdf.pdf(x), label="Product PDF")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("ProductPDF: Multiplying PDFs in the same dimension")
        plt.legend()
        save_plot("productpdf_1d_multiplication.png")
    except Exception as e:
        handle_error("ProductPDF 1D", e, "productpdf_1d_multiplication.png")


def plot_fftconvpdf():
    """Plot FFTConvPDFV1 with different resolutions and signal shapes."""
    try:
        obs = zfit.Space("x", limits=(-10, 10))

        # Plot convolution with different resolutions
        plt.figure()

        mu_signal = Parameter("mu_signal", 0.0)
        sigma_signal = Parameter("sigma_signal", 0.5)
        signal = zfit.pdf.Gauss(mu=mu_signal, sigma=sigma_signal, obs=obs)

        mu_res = Parameter("mu_res", 0.0)
        sigma_res_values = [0.5, 1.0, 2.0]

        x = np.linspace(-10, 10, 1000)
        plt.plot(x, signal.pdf(x), label="Signal (narrow Gaussian)")

        for sigma_res_val in sigma_res_values:
            sigma_res = Parameter("sigma_res", sigma_res_val)
            resolution = zfit.pdf.Gauss(mu=mu_res, sigma=sigma_res, obs=obs)

            conv = zfit.pdf.FFTConvPDFV1(signal, resolution)
            y_conv = conv.pdf(x)
            plt.plot(x, y_conv, label=rf"Convolution with \sigma_res = {sigma_res_val}")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("FFTConvPDFV1: Gaussian convolved with different resolutions")
        plt.legend()
        save_plot("fftconvpdf_resolutions.png")

        # Plot convolution with different signal shapes
        plt.figure()

        sigma_res = Parameter("sigma_res", 1.0)
        resolution = zfit.pdf.Gauss(mu=mu_res, sigma=sigma_res, obs=obs)

        # Double Gaussian signal
        mu1 = Parameter("mu1", -2.0)
        sigma1 = Parameter("sigma1", 0.5)
        mu2 = Parameter("mu2", 2.0)
        sigma2 = Parameter("sigma2", 0.5)

        gauss1 = zfit.pdf.Gauss(mu=mu1, sigma=sigma1, obs=obs)
        gauss2 = zfit.pdf.Gauss(mu=mu2, sigma=sigma2, obs=obs)

        frac = Parameter("frac", 0.5)
        double_gauss = zfit.pdf.SumPDF([gauss1, gauss2], fracs=[frac])

        # Crystal Ball signal
        mu_cb = Parameter("mu_cb", 0.0)
        sigma_cb = Parameter("sigma_cb", 0.5)
        alpha_cb = Parameter("alpha_cb", 1.0)
        n_cb = Parameter("n_cb", 2.0)

        cb = zfit.pdf.CrystalBall(mu=mu_cb, sigma=sigma_cb, alpha=alpha_cb, n=n_cb, obs=obs)

        # Plot original and convolved PDFs
        conv_double = zfit.pdf.FFTConvPDFV1(double_gauss, resolution)
        conv_cb = zfit.pdf.FFTConvPDFV1(cb, resolution)

        plt.plot(x, double_gauss.pdf(x), label="Double Gaussian signal")
        plt.plot(x, conv_double.pdf(x), label="Convolved double Gaussian")
        plt.plot(x, cb.pdf(x), label="Crystal Ball signal")
        plt.plot(x, conv_cb.pdf(x), label="Convolved Crystal Ball")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("FFTConvPDFV1: Different signals convolved with Gaussian")
        plt.legend()
        save_plot("fftconvpdf_signals.png")
    except Exception as e:
        handle_error("FFTConvPDFV1", e, "fftconvpdf_resolutions.png")


def plot_conditionalpdf():
    """Create a dummy plot for ConditionalPDFV1."""
    create_dummy_plot("ConditionalPDFV1 comparison", "conditionalpdf_gaussian.png")
    create_dummy_plot("ConditionalPDFV1 width comparison", "conditionalpdf_width.png")


def plot_truncatedpdf():
    """Plot TruncatedPDF with different truncation ranges and base PDFs."""
    try:
        obs = zfit.Space("x", limits=(-10, 10))

        # Plot truncated Gaussians
        plt.figure()

        mu = Parameter("mu", 0.0)
        sigma = Parameter("sigma", 2.0)
        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

        x = np.linspace(-10, 10, 1000)
        plt.plot(x, gauss.pdf(x), label="Original Gaussian")

        limit_sets = [(-5, 5), (-2, 2), (0, 5), (-5, 0)]

        for low, high in limit_sets:
            trunc_space = zfit.Space("x", limits=(low, high))
            trunc_gauss = zfit.pdf.TruncatedPDF(gauss, limits=trunc_space)

            y_trunc = trunc_gauss.pdf(x)
            plt.plot(x, y_trunc, label=f"Truncated to [{low}, {high}]")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("TruncatedPDF: Gaussian with different truncation ranges")
        plt.legend()
        save_plot("truncatedpdf_gaussian.png")

        # Plot truncated different PDFs
        plt.figure()

        lambda_param = Parameter("lambda", 0.5)
        exp_pdf = zfit.pdf.Exponential(lambda_param, obs=obs)

        low_param = Parameter("low", -5.0)
        high_param = Parameter("high", 5.0)
        uniform = zfit.pdf.Uniform(low=low_param, high=high_param, obs=obs)

        plt.plot(x, exp_pdf.pdf(x), label="Original Exponential")
        plt.plot(x, uniform.pdf(x), label="Original Uniform")

        trunc_space = zfit.Space("x", limits=(-2, 2))
        trunc_exp = zfit.pdf.TruncatedPDF(exp_pdf, limits=trunc_space)
        trunc_uniform = zfit.pdf.TruncatedPDF(uniform, limits=trunc_space)

        plt.plot(x, trunc_exp.pdf(x), label="Truncated Exponential")
        plt.plot(x, trunc_uniform.pdf(x), label="Truncated Uniform")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("TruncatedPDF: Different PDFs truncated to [-2, 2]")
        plt.legend()
        save_plot("truncatedpdf_various.png")
    except Exception as e:
        handle_error("TruncatedPDF", e, "truncatedpdf_gaussian.png")


# ========================
# KDE PDFs
# ========================


def plot_kde():
    """Plot KDEs with different parameters."""
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    samples1 = np.random.normal(-1.5, 0.5, size=int(0.4 * n_samples))
    samples2 = np.random.normal(1.0, 0.7, size=int(0.6 * n_samples))
    samples = np.concatenate([samples1, samples2])
    data = zfit.Data.from_numpy(obs=obs, array=samples[:, np.newaxis])

    # Prepare the x values and true PDF for plotting
    x = np.linspace(-5, 5, 1000)
    true_pdf = 0.4 * np.exp(-0.5 * ((x + 1.5) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi)) + 0.6 * np.exp(
        -0.5 * ((x - 1.0) / 0.7) ** 2
    ) / (0.7 * np.sqrt(2 * np.pi))

    # Plot KDEs with different bandwidths
    try:
        plt.figure()
        plt.hist(samples, bins=30, density=True, alpha=0.3, label="Data histogram")
        plt.plot(x, true_pdf, "k--", label="True distribution")

        for bw in [0.1, 0.3, 0.8]:
            kde = zfit.pdf.KDE1DimExact(data=data, bandwidth=bw, obs=obs)
            y = kde.pdf(x)
            plt.plot(x, y, label=f"Bandwidth = {bw}")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("KDE with different bandwidth values")
        plt.legend()
        save_plot("kde_bandwidth.png")
    except Exception as e:
        handle_error("KDE bandwidth comparison", e, "kde_bandwidth.png")

    # Plot different KDE implementations
    try:
        plt.figure()
        plt.hist(samples, bins=30, density=True, alpha=0.3, label="Data histogram")
        plt.plot(x, true_pdf, "k--", label="True distribution")

        kde_exact = zfit.pdf.KDE1DimExact(data=data, bandwidth=0.3, obs=obs)
        kde_grid = zfit.pdf.KDE1DimGrid(data=data, bandwidth=0.3, obs=obs, num_grid_points=100)
        kde_fft = zfit.pdf.KDE1DimFFT(data=data, bandwidth=0.3, obs=obs, num_grid_points=100)
        kde_gaussian = zfit.pdf.GaussianKDE1DimV1(obs=obs, data=data, bandwidth=0.3)

        plt.plot(x, kde_exact.pdf(x), label="KDE1DimExact")
        plt.plot(x, kde_grid.pdf(x), label="KDE1DimGrid")
        plt.plot(x, kde_fft.pdf(x), label="KDE1DimFFT")
        plt.plot(x, kde_gaussian.pdf(x), label="GaussianKDE1DimV1")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("Different KDE implementations")
        plt.legend()
        save_plot("kde_implementations.png")
    except Exception as e:
        handle_error("KDE implementations", e, "kde_implementations.png")

    # Plot KDE1DimISJ implementation
    try:
        plt.figure()
        plt.hist(samples, bins=30, density=True, alpha=0.3, label="Data histogram")
        plt.plot(x, true_pdf, "k--", label="True distribution")

        kde_isj = zfit.pdf.KDE1DimISJ(data=data, obs=obs)
        plt.plot(x, kde_isj.pdf(x), label="KDE1DimISJ")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("KDE1DimISJ implementation")
        plt.legend()
        save_plot("kde_isj.png")
    except Exception as e:
        handle_error("KDE1DimISJ", e, "kde_isj.png")

    # Plot KDEs with different bandwidth methods
    try:
        plt.figure()
        plt.hist(samples, bins=30, density=True, alpha=0.3, label="Data histogram")
        plt.plot(x, true_pdf, "k--", label="True distribution")

        kde_scott = zfit.pdf.KDE1DimExact(data=data, bandwidth="scott", obs=obs)
        kde_silverman = zfit.pdf.KDE1DimExact(data=data, bandwidth="silverman", obs=obs)
        kde_isj_method = zfit.pdf.KDE1DimExact(data=data, bandwidth="isj", obs=obs)
        kde_adaptive_geom = zfit.pdf.KDE1DimExact(data=data, bandwidth="adaptive_geom", obs=obs)
        kde_adaptive_std = zfit.pdf.KDE1DimExact(data=data, bandwidth="adaptive_std", obs=obs)
        kde_adaptive_zfit = zfit.pdf.KDE1DimExact(data=data, bandwidth="adaptive_zfit", obs=obs)

        plt.plot(x, kde_scott.pdf(x), label="Scott's rule")
        plt.plot(x, kde_silverman.pdf(x), label="Silverman's rule")
        plt.plot(x, kde_isj_method.pdf(x), label="ISJ method")
        plt.plot(x, kde_adaptive_geom.pdf(x), label="Adaptive (geom)")
        plt.plot(x, kde_adaptive_std.pdf(x), label="Adaptive (std)")
        plt.plot(x, kde_adaptive_zfit.pdf(x), label="Adaptive (zfit)")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("KDE with different bandwidth methods")
        plt.legend()
        save_plot("kde_bandwidth_methods.png")
    except Exception as e:
        handle_error("KDE bandwidth methods", e, "kde_bandwidth_methods.png")

    # Plot KDEs with different kernel types
    try:
        plt.figure()
        plt.hist(samples, bins=30, density=True, alpha=0.3, label="Data histogram")
        plt.plot(x, true_pdf, "k--", label="True distribution")

        import tensorflow_probability as tfp

        tfd = tfp.distributions

        # Create KDEs with different kernel types
        kde_normal = zfit.pdf.KDE1DimExact(data=data, bandwidth=0.3, kernel=tfd.Normal, obs=obs)
        kde_laplace = zfit.pdf.KDE1DimExact(data=data, bandwidth=0.3, kernel=tfd.Laplace, obs=obs)
        kde_logistic = zfit.pdf.KDE1DimExact(data=data, bandwidth=0.3, kernel=tfd.Logistic, obs=obs)

        plt.plot(x, kde_normal.pdf(x), label="Normal kernel")
        plt.plot(x, kde_laplace.pdf(x), label="Laplace kernel")
        plt.plot(x, kde_logistic.pdf(x), label="Logistic kernel")

        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title("KDE with different kernel types")
        plt.legend()
        save_plot("kde_kernel.png")
    except Exception as e:
        handle_error("KDE kernel types", e, "kde_kernel.png")


# ========================
# Physics PDFs from zfit_physics
# ========================


def plot_physics_pdfs():
    """Plot PDFs from zfit_physics package if installed."""
    try:
        import zfit_physics.pdf

        print("zfit_physics is installed, generating physics PDF plots...")

        # Create the observable
        obs = zfit.Space("x", limits=(0, 10))

        # Define physics PDFs with their parameters
        physics_pdfs = [
            {
                "name": "Argus",
                "class": zfit_physics.pdf.Argus,
                "params": [
                    ("c", [0.5, 1.0, 2.0], {"m0": 5.0, "p": 0.5}, None, None, "argus_c.png"),
                    ("m0", [3.0, 5.0, 7.0], {"c": 1.0, "p": 0.5}, None, None, "argus_m0.png"),
                    ("p", [0.3, 0.5, 0.7], {"c": 1.0, "m0": 5.0}, None, None, "argus_p.png"),
                ],
            },
            {
                "name": "RelativisticBreitWigner",
                "class": zfit_physics.pdf.RelativisticBreitWigner,
                "params": [
                    ("m", [4.0, 5.0, 6.0], {"gamma": 0.5}, None, None, "rbw_m.png"),
                    ("gamma", [0.3, 0.5, 1.0], {"m": 5.0}, None, None, "rbw_gamma.png"),
                ],
            },
            {
                "name": "CMSShape",
                "class": zfit_physics.pdf.CMSShape,
                "params": [
                    ("m", [1.0, 2.0, 3.0], {"beta": 0.5, "gamma": 0.1}, None, None, "cms_m.png"),
                    ("beta", [0.3, 0.5, 0.7], {"m": 2.0, "gamma": 0.1}, None, None, "cms_beta.png"),
                    ("gamma", [0.05, 0.1, 0.2], {"m": 2.0, "beta": 0.5}, None, None, "cms_gamma.png"),
                ],
            },
            {
                "name": "Cruijff",
                "class": zfit_physics.pdf.Cruijff,
                "params": [
                    (
                        "mu",
                        [4.0, 5.0, 6.0],
                        {"sigmal": 1.0, "sigmar": 1.0, "alphal": 0.1, "alphar": 0.1},
                        None,
                        None,
                        "cruijff_mu.png",
                    ),
                    (
                        "sigmal",
                        [0.5, 1.0, 1.5],
                        {"mu": 5.0, "sigmar": 1.0, "alphal": 0.1, "alphar": 0.1},
                        None,
                        None,
                        "cruijff_sigmal.png",
                    ),
                    (
                        "sigmar",
                        [0.5, 1.0, 1.5],
                        {"mu": 5.0, "sigmal": 1.0, "alphal": 0.1, "alphar": 0.1},
                        None,
                        None,
                        "cruijff_sigmar.png",
                    ),
                    (
                        "alphal",
                        [0.05, 0.1, 0.2],
                        {"mu": 5.0, "sigmal": 1.0, "sigmar": 1.0, "alphar": 0.1},
                        None,
                        None,
                        "cruijff_alphal.png",
                    ),
                    (
                        "alphar",
                        [0.05, 0.1, 0.2],
                        {"mu": 5.0, "sigmal": 1.0, "sigmar": 1.0, "alphal": 0.1},
                        None,
                        None,
                        "cruijff_alphar.png",
                    ),
                ],
            },
            {
                "name": "ErfExp",
                "class": zfit_physics.pdf.ErfExp,
                "params": [
                    ("mu", [4.0, 5.0, 6.0], {"beta": 1.0, "gamma": 0.5, "n": 1.0}, None, None, "erfexp_mu.png"),
                    ("beta", [0.5, 1.0, 2.0], {"mu": 5.0, "gamma": 0.5, "n": 1.0}, None, None, "erfexp_beta.png"),
                    ("gamma", [0.3, 0.5, 0.7], {"mu": 5.0, "beta": 1.0, "n": 1.0}, None, None, "erfexp_gamma.png"),
                    ("n", [0.5, 1.0, 1.5], {"mu": 5.0, "beta": 1.0, "gamma": 0.5}, None, None, "erfexp_n.png"),
                ],
            },
            {
                "name": "Novosibirsk",
                "class": zfit_physics.pdf.Novosibirsk,
                "params": [
                    ("mu", [4.0, 5.0, 6.0], {"sigma": 1.0, "lambd": 0.5}, None, None, "novo_mu.png"),
                    ("sigma", [0.5, 1.0, 1.5], {"mu": 5.0, "lambd": 0.5}, None, None, "novo_sigma.png"),
                    ("lambd", [0.2, 0.5, 0.8], {"mu": 5.0, "sigma": 1.0}, None, None, "novo_lambd.png"),
                ],
            },
            {
                "name": "Tsallis",
                "class": zfit_physics.pdf.Tsallis,
                "params": [
                    ("m", [0.5, 1.0, 1.5], {"n": 5.0, "t": 1.0}, None, None, "tsallis_m.png"),
                    ("n", [3.0, 5.0, 7.0], {"m": 1.0, "t": 1.0}, None, None, "tsallis_n.png"),
                    ("t", [0.5, 1.0, 1.5], {"m": 1.0, "n": 5.0}, None, None, "tsallis_t.png"),
                ],
            },
        ]

        # Plot each PDF with its variations
        for pdf_info in physics_pdfs:
            for param_name, values, fixed, _label, label_fn, filename in pdf_info["params"]:
                try:
                    # Create specific title for each plot
                    title = f"{pdf_info['name']} PDF with different {param_name} values"

                    # Create PDFConfig with custom filename
                    config = PDFConfig(
                        pdf_class=pdf_info["class"],
                        param_name=param_name,
                        param_values=values,
                        title=title,
                        filename=filename,  # Use the specific filename from the RST
                        x_range=(0, 10),
                        fixed_params=fixed,
                        label_fn=label_fn,
                    )
                    plot_pdf(config, obs)
                except Exception as e:
                    handle_error(f"{pdf_info['name']} {param_name}", e, filename)

    except ImportError:
        print("zfit_physics not installed, skipping physics PDFs")
        create_dummy_plot("Physics PDFs from zfit_physics not available", "physics_pdfs.png")


# ========================
# Main function
# ========================


def main():
    """Generate all PDF plots."""
    print("Generating PDF plots...")

    # Basic PDFs
    basic_pdfs = [
        plot_gaussian,
        plot_exponential,
        plot_uniform,
        plot_cauchy,
        plot_voigt,
        plot_crystalball,
        plot_lognormal,
        plot_chisquared,
        plot_studentt,
        plot_gamma,
        plot_bifurgauss,
        plot_poisson,
        plot_qgauss,
        plot_johnsonsu,
        plot_generalizedgauss,
        plot_truncatedgauss,
    ]

    with tqdm(total=len(basic_pdfs), desc="Generating basic PDF plots") as pbar:
        for func in basic_pdfs:
            try:
                func()
            except Exception as e:
                handle_error(func.__name__.replace("plot_", ""), e)
            pbar.update(1)

    # Polynomial PDFs
    polynomial_pdfs = [
        plot_bernstein,
        plot_chebyshev,
        plot_legendre,
        plot_chebyshev2,
        plot_hermite,
        plot_laguerre,
        plot_recursivepolynomial,
    ]

    with tqdm(total=len(polynomial_pdfs), desc="Generating polynomial PDF plots") as pbar:
        for func in polynomial_pdfs:
            try:
                func()
            except Exception as e:
                handle_error(func.__name__.replace("plot_", ""), e)
            pbar.update(1)

    # Physics PDFs
    physics_pdfs = [plot_doublecb, plot_gaussexptail, plot_generalizedcb, plot_generalizedgaussexptail]

    with tqdm(total=len(physics_pdfs), desc="Generating physics PDF plots") as pbar:
        for func in physics_pdfs:
            try:
                func()
            except Exception as e:
                handle_error(func.__name__.replace("plot_", ""), e)
            pbar.update(1)

    # Binned PDFs
    binned_pdfs = [
        plot_histogrampdf,
        plot_binwisescalemodifier,
        plot_binnedfromunbinnedpdf,
        plot_splinemorphingpdf,
        plot_binnedsumpdf,
        plot_splinepdf,
        plot_unbinnedfromibinnedpdf,
    ]

    with tqdm(total=len(binned_pdfs), desc="Generating binned PDF plots") as pbar:
        for func in binned_pdfs:
            try:
                func()
            except Exception as e:
                handle_error(func.__name__.replace("plot_", ""), e)
            pbar.update(1)

    # Composed PDFs
    composed_pdfs = [
        plot_sumpdf,
        plot_productpdf,
        plot_productpdf_1d,
        plot_fftconvpdf,
        plot_conditionalpdf,
        plot_truncatedpdf,
    ]

    with tqdm(total=len(composed_pdfs), desc="Generating composed PDF plots") as pbar:
        for func in composed_pdfs:
            try:
                func()
            except Exception as e:
                handle_error(func.__name__.replace("plot_", ""), e)
            pbar.update(1)

    # KDE plots
    print("Generating KDE PDF plots...")
    plot_kde()

    # Physics PDFs from zfit_physics
    print("Generating physics PDFs from zfit_physics...")
    plot_physics_pdfs()

    print("Done generating PDF plots.")


if __name__ == "__main__":
    main()
