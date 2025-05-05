#  Copyright (c) 2025 zfit
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import zfit
from zfit import Parameter

# Create the output directory if it doesn't exist
os.makedirs("../images/pdfs", exist_ok=True)

# Set the figure size and style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


def save_plot(filename):
    """Save the current plot to the specified filename."""
    plt.tight_layout()
    plt.savefig(f"images/pdfs/{filename}", dpi=100, bbox_inches="tight")
    plt.close()


# ========================
# Basic PDFs
# ========================


# Gaussian PDF
def plot_gaussian():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different mu values
    plt.figure()
    mu_values = [-2, 0, 2]
    sigma = Parameter("sigma", 1.0)

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gauss.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Gaussian PDF with different μ values")
    plt.legend()
    save_plot("gauss_mu.png")

    # Plot with different sigma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma_values = [0.5, 1.0, 2.0]

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gauss.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Gaussian PDF with different σ values")
    plt.legend()
    save_plot("gauss_sigma.png")


# Exponential PDF
def plot_exponential():
    # Create the observable
    obs = zfit.Space("x", limits=(0, 5))

    # Plot with different lambda values
    plt.figure()
    lambda_values = [0.5, 1.0, 2.0]

    for lambda_val in lambda_values:
        lambda_param = Parameter("lambda", lambda_val)
        exp = zfit.pdf.Exponential(lambda_param, obs=obs)
        x = np.linspace(0, 5, 1000)
        y = exp.pdf(x)
        plt.plot(x, y, label=f"λ = {lambda_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Exponential PDF with different λ values")
    plt.legend()
    save_plot("exponential_lambda.png")


# Uniform PDF
def plot_uniform():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different ranges
    plt.figure()
    ranges = [(-4, 4), (-2, 2), (0, 4)]

    for low, high in ranges:
        low_param = Parameter("low", low)
        high_param = Parameter("high", high)
        uniform = zfit.pdf.Uniform(low=low_param, high=high_param, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = uniform.pdf(x)
        plt.plot(x, y, label=f"Range: [{low}, {high}]")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Uniform PDF with different ranges")
    plt.legend()
    save_plot("uniform_range.png")


# Cauchy PDF
def plot_cauchy():
    # Create the observable
    obs = zfit.Space("x", limits=(-10, 10))

    # Plot with different m values
    plt.figure()
    m_values = [-2, 0, 2]
    gamma = Parameter("gamma", 1.0)

    for m_val in m_values:
        m = Parameter("m", m_val)
        cauchy = zfit.pdf.Cauchy(m=m, gamma=gamma, obs=obs)
        x = np.linspace(-10, 10, 1000)
        y = cauchy.pdf(x)
        plt.plot(x, y, label=f"m = {m_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Cauchy PDF with different m values")
    plt.legend()
    save_plot("cauchy_m.png")

    # Plot with different gamma values
    plt.figure()
    m = Parameter("m", 0.0)
    gamma_values = [0.5, 1.0, 2.0]

    for gamma_val in gamma_values:
        gamma = Parameter("gamma", gamma_val)
        cauchy = zfit.pdf.Cauchy(m=m, gamma=gamma, obs=obs)
        x = np.linspace(-10, 10, 1000)
        y = cauchy.pdf(x)
        plt.plot(x, y, label=f"γ = {gamma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Cauchy PDF with different γ values")
    plt.legend()
    save_plot("cauchy_gamma.png")


# Voigt PDF
def plot_voigt():
    # Create the observable
    obs = zfit.Space("x", limits=(-10, 10))

    # Plot with different sigma values
    plt.figure()
    m = Parameter("m", 0.0)
    gamma = Parameter("gamma", 1.0)
    sigma_values = [0.5, 1.0, 2.0]

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        voigt = zfit.pdf.Voigt(m=m, sigma=sigma, gamma=gamma, obs=obs)
        x = np.linspace(-10, 10, 1000)
        y = voigt.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Voigt PDF with different σ values")
    plt.legend()
    save_plot("voigt_sigma.png")

    # Plot with different gamma values
    plt.figure()
    m = Parameter("m", 0.0)
    sigma = Parameter("sigma", 1.0)
    gamma_values = [0.5, 1.0, 2.0]

    for gamma_val in gamma_values:
        gamma = Parameter("gamma", gamma_val)
        voigt = zfit.pdf.Voigt(m=m, sigma=sigma, gamma=gamma, obs=obs)
        x = np.linspace(-10, 10, 1000)
        y = voigt.pdf(x)
        plt.plot(x, y, label=f"γ = {gamma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Voigt PDF with different γ values")
    plt.legend()
    save_plot("voigt_gamma.png")


# CrystalBall PDF
def plot_crystalball():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different alpha values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    n = Parameter("n", 2.0)
    alpha_values = [0.5, 1.0, 2.0]

    for alpha_val in alpha_values:
        alpha = Parameter("alpha", alpha_val)
        cb = zfit.pdf.CrystalBall(mu=mu, sigma=sigma, alpha=alpha, n=n, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = cb.pdf(x)
        plt.plot(x, y, label=f"α = {alpha_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("CrystalBall PDF with different α values")
    plt.legend()
    save_plot("crystalball_alpha.png")

    # Plot with different n values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    alpha = Parameter("alpha", 1.0)
    n_values = [1.0, 2.0, 5.0]

    for n_val in n_values:
        n = Parameter("n", n_val)
        cb = zfit.pdf.CrystalBall(mu=mu, sigma=sigma, alpha=alpha, n=n, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = cb.pdf(x)
        plt.plot(x, y, label=f"n = {n_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("CrystalBall PDF with different n values")
    plt.legend()
    save_plot("crystalball_n.png")

    # Plot with different mu values
    plt.figure()
    sigma = Parameter("sigma", 1.0)
    alpha = Parameter("alpha", 1.0)
    n = Parameter("n", 2.0)
    mu_values = [-1.0, 0.0, 1.0]

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        cb = zfit.pdf.CrystalBall(mu=mu, sigma=sigma, alpha=alpha, n=n, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = cb.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("CrystalBall PDF with different μ values")
    plt.legend()
    save_plot("crystalball_mu.png")

    # Plot with different sigma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    alpha = Parameter("alpha", 1.0)
    n = Parameter("n", 2.0)
    sigma_values = [0.5, 1.0, 1.5]

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        cb = zfit.pdf.CrystalBall(mu=mu, sigma=sigma, alpha=alpha, n=n, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = cb.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("CrystalBall PDF with different σ values")
    plt.legend()
    save_plot("crystalball_sigma.png")


# LogNormal PDF
def plot_lognormal():
    # Create the observable
    obs = zfit.Space("x", limits=(0, 10))

    # Plot with different mu values
    plt.figure()
    mu_values = [-0.5, 0.0, 0.5]
    sigma = Parameter("sigma", 0.5)

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        lognormal = zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs)
        x = np.linspace(0.1, 10, 1000)
        y = lognormal.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("LogNormal PDF with different μ values")
    plt.legend()
    save_plot("lognormal_mu.png")

    # Plot with different sigma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma_values = [0.2, 0.5, 1.0]

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        lognormal = zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs)
        x = np.linspace(0.1, 10, 1000)
        y = lognormal.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("LogNormal PDF with different σ values")
    plt.legend()
    save_plot("lognormal_sigma.png")


# ChiSquared PDF
def plot_chisquared():
    # Create the observable
    obs = zfit.Space("x", limits=(0, 20))

    # Plot with different ndof values
    plt.figure()
    ndof_values = [1, 3, 5]

    for ndof_val in ndof_values:
        ndof = Parameter("ndof", ndof_val)
        chisq = zfit.pdf.ChiSquared(ndof=ndof, obs=obs)
        x = np.linspace(0.1, 20, 1000)
        y = chisq.pdf(x)
        plt.plot(x, y, label=f"ndof = {ndof_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("ChiSquared PDF with different ndof values")
    plt.legend()
    save_plot("chisquared_ndof.png")


# StudentT PDF
def plot_studentt():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different ndof values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    ndof_values = [1, 3, 10]

    for ndof_val in ndof_values:
        ndof = Parameter("ndof", ndof_val)
        studentt = zfit.pdf.StudentT(ndof=ndof, mu=mu, sigma=sigma, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = studentt.pdf(x)
        plt.plot(x, y, label=f"ndof = {ndof_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("StudentT PDF with different ndof values")
    plt.legend()
    save_plot("studentt_ndof.png")


# Gamma PDF
def plot_gamma():
    # Create the observable
    obs = zfit.Space("x", limits=(0, 10))

    # Plot with different gamma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    beta = Parameter("beta", 1.0)
    gamma_values = [1.0, 2.0, 5.0]

    for gamma_val in gamma_values:
        gamma = Parameter("gamma", gamma_val)
        gamma_pdf = zfit.pdf.Gamma(gamma=gamma, beta=beta, mu=mu, obs=obs)
        x = np.linspace(0.1, 10, 1000)
        y = gamma_pdf.pdf(x)
        plt.plot(x, y, label=f"γ = {gamma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Gamma PDF with different γ values")
    plt.legend()
    save_plot("gamma_gamma.png")

    # Plot with different beta values
    plt.figure()
    mu = Parameter("mu", 0.0)
    gamma = Parameter("gamma", 2.0)
    beta_values = [0.5, 1.0, 2.0]

    for beta_val in beta_values:
        beta = Parameter("beta", beta_val)
        gamma_pdf = zfit.pdf.Gamma(gamma=gamma, beta=beta, mu=mu, obs=obs)
        x = np.linspace(0.1, 10, 1000)
        y = gamma_pdf.pdf(x)
        plt.plot(x, y, label=f"β = {beta_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Gamma PDF with different β values")
    plt.legend()
    save_plot("gamma_beta.png")


# ========================
# Polynomial PDFs
# ========================


def plot_bernstein():
    # Create the observable
    obs = zfit.Space("x", limits=(0, 1))

    # Plot with different degrees
    plt.figure()
    degrees = [2, 3, 5]

    for degree in degrees:
        # Create coefficients for a simple shape
        coeffs = []
        for i in range(degree + 1):
            # Create a simple pattern: coefficients increase and then decrease
            val = 1.0 - abs(i - degree / 2) / (degree / 2)
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

    # Plot with different coefficient patterns for degree 3
    plt.figure()
    patterns = [
        [1.0, 0.2, 0.2, 1.0],  # U-shape
        [0.2, 1.0, 1.0, 0.2],  # Inverted U-shape
        [0.2, 0.5, 1.0, 0.2],  # Increasing then decreasing
    ]

    for i, pattern in enumerate(patterns):
        coeffs = [Parameter(f"c{j}", val) for j, val in enumerate(pattern)]
        bernstein = zfit.pdf.Bernstein(obs=obs, coeffs=coeffs)
        x = np.linspace(0, 1, 1000)
        y = bernstein.pdf(x)
        plt.plot(x, y, label=f"Pattern {i + 1}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Bernstein PDF (degree 3) with different coefficient patterns")
    plt.legend()
    save_plot("bernstein_patterns.png")


def plot_chebyshev():
    # Create the observable
    obs = zfit.Space("x", limits=(-1, 1))

    # Plot with different degrees
    plt.figure()
    degrees = [2, 3, 5]

    for degree in degrees:
        # Create coefficients
        coeffs = []
        for i in range(degree + 1):
            # First coefficient is 1, others are smaller
            val = 1.0 if i == 0 else 0.3
            coeffs.append(Parameter(f"c{i}", val))

        chebyshev = zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = chebyshev.pdf(x)
        plt.plot(x, y, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Chebyshev PDF with different degrees")
    plt.legend()
    save_plot("chebyshev_degree.png")

    # Plot with different coefficient patterns for degree 3
    plt.figure()
    patterns = [
        [1.0, 0.0, 0.0, 0.0],  # Constant
        [1.0, 0.5, 0.0, 0.0],  # Linear trend
        [1.0, 0.0, 0.5, 0.0],  # Quadratic
        [1.0, 0.0, 0.0, 0.5],  # Cubic
    ]

    for i, pattern in enumerate(patterns):
        coeffs = [Parameter(f"c{j}", val) for j, val in enumerate(pattern)]
        chebyshev = zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = chebyshev.pdf(x)
        plt.plot(x, y, label=f"Pattern {i + 1}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Chebyshev PDF (degree 3) with different coefficient patterns")
    plt.legend()
    save_plot("chebyshev_patterns.png")


def plot_legendre():
    # Create the observable
    obs = zfit.Space("x", limits=(-1, 1))

    # Plot with different degrees
    plt.figure()
    degrees = [2, 3, 5]

    for degree in degrees:
        # Create coefficients
        coeffs = []
        for i in range(degree + 1):
            # First coefficient is 1, others are smaller
            val = 1.0 if i == 0 else 0.3
            coeffs.append(Parameter(f"c{i}", val))

        legendre = zfit.pdf.Legendre(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = legendre.pdf(x)
        plt.plot(x, y, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Legendre PDF with different degrees")
    plt.legend()
    save_plot("legendre_degree.png")

    # Plot with different coefficient patterns for degree 3
    plt.figure()
    patterns = [
        [1.0, 0.0, 0.0, 0.0],  # Constant
        [1.0, 0.5, 0.0, 0.0],  # Linear trend
        [1.0, 0.0, 0.5, 0.0],  # Quadratic
        [1.0, 0.0, 0.0, 0.5],  # Cubic
    ]

    for i, pattern in enumerate(patterns):
        coeffs = [Parameter(f"c{j}", val) for j, val in enumerate(pattern)]
        legendre = zfit.pdf.Legendre(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = legendre.pdf(x)
        plt.plot(x, y, label=f"Pattern {i + 1}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Legendre PDF (degree 3) with different coefficient patterns")
    plt.legend()
    save_plot("legendre_patterns.png")


# ========================
# Physics PDFs
# ========================


def plot_crystalball_physics():
    # This is already covered in the basic PDFs section
    pass


def plot_doublecb():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different alpha values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    nl = Parameter("nl", 2.0)
    nr = Parameter("nr", 2.0)
    alphar = Parameter("alphar", 1.0)

    alphal_values = [0.5, 1.0, 2.0]

    for alphal_val in alphal_values:
        alphal = Parameter("alphal", alphal_val)
        doublecb = zfit.pdf.DoubleCB(mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = doublecb.pdf(x)
        plt.plot(x, y, label=f"αL = {alphal_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("DoubleCB PDF with different αL values")
    plt.legend()
    save_plot("doublecb_alphal.png")

    # Plot with different alphar values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    nl = Parameter("nl", 2.0)
    nr = Parameter("nr", 2.0)
    alphal = Parameter("alphal", 1.0)

    alphar_values = [0.5, 1.0, 2.0]

    for alphar_val in alphar_values:
        alphar = Parameter("alphar", alphar_val)
        doublecb = zfit.pdf.DoubleCB(mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = doublecb.pdf(x)
        plt.plot(x, y, label=f"αR = {alphar_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("DoubleCB PDF with different αR values")
    plt.legend()
    save_plot("doublecb_alphar.png")


def plot_gaussexptail():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different alpha values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)

    alpha_values = [0.5, 1.0, 2.0]

    for alpha_val in alpha_values:
        alpha = Parameter("alpha", alpha_val)
        gaussexptail = zfit.pdf.GaussExpTail(mu=mu, sigma=sigma, alpha=alpha, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gaussexptail.pdf(x)
        plt.plot(x, y, label=f"α = {alpha_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GaussExpTail PDF with different α values")
    plt.legend()
    save_plot("gaussexptail_alpha.png")

    # Plot with different sigma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    alpha = Parameter("alpha", 1.0)

    sigma_values = [0.5, 1.0, 2.0]

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        gaussexptail = zfit.pdf.GaussExpTail(mu=mu, sigma=sigma, alpha=alpha, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gaussexptail.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GaussExpTail PDF with different σ values")
    plt.legend()
    save_plot("gaussexptail_sigma.png")


# ========================
# Main function
# ========================


def main():
    """Generate all PDF plots."""
    print("Generating PDF plots...")

    # Basic PDFs
    plot_gaussian()
    plot_exponential()
    plot_uniform()
    plot_cauchy()
    plot_voigt()
    plot_crystalball()
    plot_lognormal()
    plot_chisquared()
    plot_studentt()
    plot_gamma()

    # Polynomial PDFs
    plot_bernstein()
    plot_chebyshev()
    plot_legendre()

    # Physics PDFs
    plot_doublecb()
    plot_gaussexptail()

    print("Done generating PDF plots.")


if __name__ == "__main__":
    main()
