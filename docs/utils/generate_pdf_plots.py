#  Copyright (c) 2025 zfit
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import zfit
from zfit import Parameter

# Create the output directory if it doesn't exist
outpath = Path("../images/_generated/pdfs")
outpath.mkdir(parents=True, exist_ok=True)

# Set the figure size and style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


def save_plot(filename):
    """Save the current plot to the specified filename."""
    plt.tight_layout()
    plt.savefig(outpath / f"{filename}", dpi=100, bbox_inches="tight")
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


# BifurGauss PDF
def plot_bifurgauss():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different mu values
    plt.figure()
    mu_values = [-1.0, 0.0, 1.0]
    sigmal = Parameter("sigmal", 1.0)
    sigmar = Parameter("sigmar", 1.0)

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        bifurgauss = zfit.pdf.BifurGauss(mu=mu, sigmal=sigmal, sigmar=sigmar, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = bifurgauss.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("BifurGauss PDF with different μ values")
    plt.legend()
    save_plot("bifurgauss_mu.png")

    # Plot with different sigmal values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigmal_values = [0.5, 1.0, 1.5]
    sigmar = Parameter("sigmar", 1.0)

    for sigmal_val in sigmal_values:
        sigmal = Parameter("sigmal", sigmal_val)
        bifurgauss = zfit.pdf.BifurGauss(mu=mu, sigmal=sigmal, sigmar=sigmar, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = bifurgauss.pdf(x)
        plt.plot(x, y, label=f"σ_left = {sigmal_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("BifurGauss PDF with different σ_left values")
    plt.legend()
    save_plot("bifurgauss_sigmal.png")

    # Plot with different sigmar values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigmal = Parameter("sigmal", 1.0)
    sigmar_values = [0.5, 1.0, 1.5]

    for sigmar_val in sigmar_values:
        sigmar = Parameter("sigmar", sigmar_val)
        bifurgauss = zfit.pdf.BifurGauss(mu=mu, sigmal=sigmal, sigmar=sigmar, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = bifurgauss.pdf(x)
        plt.plot(x, y, label=f"σ_right = {sigmar_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("BifurGauss PDF with different σ_right values")
    plt.legend()
    save_plot("bifurgauss_sigmar.png")


# Poisson PDF
def plot_poisson():
    # Create the observable
    obs = zfit.Space("x", limits=(0, 20))

    # Plot with different lambda values
    plt.figure()
    lambda_values = [1.0, 5.0, 10.0]

    for lambda_val in lambda_values:
        lambda_param = Parameter("lambda", lambda_val)
        poisson = zfit.pdf.Poisson(lam=lambda_param, obs=obs)
        x = np.arange(0, 20)
        y = poisson.pdf(x)
        plt.step(x, y, where="mid", label=f"λ = {lambda_val}")

    plt.xlabel("x")
    plt.ylabel("Probability mass")
    plt.title("Poisson PDF with different λ values")
    plt.legend()
    save_plot("poisson_lambda.png")


# QGauss PDF
def plot_qgauss():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different mu values
    plt.figure()
    mu_values = [-1.0, 0.0, 1.0]
    sigma = Parameter("sigma", 1.0)
    q = Parameter("q", 0.7)

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        qgauss = zfit.pdf.QGauss(mu=mu, sigma=sigma, q=q, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = qgauss.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("QGauss PDF with different μ values")
    plt.legend()
    save_plot("qgauss_mu.png")

    # Plot with different sigma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma_values = [0.5, 1.0, 1.5]
    q = Parameter("q", 0.7)

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        qgauss = zfit.pdf.QGauss(mu=mu, sigma=sigma, q=q, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = qgauss.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("QGauss PDF with different σ values")
    plt.legend()
    save_plot("qgauss_sigma.png")

    # Plot with different q values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    q_values = [0.5, 0.7, 0.9]

    for q_val in q_values:
        q = Parameter("q", q_val)
        qgauss = zfit.pdf.QGauss(mu=mu, sigma=sigma, q=q, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = qgauss.pdf(x)
        plt.plot(x, y, label=f"q = {q_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("QGauss PDF with different q values")
    plt.legend()
    save_plot("qgauss_q.png")


# JohnsonSU PDF
def plot_johnsonsu():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different mu values
    plt.figure()
    mu_values = [-1.0, 0.0, 1.0]
    sigma = Parameter("sigma", 1.0)
    gamma = Parameter("gamma", 1.0)
    delta = Parameter("delta", 1.0)

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        johnsonsu = zfit.pdf.JohnsonSU(mu=mu, sigma=sigma, gamma=gamma, delta=delta, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = johnsonsu.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("JohnsonSU PDF with different μ values")
    plt.legend()
    save_plot("johnsonsu_mu.png")

    # Plot with different gamma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    gamma_values = [0.0, 1.0, 2.0]
    delta = Parameter("delta", 1.0)

    for gamma_val in gamma_values:
        gamma = Parameter("gamma", gamma_val)
        johnsonsu = zfit.pdf.JohnsonSU(mu=mu, sigma=sigma, gamma=gamma, delta=delta, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = johnsonsu.pdf(x)
        plt.plot(x, y, label=f"γ = {gamma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("JohnsonSU PDF with different γ values")
    plt.legend()
    save_plot("johnsonsu_gamma.png")

    # Plot with different delta values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    gamma = Parameter("gamma", 1.0)
    delta_values = [0.5, 1.0, 2.0]

    for delta_val in delta_values:
        delta = Parameter("delta", delta_val)
        johnsonsu = zfit.pdf.JohnsonSU(mu=mu, sigma=sigma, gamma=gamma, delta=delta, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = johnsonsu.pdf(x)
        plt.plot(x, y, label=f"δ = {delta_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("JohnsonSU PDF with different δ values")
    plt.legend()
    save_plot("johnsonsu_delta.png")


# GeneralizedGauss PDF
def plot_generalizedgauss():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different mu values
    plt.figure()
    mu_values = [-1.0, 0.0, 1.0]
    alpha = Parameter("alpha", 1.0)
    p = Parameter("p", 2.0)

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        gengauss = zfit.pdf.GeneralizedGauss(mu=mu, alpha=alpha, p=p, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gengauss.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedGauss PDF with different μ values")
    plt.legend()
    save_plot("generalizedgauss_mu.png")

    # Plot with different alpha values
    plt.figure()
    mu = Parameter("mu", 0.0)
    alpha_values = [0.5, 1.0, 2.0]
    p = Parameter("p", 2.0)

    for alpha_val in alpha_values:
        alpha = Parameter("alpha", alpha_val)
        gengauss = zfit.pdf.GeneralizedGauss(mu=mu, alpha=alpha, p=p, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gengauss.pdf(x)
        plt.plot(x, y, label=f"α = {alpha_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedGauss PDF with different α values")
    plt.legend()
    save_plot("generalizedgauss_alpha.png")

    # Plot with different p values
    plt.figure()
    mu = Parameter("mu", 0.0)
    alpha = Parameter("alpha", 1.0)
    p_values = [1.0, 2.0, 4.0]

    for p_val in p_values:
        p = Parameter("p", p_val)
        gengauss = zfit.pdf.GeneralizedGauss(mu=mu, alpha=alpha, p=p, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gengauss.pdf(x)
        plt.plot(x, y, label=f"p = {p_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedGauss PDF with different p values")
    plt.legend()
    save_plot("generalizedgauss_p.png")


# TruncatedGauss PDF
def plot_truncatedgauss():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different mu values
    plt.figure()
    mu_values = [-1.0, 0.0, 1.0]
    sigma = Parameter("sigma", 1.0)
    low = Parameter("low", -2.0)
    high = Parameter("high", 2.0)

    for mu_val in mu_values:
        mu = Parameter("mu", mu_val)
        truncgauss = zfit.pdf.TruncatedGauss(mu=mu, sigma=sigma, low=low, high=high, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = truncgauss.pdf(x)
        plt.plot(x, y, label=f"μ = {mu_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("TruncatedGauss PDF with different μ values")
    plt.legend()
    save_plot("truncatedgauss_mu.png")

    # Plot with different sigma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma_values = [0.5, 1.0, 1.5]
    low = Parameter("low", -2.0)
    high = Parameter("high", 2.0)

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        truncgauss = zfit.pdf.TruncatedGauss(mu=mu, sigma=sigma, low=low, high=high, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = truncgauss.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("TruncatedGauss PDF with different σ values")
    plt.legend()
    save_plot("truncatedgauss_sigma.png")

    # Plot with different truncation ranges
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    ranges = [(-1.0, 1.0), (-2.0, 2.0), (-0.5, 2.0)]

    for i, (low_val, high_val) in enumerate(ranges):
        low = Parameter("low", low_val)
        high = Parameter("high", high_val)
        truncgauss = zfit.pdf.TruncatedGauss(mu=mu, sigma=sigma, low=low, high=high, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = truncgauss.pdf(x)
        plt.plot(x, y, label=f"Range: [{low_val}, {high_val}]")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("TruncatedGauss PDF with different truncation ranges")
    plt.legend()
    save_plot("truncatedgauss_range.png")


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


def plot_chebyshev2():
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

        chebyshev2 = zfit.pdf.Chebyshev2(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = chebyshev2.pdf(x)
        plt.plot(x, y, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Chebyshev2 PDF with different degrees")
    plt.legend()
    save_plot("chebyshev2_degree.png")

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
        chebyshev2 = zfit.pdf.Chebyshev2(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = chebyshev2.pdf(x)
        plt.plot(x, y, label=f"Pattern {i + 1}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Chebyshev2 PDF (degree 3) with different coefficient patterns")
    plt.legend()
    save_plot("chebyshev2_patterns.png")


def plot_hermite():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

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

        hermite = zfit.pdf.Hermite(obs=obs, coeffs=coeffs)
        x = np.linspace(-5, 5, 1000)
        y = hermite.pdf(x)
        plt.plot(x, y, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Hermite PDF with different degrees")
    plt.legend()
    save_plot("hermite_degree.png")

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
        hermite = zfit.pdf.Hermite(obs=obs, coeffs=coeffs)
        x = np.linspace(-5, 5, 1000)
        y = hermite.pdf(x)
        plt.plot(x, y, label=f"Pattern {i + 1}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Hermite PDF (degree 3) with different coefficient patterns")
    plt.legend()
    save_plot("hermite_patterns.png")


def plot_laguerre():
    # Create the observable
    obs = zfit.Space("x", limits=(0, 10))

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

        laguerre = zfit.pdf.Laguerre(obs=obs, coeffs=coeffs)
        x = np.linspace(0, 10, 1000)
        y = laguerre.pdf(x)
        plt.plot(x, y, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Laguerre PDF with different degrees")
    plt.legend()
    save_plot("laguerre_degree.png")

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
        laguerre = zfit.pdf.Laguerre(obs=obs, coeffs=coeffs)
        x = np.linspace(0, 10, 1000)
        y = laguerre.pdf(x)
        plt.plot(x, y, label=f"Pattern {i + 1}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Laguerre PDF (degree 3) with different coefficient patterns")
    plt.legend()
    save_plot("laguerre_patterns.png")


def plot_recursivepolynomial():
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

        recpoly = zfit.pdf.RecursivePolynomial(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = recpoly.pdf(x)
        plt.plot(x, y, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("RecursivePolynomial PDF with different degrees")
    plt.legend()
    save_plot("recursivepolynomial_degree.png")

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
        recpoly = zfit.pdf.RecursivePolynomial(obs=obs, coeffs=coeffs)
        x = np.linspace(-1, 1, 1000)
        y = recpoly.pdf(x)
        plt.plot(x, y, label=f"Pattern {i + 1}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("RecursivePolynomial PDF (degree 3) with different coefficient patterns")
    plt.legend()
    save_plot("recursivepolynomial_patterns.png")


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


def plot_generalizedcb():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different alpha values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    n = Parameter("n", 2.0)
    t = Parameter("t", 3.0)

    alpha_values = [0.5, 1.0, 2.0]

    for alpha_val in alpha_values:
        alpha = Parameter("alpha", alpha_val)
        gencb = zfit.pdf.GeneralizedCB(mu=mu, sigma=sigma, alpha=alpha, n=n, t=t, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gencb.pdf(x)
        plt.plot(x, y, label=f"α = {alpha_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedCB PDF with different α values")
    plt.legend()
    save_plot("generalizedcb_alpha.png")

    # Plot with different n values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    alpha = Parameter("alpha", 1.0)
    t = Parameter("t", 3.0)

    n_values = [1.0, 2.0, 5.0]

    for n_val in n_values:
        n = Parameter("n", n_val)
        gencb = zfit.pdf.GeneralizedCB(mu=mu, sigma=sigma, alpha=alpha, n=n, t=t, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gencb.pdf(x)
        plt.plot(x, y, label=f"n = {n_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedCB PDF with different n values")
    plt.legend()
    save_plot("generalizedcb_n.png")

    # Plot with different t values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    alpha = Parameter("alpha", 1.0)
    n = Parameter("n", 2.0)

    t_values = [1.0, 3.0, 5.0]

    for t_val in t_values:
        t = Parameter("t", t_val)
        gencb = zfit.pdf.GeneralizedCB(mu=mu, sigma=sigma, alpha=alpha, n=n, t=t, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gencb.pdf(x)
        plt.plot(x, y, label=f"t = {t_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedCB PDF with different t values")
    plt.legend()
    save_plot("generalizedcb_t.png")


def plot_generalizedgaussexptail():
    # Create the observable
    obs = zfit.Space("x", limits=(-5, 5))

    # Plot with different alpha values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    k = Parameter("k", 1.0)

    alpha_values = [0.5, 1.0, 2.0]

    for alpha_val in alpha_values:
        alpha = Parameter("alpha", alpha_val)
        gengaussexptail = zfit.pdf.GeneralizedGaussExpTail(mu=mu, sigma=sigma, alpha=alpha, k=k, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gengaussexptail.pdf(x)
        plt.plot(x, y, label=f"α = {alpha_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedGaussExpTail PDF with different α values")
    plt.legend()
    save_plot("generalizedgaussexptail_alpha.png")

    # Plot with different k values
    plt.figure()
    mu = Parameter("mu", 0.0)
    sigma = Parameter("sigma", 1.0)
    alpha = Parameter("alpha", 1.0)

    k_values = [0.5, 1.0, 2.0]

    for k_val in k_values:
        k = Parameter("k", k_val)
        gengaussexptail = zfit.pdf.GeneralizedGaussExpTail(mu=mu, sigma=sigma, alpha=alpha, k=k, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gengaussexptail.pdf(x)
        plt.plot(x, y, label=f"k = {k_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedGaussExpTail PDF with different k values")
    plt.legend()
    save_plot("generalizedgaussexptail_k.png")

    # Plot with different sigma values
    plt.figure()
    mu = Parameter("mu", 0.0)
    alpha = Parameter("alpha", 1.0)
    k = Parameter("k", 1.0)

    sigma_values = [0.5, 1.0, 2.0]

    for sigma_val in sigma_values:
        sigma = Parameter("sigma", sigma_val)
        gengaussexptail = zfit.pdf.GeneralizedGaussExpTail(mu=mu, sigma=sigma, alpha=alpha, k=k, obs=obs)
        x = np.linspace(-5, 5, 1000)
        y = gengaussexptail.pdf(x)
        plt.plot(x, y, label=f"σ = {sigma_val}")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("GeneralizedGaussExpTail PDF with different σ values")
    plt.legend()
    save_plot("generalizedgaussexptail_sigma.png")


# ========================
# Main function
# ========================


def main():
    """Generate all PDF plots."""
    print("Generating PDF plots...")

    # Basic PDFs
    allpdfs = [
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
        # Polynomial PDFs
        plot_bernstein,
        plot_chebyshev,
        plot_legendre,
        plot_chebyshev2,
        plot_hermite,
        plot_laguerre,
        plot_recursivepolynomial,
        # Physics PDFs
        plot_doublecb,
        plot_gaussexptail,
        plot_generalizedcb,
        plot_generalizedgaussexptail,
    ]
    for pdfplot in tqdm(allpdfs, desc="Generating PDF plots"):
        pdfplot()
    print("Done generating PDF plots.")


if __name__ == "__main__":
    main()
