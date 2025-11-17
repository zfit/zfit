"""Bayesian inference module for zfit.

- **Priors**: Essential prior distributions (Normal, Uniform, Gamma, etc.)
- **PosteriorSamples**: Clean interface to MCMC results with ArviZ integration
- **Samplers**: MCMC samplers (EmceeSampler)


Example:
    >>> import zfit
    >>> # Create model with priors using modern names
    >>> mu = zfit.Parameter("mu", 0, prior=zfit.prior.Normal(0, 1))
    >>> sigma = zfit.Parameter("sigma", 1, prior=zfit.prior.HalfNormal(1))
    >>> model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    >>>
    >>> # Sample posterior
    >>> sampler = zfit.mcmc.EmceeSampler(nwalkers=20)
    >>> posterior = sampler.sample(loss, n_samples=1000)
    >>>
    >>> # Analyze with ArviZ
    >>> idata = posterior.to_arviz()
    >>> az.plot_trace(idata)
"""

#  Copyright (c) 2025 zfit

from .posterior import PosteriorSamples
from .priors import Normal, Uniform, HalfNormal, Gamma, Beta, LogNormal, Cauchy, KDE, Exponential, Poisson, StudentT
from .mathconstrain import (
    IdentityTransform,
    LogTransform,
    SigmoidTransform,
    AffineTransform,
    LowerBoundTransform,
    UpperBoundTransform,
    ConstraintType,
    PriorConstraint,
    UNCONSTRAINED,
    POSITIVE,
    UNIT_INTERVAL,
)

__all__ = [
    "PosteriorSamples",
    "Normal", "Uniform", "HalfNormal", "Gamma", "Beta", "LogNormal", "Cauchy", "KDE", "Exponential", "Poisson", "StudentT",
    # Transformations
    "IdentityTransform",
    "LogTransform",
    "SigmoidTransform",
    "AffineTransform",
    "LowerBoundTransform",
    "UpperBoundTransform",
    # Constraint system
    "ConstraintType",
    "PriorConstraint",
    "UNCONSTRAINED",
    "POSITIVE",
    "UNIT_INTERVAL",
]
