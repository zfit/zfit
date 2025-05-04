"""Bayesian inference for zfit.

This module provides tools for Bayesian inference in zfit, including priors,
MCMC _mcmc, and methods for analyzing posterior distributions.
"""

#  Copyright (c) 2025 zfit

# Import results
from .results import BayesianResult

# Import priors
from .priors import (
    NormalPrior,
    UniformPrior,
    HalfNormalPrior,
    GammaPrior,
    BetaPrior,
)

# Import _mcmc
from zfit._mcmc import (
    EmceeSampler,
)

# Define public API
__all__ = [
    # Results
    "BayesianResult",
    "Posteriors",
    # Priors
    "NormalPrior",
    "UniformPrior",
    "HalfNormalPrior",
    "GammaPrior",
    "BetaPrior",
    # Samplers
    "EmceeSampler",
]
