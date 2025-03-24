"""Bayesian inference for zfit.

This module provides tools for Bayesian inference in zfit, including priors,
MCMC samplers, and methods for analyzing posterior distributions.
"""

#  Copyright (c) 2025 zfit

# Import results
from .results import Posteriors

# Import priors
from .priors import (
    ZfitPrior,
    NormalPrior,
    UniformPrior,
    HalfNormalPrior,
    GammaPrior,
    BetaPrior,
    add_prior_to_parameter,
    set_priors,
)

# Import samplers
from .samplers import (
    EmceeSampler,
)

# Define public API
__all__ = [
    # Results
    "Posteriors",
    # Priors
    "ZfitPrior",
    "NormalPrior",
    "UniformPrior",
    "HalfNormalPrior",
    "GammaPrior",
    "BetaPrior",
    "add_prior_to_parameter",
    "set_priors",
    # Samplers
    "EmceeSampler",
]
