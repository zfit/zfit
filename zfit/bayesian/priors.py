"""Prior distributions for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitObject


class ZfitPrior(ZfitObject):
    """Base class for parameter priors in Bayesian inference."""

    def __init__(self, name=None):
        """Initialize a prior distribution.

        Args:
            name: Optional name for the prior
        """
        self.name = name

    def log_pdf(self, value):
        """Return the log probability of the prior at the given value(s).

        Args:
            value: The parameter value(s) to evaluate the log probability at

        Returns:
            The log probability
        """
        raise NotImplementedError

    def sample(self, n):
        """Sample n values from the prior distribution.

        Args:
            n: Number of samples to draw

        Returns:
            An array of samples
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class NormalPrior(ZfitPrior):
    """Normal (Gaussian) prior distribution.

    p(x|μ,σ) ∝ exp(-(x-μ)²/(2σ²))
    """

    def __init__(self, mu, sigma, name=None):
        """Initialize a Normal prior.

        Args:
            mu: Mean of the normal distribution
            sigma: Standard deviation of the normal distribution
            name: Name of the prior
        """
        super().__init__(name=name)
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self, value):
        """Return the log probability of the normal prior at the given value.

        Args:
            value: The parameter value(s) to evaluate the log probability at

        Returns:
            The log probability
        """
        return -0.5 * znp.square((value - self.mu) / self.sigma) - znp.log(self.sigma * znp.sqrt(2 * np.pi))

    def sample(self, n):
        """Sample n values from the normal prior distribution.

        Args:
            n: Number of samples to draw

        Returns:
            An array of samples
        """
        return self.mu + self.sigma * znp.random.normal(size=(n,))

    def __repr__(self):
        return f"NormalPrior(mu={self.mu}, sigma={self.sigma}, name='{self.name}')"


class UniformPrior(ZfitPrior):
    """Uniform prior distribution.

    p(x|a,b) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
    """

    def __init__(self, lower, upper, name=None):
        """Initialize a Uniform prior.

        Args:
            lower: Lower bound of the uniform distribution
            upper: Upper bound of the uniform distribution
            name: Name of the prior
        """
        super().__init__(name=name)
        self.lower = lower
        self.upper = upper

    def log_pdf(self, value):
        """Return the log probability of the uniform prior at the given value.

        Args:
            value: The parameter value(s) to evaluate the log probability at

        Returns:
            The log probability
        """
        in_range = znp.logical_and(value >= self.lower, value <= self.upper)
        return znp.where(in_range, -znp.log(self.upper - self.lower), -float("inf"))

    def sample(self, n):
        """Sample n values from the uniform prior distribution.

        Args:
            n: Number of samples to draw

        Returns:
            An array of samples
        """
        return self.lower + (self.upper - self.lower) * znp.random.uniform(size=(n,))

    def __repr__(self):
        return f"UniformPrior(lower={self.lower}, upper={self.upper}, name='{self.name}')"


class HalfNormalPrior(ZfitPrior):
    """Half-normal prior distribution for scale parameters (σ > 0).

    p(x|σ) ∝ exp(-x²/(2σ²)) for x > 0
    """

    def __init__(self, sigma, name=None):
        """Initialize a Half-Normal prior.

        Args:
            sigma: Scale parameter of the half-normal distribution
            name: Name of the prior
        """
        super().__init__(name=name)
        self.sigma = sigma

    def log_pdf(self, value):
        """Return the log probability of the half-normal prior at the given value.

        Args:
            value: The parameter value(s) to evaluate the log probability at

        Returns:
            The log probability
        """
        in_range = value > 0
        log_prob_valid = (
            -0.5 * znp.square(value / self.sigma) - znp.log(self.sigma * znp.sqrt(2 * np.pi)) + znp.log(2.0)
        )
        return znp.where(in_range, log_prob_valid, -float("inf"))

    def sample(self, n):
        """Sample n values from the half-normal prior distribution.

        Args:
            n: Number of samples to draw

        Returns:
            An array of samples
        """
        return znp.abs(self.sigma * znp.random.normal(size=(n,)))

    def __repr__(self):
        return f"HalfNormalPrior(sigma={self.sigma}, name='{self.name}')"


class GammaPrior(ZfitPrior):
    """Gamma prior distribution for positive parameters.

    p(x|α,β) ∝ x^(α-1) * exp(-βx) for x > 0
    """

    def __init__(self, alpha, beta, name=None):
        """Initialize a Gamma prior.

        Args:
            alpha: Shape parameter (α > 0)
            beta: Rate parameter (β > 0)
            name: Name of the prior
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def log_pdf(self, value):
        """Return the log probability of the gamma prior at the given value.

        Args:
            value: The parameter value(s) to evaluate the log probability at

        Returns:
            The log probability
        """
        in_range = value > 0
        log_prob_valid = (
            (self.alpha - 1) * znp.log(value)
            - self.beta * value
            - self.alpha * znp.log(self.beta)
            - tf.math.lgamma(self.alpha)
        )
        return znp.where(in_range, log_prob_valid, -float("inf"))

    def sample(self, n):
        """Sample n values from the gamma prior distribution.

        Args:
            n: Number of samples to draw

        Returns:
            An array of samples
        """
        # Use numpy for gamma sampling
        return np.random.gamma(self.alpha, 1 / self.beta, size=(n,))

    def __repr__(self):
        return f"GammaPrior(alpha={self.alpha}, beta={self.beta}, name='{self.name}')"


class BetaPrior(ZfitPrior):
    """Beta prior distribution for parameters in the range [0, 1].

    p(x|α,β) ∝ x^(α-1) * (1-x)^(β-1) for 0 ≤ x ≤ 1
    """

    def __init__(self, alpha, beta, name=None):
        """Initialize a Beta prior.

        Args:
            alpha: First shape parameter (α > 0)
            beta: Second shape parameter (β > 0)
            name: Name of the prior
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def log_pdf(self, value):
        """Return the log probability of the beta prior at the given value.

        Args:
            value: The parameter value(s) to evaluate the log probability at

        Returns:
            The log probability
        """
        in_range = znp.logical_and(value >= 0, value <= 1)
        log_prob_valid = (
            (self.alpha - 1) * znp.log(value)
            + (self.beta - 1) * znp.log(1 - value)
            - tf.math.lgamma(self.alpha)
            - tf.math.lgamma(self.beta)
            + tf.math.lgamma(self.alpha + self.beta)
        )
        return znp.where(in_range, log_prob_valid, -float("inf"))

    def sample(self, n):
        """Sample n values from the beta prior distribution.

        Args:
            n: Number of samples to draw

        Returns:
            An array of samples
        """
        # Use numpy for beta sampling
        return np.random.beta(self.alpha, self.beta, size=(n,))

    def __repr__(self):
        return f"BetaPrior(alpha={self.alpha}, beta={self.beta}, name='{self.name}')"


def add_prior_to_parameter(param, prior):
    """Add a prior to a zfit Parameter.

    Args:
        param: A zfit Parameter
        prior: A ZfitPrior instance

    Returns:
        The parameter with the prior attached
    """
    if not isinstance(param, zfit.Parameter):
        msg = f"param must be a zfit.Parameter, got {type(param)}"
        raise TypeError(msg)
    if not isinstance(prior, ZfitPrior):
        msg = f"prior must be a ZfitPrior, got {type(prior)}"
        raise TypeError(msg)

    # Attach the prior as an attribute
    param._prior = prior

    # Add method to get the prior if it doesn't exist
    if not hasattr(param, "get_prior"):
        param.get_prior = lambda: getattr(param, "_prior", None)

    # Add method to calculate log prior if it doesn't exist
    if not hasattr(param, "log_prior"):

        def log_prior():
            p = getattr(param, "_prior", None)
            if p is None:
                return 0.0  # Improper flat prior
            return p.log_prob(param.value())

        param.log_prior = log_prior

    return param


def set_priors(param_prior_mapping):
    """Set priors for multiple parameters.

    Args:
        param_prior_mapping: A dictionary mapping parameters to priors

    Returns:
        List of parameters with priors attached
    """
    return [add_prior_to_parameter(param, prior) for param, prior in param_prior_mapping.items()]
