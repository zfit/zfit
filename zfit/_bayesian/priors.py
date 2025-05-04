"""Prior distributions for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

from ..core.interfaces import ZfitPrior


class BasePrior(ZfitPrior):
    pass


class NormalPrior(BasePrior):
    """Normal (Gaussian) prior distribution."""

    def __init__(self, mu, sigma, name=None):
        """Initialize a Normal prior.

        Args:
            mu: Mean of the normal distribution
            sigma: Standard deviation of the normal distribution
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))
        pdf = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
        super().__init__(pdf=pdf, name=name)


class UniformPrior(BasePrior):
    """Uniform prior distribution."""

    def __init__(self, lower, upper, name=None):
        """Initialize a Uniform prior.

        Args:
            lower: Lower bound of the uniform distribution
            upper: Upper bound of the uniform distribution
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(lower, upper))
        pdf = zfit.pdf.Uniform(low=lower, high=upper, obs=obs)
        super().__init__(pdf=pdf, name=name)


class HalfNormalPrior(BasePrior):
    """Half-normal prior distribution for scale parameters."""

    def __init__(self, mu, sigma, name=None):
        """Initialize a Half-Normal prior.

        Args:
            sigma: Scale parameter of the half-normal distribution
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(0, float("inf")))
        pdf = zfit.pdf.TruncatedGauss(mu=mu, sigma=sigma, low=0, high=float("inf"), obs=obs)
        super().__init__(pdf=pdf, name=name)


class GammaPrior(BasePrior):
    """Gamma prior distribution for positive parameters."""

    def __init__(self, alpha, beta, name=None):
        """Initialize a Gamma prior.

        Args:
            alpha: Shape parameter (α > 0)
            beta: Rate parameter (β > 0)
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(0, float("inf")))
        pdf = zfit.pdf.Gamma(gamma=alpha, beta=beta, mu=0, obs=obs)
        super().__init__(pdf=pdf, name=name)


class BetaPrior(BasePrior):
    """Beta prior distribution for parameters in the range [0, 1]."""

    def __init__(self, alpha, beta, name=None):
        """Initialize a Beta prior.

        Args:
            alpha: First shape parameter (α > 0)
            beta: Second shape parameter (β > 0)
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(0, 1))
        pdf = zfit.pdf.Beta(alpha=alpha, beta=beta, obs=obs)
        super().__init__(pdf=pdf, name=name)
