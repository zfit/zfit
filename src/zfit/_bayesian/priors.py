"""Prior distributions for Bayesian inference in zfit.

This module provides a collection of commonly used prior distributions that can be
attached to parameters in Bayesian inference workflows. All priors automatically
adapt to parameter bounds when specified, ensuring proper normalization within
the valid parameter range.
"""

#  Copyright (c) 2025 zfit

from __future__ import annotations

from abc import ABC, abstractmethod

import zfit
import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitPrior


class AdaptivePrior(ZfitPrior, ABC):
    """Base class for priors that automatically adapt to parameter limits.

    This abstract base class provides the foundation for all prior distributions
    that can adjust their support to match parameter constraints. When a prior
    is assigned to a parameter with bounds, it automatically truncates or adjusts
    its distribution to respect those limits while maintaining proper normalization.

    Subclasses must implement the `_create_pdf` method to define the specific
    probability distribution. The adaptation logic handles bounds checking and
    PDF recreation when necessary.

    Attributes:
        _default_bounds: Default bounds for the distribution (can be overridden)
        _requires_positive: Whether the distribution requires positive support
    """

    # Prior type configuration - subclasses should override these
    _default_bounds: tuple[float, float] = (-float("inf"), float("inf"))
    _requires_positive: bool = False

    def __init__(self, pdf_params: dict, bounds: tuple[float, float] | None = None, name=None):
        """Initialize an adaptive prior.

        Args:
            pdf_params: Parameters to pass to the PDF constructor
            bounds: Optional custom bounds for the prior
            name: Optional name for the prior
        """
        # Store parameters for potential adaptation
        self._pdf_params = pdf_params.copy()
        self._original_bounds = bounds or self._default_bounds

        # Create initial PDF
        obs = zfit.Space("prior_space", limits=self._original_bounds)
        pdf = self._create_pdf(obs, **self._pdf_params)
        super().__init__(pdf=pdf, name=name)

    @abstractmethod
    def _create_pdf(self, obs, **params):
        """Create the underlying PDF. Must be implemented by subclasses."""

    def _register_default_param(self, param):
        """Register a parameter and potentially adapt the prior's range."""
        super()._register_default_param(param)

        # Check if we should adapt to parameter limits
        if hasattr(param, "has_limits") and param.has_limits:
            self._adapt_to_parameter_limits(param)

        return self

    def _get_adapted_bounds(self, param) -> tuple[float, float]:
        """Get bounds adapted to parameter limits."""
        lower, upper = self._original_bounds

        # Use parameter limits if available
        if hasattr(param, "lower") and param.lower is not None:
            if lower == -float("inf") or (self._requires_positive and lower < param.lower):
                lower = param.lower
            else:
                lower = max(lower, param.lower)

        if hasattr(param, "upper") and param.upper is not None:
            upper = param.upper if upper == float("inf") else min(upper, param.upper)

        # Ensure positive values if required
        if self._requires_positive:
            lower = max(0, lower)

        return lower, upper

    def _adapt_to_parameter_limits(self, param):
        """Adapt the prior's observation space to the parameter limits."""
        lower, upper = self._get_adapted_bounds(param)

        # Only update if bounds actually changed
        if (lower, upper) != self._original_bounds:
            obs = zfit.Space("prior_space", limits=(lower, upper))
            self.pdf = self._create_adapted_pdf(obs, lower, upper)

    def _create_adapted_pdf(self, obs, _lower, _upper):
        """Create an adapted PDF with new bounds. Can be overridden for special handling."""
        return self._create_pdf(obs, **self._pdf_params)


class Normal(AdaptivePrior):
    """Normal (Gaussian) prior distribution.

    The Normal prior is one of the most commonly used priors in Bayesian inference.
    It represents beliefs that parameter values follow a bell-shaped distribution
    centered at a specific value with a given spread. When assigned to a parameter
    with bounds, it automatically becomes a truncated normal distribution.

    This prior is suitable for:
    - Parameters where you have informative prior knowledge about the likely value
    - Regression coefficients and effect sizes
    - Location parameters when uncertainty is approximately symmetric

    Example:
        >>> # Prior centered at 0 with standard deviation 1
        >>> prior = Normal(mu=0.0, sigma=1.0)
        >>>
        >>> # Prior for a parameter expected around 10 with uncertainty ±2
        >>> prior = Normal(mu=10.0, sigma=2.0)
    """

    def __init__(self, mu, sigma, name=None):
        """Initialize a Normal prior.

        Args:
            mu: Mean (center) of the normal distribution
            sigma: Standard deviation (spread) of the normal distribution. Must be positive.
            name: Optional name for the prior
        """
        pdf_params = {"mu": mu, "sigma": sigma}
        super().__init__(pdf_params, name=name)

    def _create_pdf(self, obs, mu, sigma):
        """Create a Gaussian PDF."""
        return zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    def _create_adapted_pdf(self, obs, lower, upper):
        """Create a truncated Gaussian when adapting to limits."""
        return zfit.pdf.TruncatedGauss(
            mu=self._pdf_params["mu"], sigma=self._pdf_params["sigma"], low=lower, high=upper, obs=obs
        )


class Uniform(AdaptivePrior):
    """Uniform prior distribution.

    The Uniform prior assigns equal probability to all values within a specified
    range. It represents complete uncertainty about which values are more likely
    within the bounds. This is often used as a "non-informative" or "flat" prior
    when you want the data to dominate the inference.

    Special behavior:
    - If bounds are not specified, the prior adapts to the parameter's limits
    - If only one bound is specified, a default range of 1e6 is used for the other

    This prior is suitable for:
    - Parameters where you have no prior preference within a range
    - Initial explorations when prior knowledge is minimal
    - Bounded parameters where all values are equally plausible

    Example:
        >>> # Uniform prior between 0 and 1
        >>> prior = Uniform(lower=0.0, upper=1.0)
        >>>
        >>> # Uniform prior that adapts to parameter bounds
        >>> prior = Uniform()  # Will use parameter's limits when assigned
    """

    def __init__(self, lower=None, upper=None, name=None):
        """Initialize a Uniform prior.

        Args:
            lower: Lower bound of the uniform distribution. If None, will use
                   the parameter's lower limit when the prior is assigned.
            upper: Upper bound of the uniform distribution. If None, will use
                   the parameter's upper limit when the prior is assigned.
            name: Optional name for the prior

        Note:
            If both bounds are None, the prior will adapt completely to the
            parameter's limits. If the parameter has no limits, temporary
            bounds of [-1e6, 1e6] are used.
        """
        self._user_lower = lower
        self._user_upper = upper

        # Determine initial bounds
        if lower is None and upper is None:
            bounds = (-1e6, 1e6)  # temporary defaults
        elif lower is None:
            bounds = (upper - 1e6, upper)
        elif upper is None:
            bounds = (lower, lower + 1e6)
        else:
            bounds = (lower, upper)

        pdf_params = {}  # Uniform doesn't need location/scale params
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _create_pdf(self, obs, **_params):
        """Create a Uniform PDF."""
        lower = float(obs.lower[0][0])
        upper = float(obs.upper[0][0])
        return zfit.pdf.Uniform(low=lower, high=upper, obs=obs)

    def _get_adapted_bounds(self, param):
        """Get bounds adapted to parameter limits, preferring user bounds."""
        lower = self._user_lower
        upper = self._user_upper

        # Use parameter limits only if user didn't specify
        if lower is None and hasattr(param, "lower") and param.lower is not None:
            lower = param.lower
        if upper is None and hasattr(param, "upper") and param.upper is not None:
            upper = param.upper

        # Fallback to original bounds if still None
        if lower is None or upper is None:
            orig_lower, orig_upper = self._original_bounds
            if lower is None:
                lower = orig_lower
            if upper is None:
                upper = orig_upper

        return lower, upper


class HalfNormal(AdaptivePrior):
    """Half-normal prior distribution.

    The Half-Normal prior is a normal distribution truncated at a lower bound
    (typically zero). It's ideal for parameters that must be positive and where
    smaller values are more likely than larger ones. The distribution has its
    mode at the truncation point and decreases monotonically.

    This prior is suitable for:
    - Standard deviations and scale parameters
    - Variance components in hierarchical models
    - Any positive parameter where smaller values are preferred
    - Error terms and measurement uncertainties

    Example:
        >>> # Half-normal starting at 0 with scale 1
        >>> prior = HalfNormal(sigma=1.0)
        >>>
        >>> # Half-normal starting at 2 with scale 0.5
        >>> prior = HalfNormal(mu=2.0, sigma=0.5)
    """

    _requires_positive = True

    def __init__(self, *, sigma, mu=0, name=None):
        """Initialize a Half-Normal prior.

        Args:
            sigma: Scale parameter controlling the spread of the distribution.
                   Must be positive. Required keyword-only parameter.
            mu: Location parameter (lower bound) where the distribution starts.
                Defaults to 0 for a standard half-normal.
            name: Optional name for the prior
        """
        pdf_params = {"mu": mu, "sigma": sigma}
        bounds = (mu, float("inf"))
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _create_pdf(self, obs, mu, sigma):
        """Create a half-normal using TruncatedGauss."""
        lower = float(obs.lower[0][0])
        upper = float(obs.upper[0][0])
        return zfit.pdf.TruncatedGauss(mu=mu, sigma=sigma, low=lower, high=upper, obs=obs)

    def _get_adapted_bounds(self, param):
        """Ensure lower bound respects the mu parameter."""
        lower, upper = super()._get_adapted_bounds(param)
        # Ensure lower bound is at least mu
        lower = max(self._pdf_params["mu"], lower)
        return lower, upper


class Gamma(AdaptivePrior):
    """Gamma prior distribution.

    The Gamma distribution is a flexible family of continuous probability
    distributions for positive values. Its shape can range from exponential-like
    (alpha=1) to bell-shaped (larger alpha values). The distribution is commonly
    used in Bayesian inference due to its conjugacy properties with certain
    likelihoods.

    This prior is suitable for:
    - Rate parameters and inverse scales
    - Precision parameters (inverse of variance)
    - Waiting times and lifetimes
    - Any positive parameter with flexibility in shape

    The parameterization used here:
    - Mean = alpha/beta (when mu=0)
    - Variance = alpha/beta²

    Example:
        >>> # Gamma prior with shape=2, rate=1
        >>> prior = Gamma(alpha=2.0, beta=1.0)
        >>>
        >>> # Shifted Gamma starting at 0.5
        >>> prior = Gamma(alpha=2.0, beta=1.0, mu=0.5)
    """

    _requires_positive = True

    def __init__(self, alpha, beta, mu=0, name=None):
        """Initialize a Gamma prior.

        Args:
            alpha: Shape parameter controlling the form of the distribution.
                   Must be positive. Larger values make the distribution more
                   bell-shaped.
            beta: Rate parameter (inverse scale). Must be positive.
                  Larger values shift the distribution toward zero.
            mu: Location parameter that shifts the entire distribution.
                Defaults to 0 for a standard Gamma. The distribution
                has support on [mu, ∞).
            name: Optional name for the prior
        """
        pdf_params = {"gamma": alpha, "beta": beta, "mu": mu}
        bounds = (mu, float("inf"))
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _create_pdf(self, obs, gamma, beta, mu):
        """Create a Gamma PDF."""
        return zfit.pdf.Gamma(gamma=gamma, beta=beta, mu=mu, obs=obs)

    def _create_adapted_pdf(self, obs, lower, _upper):
        """Create adapted Gamma PDF with adjusted mu."""
        # Note: zfit's Gamma doesn't support upper truncation directly
        # Adjust mu to match the lower bound
        adapted_mu = max(self._pdf_params["mu"], lower)
        return zfit.pdf.Gamma(gamma=self._pdf_params["gamma"], beta=self._pdf_params["beta"], mu=adapted_mu, obs=obs)


class Beta(AdaptivePrior):
    """Beta prior distribution.

    The Beta distribution is the natural choice for parameters bounded between
    0 and 1. It's extremely flexible, capable of representing uniform, U-shaped,
    bell-shaped, or skewed distributions within [0,1]. The Beta is the conjugate
    prior for binomial and Bernoulli likelihoods.

    This prior is suitable for:
    - Probabilities and proportions
    - Mixing parameters and weights
    - Success rates and frequencies
    - Any parameter naturally bounded in [0, 1]

    Special cases:
    - Beta(1, 1): Uniform distribution on [0, 1]
    - Beta(0.5, 0.5): Jeffreys prior for binomial proportion
    - Beta(alpha, beta) with alpha=beta: Symmetric distribution around 0.5

    Example:
        >>> # Uniform prior on [0, 1]
        >>> prior = Beta(alpha=1.0, beta=1.0)
        >>>
        >>> # Prior favoring values near 0.8
        >>> prior = Beta(alpha=8.0, beta=2.0)
        >>>
        >>> # U-shaped prior (Jeffreys prior)
        >>> prior = Beta(alpha=0.5, beta=0.5)
    """

    _default_bounds = (0, 1)

    def __init__(self, alpha, beta, name=None):
        """Initialize a Beta prior.

        Args:
            alpha: First shape parameter controlling the behavior near 1.
                   Must be positive. Larger values push probability mass
                   toward 1.
            beta: Second shape parameter controlling the behavior near 0.
                  Must be positive. Larger values push probability mass
                  toward 0.
            name: Optional name for the prior

        Note:
            The mean of Beta(alpha, beta) is alpha/(alpha+beta) and the variance decreases
            as alpha+beta increases, making the distribution more concentrated.
        """
        pdf_params = {"alpha": alpha, "beta": beta}
        super().__init__(pdf_params, name=name)

    def _create_pdf(self, obs, alpha, beta):
        """Create a Beta PDF."""
        return zfit.pdf.Beta(alpha=alpha, beta=beta, obs=obs)

    def _adapt_to_parameter_limits(self, param):
        """Beta prior doesn't adapt to parameter limits by default."""
        # Beta is specifically for [0,1] parameters
        # Future enhancement could implement scaled/shifted Beta


class LogNormal(AdaptivePrior):
    """Log-normal prior distribution.

    The Log-Normal distribution arises when the logarithm of a variable is
    normally distributed. It's right-skewed and only defined for positive values.
    This makes it useful for parameters that are positive and potentially have
    a long right tail.

    This prior is suitable for:
    - Parameters with multiplicative effects
    - Sizes, lengths, and other positive measurements
    - Parameters where relative changes are more important than absolute
    - Economic variables like income or prices
    - Any positive parameter with potential for extreme values

    Properties:
    - If X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma)
    - Mode = exp(mu - sigma^2)
    - Median = exp(mu)
    - Mean = exp(mu + sigma^2/2)

    Example:
        >>> # Log-normal with median at 1
        >>> prior = LogNormal(mu=0.0, sigma=1.0)
        >>>
        >>> # Log-normal with median at 10 and moderate spread
        >>> prior = LogNormal(mu=2.303, sigma=0.5)  # 2.303 ≈ log(10)
    """

    _default_bounds = (0, float("inf"))
    _requires_positive = True

    def __init__(self, mu, sigma, name=None):
        """Initialize a Log-Normal prior.

        Args:
            mu: Mean of the logarithm of the variable. This controls the
                median of the distribution (median = exp(mu)).
            sigma: Standard deviation of the logarithm. Must be positive.
                   Larger values create heavier right tails.
            name: Optional name for the prior

        Note:
            The parameters mu and sigma are NOT the mean and standard deviation
            of the Log-Normal distribution itself, but of the underlying
            normal distribution of log(X).
        """
        pdf_params = {"mu": mu, "sigma": sigma}
        super().__init__(pdf_params, name=name)

    def _create_pdf(self, obs, mu, sigma):
        """Create a LogNormal PDF."""
        return zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs)


class Cauchy(AdaptivePrior):
    """Cauchy prior distribution.

    The Cauchy distribution is a heavy-tailed distribution that doesn't have
    a defined mean or variance. Its heavy tails make it useful for robust
    inference when outliers are expected or when you want to express very
    weak prior information.

    This prior is suitable for:
    - Robust modeling when outliers are expected
    - Parameters where extreme values are plausible
    - Weakly informative priors that still allow for surprises
    - Location parameters with high uncertainty

    Properties:
    - No finite mean or variance (moments don't exist)
    - Median and mode equal the location parameter m
    - Heavy tails decrease as 1/x²

    Example:
        >>> # Cauchy centered at 0 with scale 1 (standard Cauchy)
        >>> prior = Cauchy(m=0.0, gamma=1.0)
        >>>
        >>> # Cauchy centered at 10 with wider scale
        >>> prior = Cauchy(m=10.0, gamma=5.0)

    Warning:
        Due to its heavy tails, the Cauchy can lead to slower MCMC
        convergence. Consider using a Student-t with low degrees of
        freedom as an alternative with finite variance.
    """

    def __init__(self, m, gamma, name=None):
        """Initialize a Cauchy prior.

        Args:
            m: Location parameter (center) of the distribution. This is
               both the median and the mode.
            gamma: Scale parameter controlling the spread. Must be positive.
                   Larger values create wider distributions.
            name: Optional name for the prior
        """
        pdf_params = {"m": m, "gamma": gamma}
        super().__init__(pdf_params, name=name)

    def _create_pdf(self, obs, m, gamma):
        """Create a Cauchy PDF."""
        return zfit.pdf.Cauchy(m=m, gamma=gamma, obs=obs)


class KDE(ZfitPrior):
    """Kernel Density Estimate prior from samples.

    The KDE prior is a non-parametric way to construct a prior distribution
    from empirical samples. It estimates the probability density by placing
    kernels (typically Gaussian) at each sample point. This is particularly
    useful for hierarchical Bayesian models where posterior samples from one
    analysis become priors for another.

    This prior is suitable for:
    - Using posterior samples from previous analyses as priors

    Implementation details:
    - Uses exact KDE for < 1000 samples (more accurate but slower)
    - Switches to grid-based KDE for larger samples (faster approximation)
    - Automatically adds margin to bounds for numerical stability
    - Adapts to parameter limits while preserving sample structure

    Example:
        >>> # Create prior from previous posterior samples
        >>> posterior_samples = previous_result.samples['param_name']
        >>> prior = KDE(posterior_samples)
        >>>
        >>> # KDE with custom bandwidth
        >>> prior = KDE(samples, bandwidth=0.1)
        >>>
        >>> # Using KDE for hierarchical modeling
        >>> group1_posterior = fit1.samples['effect']
        >>> group2_prior = KDE(group1_posterior)
    """

    def __init__(self, samples, bandwidth=None, name=None):
        """Initialize a KDE prior.

        Args:
            samples: Array of samples to estimate the density from.
                     Can be a numpy array, list, or tensor.
            bandwidth: Bandwidth for kernel density estimation. If None,
                      uses the KDE's default.
            name: Optional name for the prior

        Note:
            The KDE is constructed with a 10% margin beyond the sample
            range to ensure numerical stability at the boundaries.
        """
        samples = znp.asarray(samples)
        self._samples = samples
        self._bandwidth = bandwidth

        # Calculate bounds with margin
        min_val = float(znp.min(samples))
        max_val = float(znp.max(samples))
        margin = 0.1 * (max_val - min_val)
        margin = max(margin, 1e-6)  # Ensure minimum margin

        self._min_val = min_val
        self._max_val = max_val
        self._margin = margin
        self._original_bounds = (min_val - margin, max_val + margin)

        # Create initial PDF
        pdf = self._create_kde_pdf(self._original_bounds)
        super().__init__(pdf=pdf, name=name)

    def _create_kde_pdf(self, bounds):
        """Create KDE PDF with given bounds."""
        lower, upper = bounds
        obs = zfit.Space("prior_space", limits=(lower, upper))
        data = zfit.Data.from_numpy(obs=obs, array=self._samples)
        n_samples = len(self._samples)

        if n_samples < 1000:
            return zfit.pdf.KDE1DimExact(data=data, obs=obs, bandwidth=self._bandwidth, padding=False)
        else:
            return zfit.pdf.KDE1DimGrid(
                data=data,
                obs=obs,
                bandwidth=self._bandwidth or "scott",
                num_grid_points=min(1024, n_samples // 5),
                padding=False,
            )

    def _register_default_param(self, param):
        """Register a parameter and potentially adapt the KDE bounds."""
        super()._register_default_param(param)

        # Check if we should adapt to parameter limits
        if hasattr(param, "has_limits") and param.has_limits:
            self._adapt_to_parameter_limits(param)

        return self

    def _adapt_to_parameter_limits(self, param):
        """Adapt KDE prior to parameter limits."""
        lower = self._min_val - self._margin
        upper = self._max_val + self._margin

        # Expand to parameter limits if they're wider
        if hasattr(param, "lower") and param.lower is not None:
            lower = min(lower, param.lower)
        if hasattr(param, "upper") and param.upper is not None:
            upper = max(upper, param.upper)

        # Only recreate if bounds changed
        if (lower, upper) != self._original_bounds:
            self.pdf = self._create_kde_pdf((lower, upper))


class Poisson(AdaptivePrior):
    """Poisson prior distribution.

    The Poisson distribution is a discrete probability distribution that models
    the number of events occurring in a fixed interval of time or space. It's
    particularly useful for parameters that represent counts, rates, or other
    discrete positive quantities.

    Properties:
    - Support: Non-negative integers {0, 1, 2, ...}
    - Mean = Variance = λ (rate parameter)
    - Mode = floor(λ) if λ is not an integer, otherwise λ and λ-1

    Example:
        >>> # Prior for a count parameter expecting ~3 events
        >>> prior = Poisson(lam=3.0)
        >>>
        >>> # Prior for rare events
        >>> prior = Poisson(lam=0.5)
    """

    _default_bounds = (0, 50)  # Reasonable upper bound for Poisson
    _requires_positive = True

    def __init__(self, lam, name=None):
        """Initialize a Poisson prior.

        Args:
            lam: Rate parameter (expected number of events). Must be positive.
                 This is both the mean and variance of the distribution.
            name: Optional name for the prior
        """
        pdf_params = {"lam": lam}
        # Use reasonable bounds for Poisson distribution
        upper_bound = max(50, int(float(lam) * 10))  # 10x the mean should cover most cases
        bounds = (0, upper_bound)
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _create_pdf(self, obs, lam):
        """Create a Poisson PDF."""
        return zfit.pdf.Poisson(lam=lam, obs=obs)

    def _get_adapted_bounds(self, param):
        """Ensure bounds are appropriate for discrete distribution."""
        lower, upper = super()._get_adapted_bounds(param)
        # Poisson is non-negative
        lower = max(0, lower)
        return lower, upper


class Exponential(AdaptivePrior):
    """Exponential prior distribution.

    The Exponential distribution is a memoryless continuous distribution often
    used to model waiting times, lifetimes, or inter-arrival times. It's the
    continuous analog of the geometric distribution and has a constant hazard rate.

    This prior is suitable for:
    - Rate parameters and inverse time scales
    - Waiting times and inter-arrival intervals
    - Lifetimes and survival analysis
    - Any parameter representing decay or hazard rates
    - Regularization when expecting small positive values

    Properties:
    - Support: Non-negative real numbers [0, ∞)
    - Mean = 1/λ, Variance = 1/λ²
    - Mode = 0 (exponential decay from maximum at 0)
    - Memoryless property: P(X > s+t | X > s) = P(X > t)

    Example:
        >>> # Prior for a rate parameter with expected value 1/2 = 0.5
        >>> prior = Exponential(lam=2.0)
        >>>
        >>> # Prior for a decay constant
        >>> prior = Exponential(lam=1.0)  # Mean = 1
    """

    _default_bounds = (0, 20)  # Reasonable upper bound for exponential
    _requires_positive = True

    def __init__(self, lam, name=None):
        """Initialize an Exponential prior.

        Args:
            lam: Rate parameter (inverse of the mean). Must be positive.
                 Higher values concentrate probability near zero.
            name: Optional name for the prior
        """
        pdf_params = {"lam": lam}
        # Use reasonable bounds for exponential distribution
        # Mean is 1/lam, so use several standard deviations as upper bound
        upper_bound = max(20, 5.0 / float(lam))  # ~5 standard deviations
        bounds = (0, upper_bound)
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _create_pdf(self, obs, lam):
        """Create an Exponential PDF."""
        return zfit.pdf.Exponential(lam=lam, obs=obs)


class StudentT(AdaptivePrior):
    """Student's t-distribution prior.

    The Student's t-distribution is a heavy-tailed distribution that approaches
    the normal distribution as degrees of freedom increase. It's useful when
    you want robustness against outliers while maintaining finite variance
    (unlike the Cauchy distribution).

    This prior is suitable for:
    - Robust inference with outlier resistance
    - Parameters where extreme values are possible but not as likely as in Cauchy
    - Alternative to Normal when heavier tails are desired
    - Location parameters with moderate uncertainty
    - Regression coefficients in robust modeling

    Properties:
    - Support: All real numbers (-∞, ∞)
    - Mean = μ (for ndof > 1), undefined for ndof ≤ 1
    - Variance = σ²·ndof/(ndof-2) (for ndof > 2)
    - Approaches Normal(μ, σ) as ndof → ∞
    - Heavier tails than normal for small ndof

    Example:
        >>> # Heavy-tailed prior (like Cauchy but with finite variance)
        >>> prior = StudentT(ndof=3, mu=0.0, sigma=1.0)
        >>>
        >>> # Moderately robust prior
        >>> prior = StudentT(ndof=10, mu=5.0, sigma=2.0)
    """

    def __init__(self, ndof, mu, sigma, name=None):
        """Initialize a Student's t prior.

        Args:
            ndof: Degrees of freedom parameter controlling tail heaviness.
                  Must be positive. Lower values give heavier tails.
                  As ndof → ∞, approaches Normal(mu, sigma).
            mu: Location parameter (center) of the distribution.
            sigma: Scale parameter controlling spread. Must be positive.
            name: Optional name for the prior
        """
        pdf_params = {"ndof": ndof, "mu": mu, "sigma": sigma}
        super().__init__(pdf_params, name=name)

    def _create_pdf(self, obs, ndof, mu, sigma):
        """Create a Student's t PDF."""
        return zfit.pdf.StudentT(ndof=ndof, mu=mu, sigma=sigma, obs=obs)

    def _create_adapted_pdf(self, obs, lower, upper):
        """Create adapted Student's t PDF with truncation if needed."""
        # Note: zfit's StudentT doesn't support direct truncation
        # For now, we'll use the regular StudentT and let the parameter bounds handle it
        return zfit.pdf.StudentT(
            ndof=self._pdf_params["ndof"], mu=self._pdf_params["mu"], sigma=self._pdf_params["sigma"], obs=obs
        )
