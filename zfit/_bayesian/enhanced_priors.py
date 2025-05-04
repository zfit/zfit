"""Enhanced Prior distributions for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .priors import BasePrior, GammaPrior, HalfNormalPrior


class StudentTPrior(BasePrior):
    """Student-t prior distribution for robust modeling."""

    def __init__(self, mu, sigma, nu, name=None):
        """Initialize a Student-t prior.

        Args:
            mu: Location parameter
            sigma: Scale parameter
            nu: Degrees of freedom
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))
        pdf = zfit.pdf.StudentT(mu=mu, sigma=sigma, ndof=nu, obs=obs)
        super().__init__(pdf=pdf, name=name)


class LaplacePrior(BasePrior):
    """Laplace (double exponential) prior distribution, useful for sparse parameters."""

    def __init__(self, loc, scale, name=None):
        """Initialize a Laplace prior.

        Args:
            loc: Location parameter
            scale: Scale parameter
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))
        pdf = zfit.pdf.Exponential(rate=1.0 / scale, obs=obs)
        # Transform to center at loc (Double-sided exponential)
        pdf = pdf.transform_coords(lambda x: tf.abs(x - loc))
        super().__init__(pdf=pdf, name=name)


class CauchyPrior(BasePrior):
    """Cauchy prior distribution, useful for heavy-tailed priors."""

    def __init__(self, loc, scale, name=None):
        """Initialize a Cauchy prior.

        Args:
            loc: Location parameter
            scale: Scale parameter
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))
        pdf = zfit.pdf.Cauchy(m=loc, gamma=scale, obs=obs)
        super().__init__(pdf=pdf, name=name)


class LogNormalPrior(BasePrior):
    """Log-normal prior distribution for strictly positive parameters."""

    def __init__(self, mu, sigma, name=None):
        """Initialize a Log-Normal prior.

        Args:
            mu: Mean of the log of the random variable
            sigma: Standard deviation of the log of the random variable
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(0, float("inf")))
        pdf = zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs)
        super().__init__(pdf=pdf, name=name)


class InverseGammaPrior(BasePrior):
    """Inverse-gamma prior distribution, commonly used for variance parameters."""

    def __init__(self, alpha, beta, name=None):
        """Initialize an Inverse-Gamma prior.

        Args:
            alpha: Shape parameter (α > 0)
            beta: Scale parameter (β > 0)
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(0, float("inf")))

        # Define the inverse gamma PDF using a custom PDF
        def _log_pdf(x):
            return alpha * tf.math.log(beta) - tf.math.lgamma(alpha) - (alpha + 1) * tf.math.log(x) - beta / x

        # Create a wrapper for the unnormalized PDF
        pdf = zfit.pdf.UnbinnedPDF(_log_pdf, obs=obs, name="InverseGamma")

        super().__init__(pdf=pdf, name=name)


class HierarchicalPrior(BasePrior):
    """Hierarchical prior with hyperparameters."""

    def __init__(self, base_pdf, hyperparams, name=None):
        """Initialize a hierarchical prior.

        Args:
            base_pdf: Base PDF to use
            hyperparams: Dictionary mapping hyperparameter names to values
            name: Name of the prior
        """

        # Extract the observation space from the base PDF

        # Cache hyperparameters
        self.hyperparams = hyperparams
        self.base_pdf = base_pdf

        super().__init__(pdf=base_pdf, name=name)

    def update_hyperparams(self, **kwargs):
        """Update the hyperparameters of the prior.

        Args:
            **kwargs: New hyperparameter values
        """
        for param_name, value in kwargs.items():
            if param_name in self.hyperparams:
                self.hyperparams[param_name] = value

                # Update the base PDF parameter
                if hasattr(self.base_pdf, param_name):
                    setattr(self.base_pdf, param_name, value)


class MixturePrior(BasePrior):
    """Mixture of priors for flexible modeling."""

    def __init__(self, priors, fracs, name=None):
        """Initialize a mixture prior.

        Args:
            priors: List of prior PDFs to mix
            fracs: List of mixing fractions (must sum to 1)
            name: Name of the prior
        """
        import zfit

        # Check that all priors have the same observation space
        obs = priors[0].obs
        for prior in priors[1:]:
            if prior.obs != obs:
                msg = "All priors must have the same observation space"
                raise ValueError(msg)

        # Create a mixture PDF
        pdf = zfit.pdf.SumPDF(pdfs=priors, fracs=fracs[:-1], obs=obs)

        super().__init__(pdf=pdf, name=name)


class HalfCauchyPrior(BasePrior):
    """Half-Cauchy prior, useful for scale parameters."""

    def __init__(self, loc, scale, name=None):
        """Initialize a Half-Cauchy prior.

        Args:
            loc: Location parameter
            scale: Scale parameter
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(loc, float("inf")))
        pdf = zfit.pdf.Cauchy(m=loc, gamma=scale, obs=obs)
        super().__init__(pdf=pdf, name=name)


class KDEPrior(BasePrior):
    """Kernel Density Estimate prior based on samples.

    This can be used to create priors from posterior samples
    for hierarchical modeling and prior updating.
    """

    def __init__(self, samples, bandwidth=None, name=None):
        """Initialize a KDE prior.

        Args:
            samples: Array of samples to estimate density from
            bandwidth: Bandwidth for KDE. If None, use Scott's rule
            name: Name of the prior
        """
        import zfit

        # Calculate min and max for observation space
        min_val = np.min(samples)
        max_val = np.max(samples)

        # Add a margin
        margin = 0.1 * (max_val - min_val)
        min_val -= margin
        max_val += margin

        # Create observation space
        obs = zfit.Space("prior_space", limits=(min_val, max_val))

        # Calculate bandwidth if not provided
        if bandwidth is None:
            # Scott's rule
            bandwidth = 1.06 * np.std(samples) * len(samples) ** (-1 / 5)

        self.samples = np.array(samples)
        self.bandwidth = bandwidth

        # Define the KDE PDF using a custom PDF
        def _kde_log_pdf(x):
            # Calculate the KDE using a Gaussian kernel
            x_expanded = tf.expand_dims(x, 1)  # Shape: [batch_size, 1]
            samples_expanded = tf.expand_dims(self.samples, 0)  # Shape: [1, n_samples]

            # Calculate squared distances
            sq_dists = tf.square(x_expanded - samples_expanded)  # Shape: [batch_size, n_samples]

            # Apply Gaussian kernel
            kernel_values = tf.exp(-0.5 * sq_dists / (self.bandwidth**2))

            # Average over samples
            kde_values = tf.reduce_mean(kernel_values, axis=1)

            # Return log of KDE values
            return tf.math.log(kde_values + 1e-10)  # Add small constant for numerical stability

        # Create a wrapper for the KDE PDF
        pdf = zfit.pdf.UnbinnedPDF(_kde_log_pdf, obs=obs, name="KDE")

        super().__init__(pdf=pdf, name=name)


class TruncatedPrior(BasePrior):
    """Truncated version of any prior."""

    def __init__(self, base_prior, lower=None, upper=None, name=None):
        """Initialize a truncated prior.

        Args:
            base_prior: The base prior to truncate
            lower: Lower bound (if None, use -infinity)
            upper: Upper bound (if None, use infinity)
            name: Name of the prior
        """
        import zfit

        # Define bounds
        lower = lower if lower is not None else -float("inf")
        upper = upper if upper is not None else float("inf")

        # Create new observation space with the truncated limits
        obs = zfit.Space("prior_space", limits=(lower, upper))

        # Create a truncated version of the base PDF
        pdf = zfit.pdf.TruncatedPDF(pdf=base_prior.pdf, obs=obs)

        super().__init__(pdf=pdf, name=name)


class DiscreteUniformPrior(BasePrior):
    """Discrete uniform prior for integer parameters."""

    def __init__(self, low, high, name=None):
        """Initialize a discrete uniform prior.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
            name: Name of the prior
        """
        import zfit

        # Create observation space
        obs = zfit.Space("prior_space", limits=(low - 0.5, high + 0.5))

        # Create a histogram PDF with uniform bins
        bins = np.arange(low - 0.5, high + 1.5, 1.0)
        values = np.ones(len(bins) - 1) / (high - low + 1)

        pdf = zfit.pdf.HistogramPDF(values, bins, obs=obs)

        super().__init__(pdf=pdf, name=name)


class JeffreysPrior(BasePrior):
    """Jeffreys prior, which is invariant to reparameterization.

    This is often used as a non-informative prior for scale parameters.
    For a scale parameter, Jeffreys prior is proportional to 1/x.
    """

    def __init__(self, lower, upper, name=None):
        """Initialize a Jeffreys prior.

        Args:
            lower: Lower bound (must be positive)
            upper: Upper bound
            name: Name of the prior
        """
        import zfit

        if lower <= 0:
            msg = "Lower bound must be positive for Jeffreys prior"
            raise ValueError(msg)

        obs = zfit.Space("prior_space", limits=(lower, upper))

        # Define Jeffreys prior PDF (proportional to 1/x)
        def _log_pdf(x):
            return -tf.math.log(x)

        # Create a wrapper for the unnormalized PDF
        pdf = zfit.pdf.UnbinnedPDF(_log_pdf, obs=obs, name="Jeffreys")

        super().__init__(pdf=pdf, name=name)


class DirichletPrior(BasePrior):
    """Dirichlet prior for compositional data (e.g., fractions that sum to 1)."""

    def __init__(self, alpha, name=None):
        """Initialize a Dirichlet prior.

        Args:
            alpha: Concentration parameters (must be positive)
            name: Name of the prior
        """
        import zfit

        # For simplicity, we implement this for the 2D case (simplex)
        # where we only need one parameter (the other is 1 - x)
        if len(alpha) != 2:
            msg = "Only 2D Dirichlet prior is currently implemented"
            raise NotImplementedError(msg)

        obs = zfit.Space("prior_space", limits=(0, 1))

        # Define Beta PDF (2D Dirichlet is a Beta)
        alpha1, alpha2 = alpha
        pdf = zfit.pdf.Beta(alpha=alpha1, beta=alpha2, obs=obs)

        super().__init__(pdf=pdf, name=name)


class InformativePrior(BasePrior):
    """Informative prior based on previous knowledge."""

    def __init__(self, mu, sigma, nu=5, name=None, distribution="student_t"):
        """Initialize an informative prior.

        Args:
            mu: Prior mean
            sigma: Prior standard deviation
            nu: Degrees of freedom for Student-t distribution
            name: Name of the prior
            distribution: Type of distribution ("normal", "student_t" or "cauchy")
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))

        if distribution == "normal":
            pdf = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
        elif distribution == "student_t":
            pdf = zfit.pdf.StudentT(mu=mu, sigma=sigma, nu=nu, obs=obs)
        elif distribution == "cauchy":
            pdf = zfit.pdf.Cauchy(m=mu, gamma=sigma, obs=obs)
        else:
            msg = f"Unknown distribution: {distribution}"
            raise ValueError(msg)

        super().__init__(pdf=pdf, name=name)


class WeaklyInformativePrior(BasePrior):
    """Weakly informative prior for regularization."""

    def __init__(self, scale=1.0, name=None, distribution="student_t"):
        """Initialize a weakly informative prior.

        Args:
            scale: Scale parameter
            name: Name of the prior
            distribution: Type of distribution ("normal", "student_t", "cauchy" or "laplace")
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))

        if distribution == "normal":
            pdf = zfit.pdf.Gauss(mu=0, sigma=scale, obs=obs)
        elif distribution == "student_t":
            pdf = zfit.pdf.StudentT(mu=0, sigma=scale, nu=3, obs=obs)
        elif distribution == "cauchy":
            pdf = zfit.pdf.Cauchy(m=0, gamma=scale, obs=obs)
        elif distribution == "laplace":
            # Implemented via double-sided exponential
            pdf = zfit.pdf.Exponential(rate=1.0 / scale, obs=obs)
            pdf = pdf.transform_coords(lambda x: tf.abs(x))
        else:
            msg = f"Unknown distribution: {distribution}"
            raise ValueError(msg)

        super().__init__(pdf=pdf, name=name)


class SpikeAndSlabPrior(BasePrior):
    """Spike and Slab prior for variable selection.

    This prior is a mixture of a point mass at zero (spike) and a continuous
    distribution (slab), useful for regression with variable selection.
    """

    def __init__(self, weight_at_zero=0.5, slab_scale=1.0, name=None):
        """Initialize a Spike and Slab prior.

        Args:
            weight_at_zero: Probability mass at zero (spike)
            slab_scale: Scale parameter for the slab component
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))

        # Create a normal distribution for the slab
        slab = zfit.pdf.Gauss(mu=0, sigma=slab_scale, obs=obs)

        # Create a spike at zero (approximated by a very narrow normal)
        spike_scale = 1e-6  # Very small scale to approximate a spike
        spike = zfit.pdf.Gauss(mu=0, sigma=spike_scale, obs=obs)

        # Create a mixture
        pdf = zfit.pdf.SumPDF(pdfs=[spike, slab], fracs=[weight_at_zero], obs=obs)

        super().__init__(pdf=pdf, name=name)


class BoundedPrior(BasePrior):
    """A prior that ensures bounds are respected."""

    def __init__(self, base_prior, lower=None, upper=None, name=None):
        """Initialize a bounded prior.

        Args:
            base_prior: Base prior distribution
            lower: Lower bound
            upper: Upper bound
            name: Name of the prior
        """
        # This is essentially a truncated prior with a different name
        import zfit

        # Define bounds
        lower = lower if lower is not None else -float("inf")
        upper = upper if upper is not None else float("inf")

        # Create new observation space with the bounds
        obs = zfit.Space("prior_space", limits=(lower, upper))

        # Create a truncated version of the base PDF
        pdf = zfit.pdf.TruncatedPDF(pdf=base_prior.pdf, obs=obs)

        super().__init__(pdf=pdf, name=name)


class RegularizedPrior(BasePrior):
    """Prior for adding regularization to parameters."""

    def __init__(self, strength=1.0, l1_ratio=0.0, name=None):
        """Initialize a regularized prior (elastic net style).

        Args:
            strength: Overall regularization strength
            l1_ratio: Ratio of L1 penalty (0 = ridge, 1 = lasso, between 0-1 = elastic net)
            name: Name of the prior
        """
        import zfit

        obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))

        if l1_ratio == 0.0:
            # Ridge regression (normal prior)
            pdf = zfit.pdf.Gauss(mu=0, sigma=1.0 / strength, obs=obs)
        elif l1_ratio == 1.0:
            # Lasso regression (Laplace prior)
            pdf = zfit.pdf.Exponential(rate=strength, obs=obs)
            pdf = pdf.transform_coords(lambda x: tf.abs(x))
        else:
            # Elastic net (weighted mixture of Gaussian and Laplace)
            ridge = zfit.pdf.Gauss(mu=0, sigma=1.0 / strength, obs=obs)

            lasso = zfit.pdf.Exponential(rate=strength, obs=obs)
            lasso = lasso.transform_coords(lambda x: tf.abs(x))

            pdf = zfit.pdf.SumPDF(pdfs=[ridge, lasso], fracs=[1.0 - l1_ratio], obs=obs)

        super().__init__(pdf=pdf, name=name)


class EmpiricalPrior(BasePrior):
    """Empirical prior created from data."""

    def __init__(self, data, kernel_width=None, name=None):
        """Initialize an empirical prior from data.

        Args:
            data: Data to create prior from
            kernel_width: Width of kernel for density estimation
            name: Name of the prior
        """
        # This is essentially a KDE prior with a different name
        return KDEPrior(samples=data, bandwidth=kernel_width, name=name)


class MultivariatePrior(BasePrior):
    """Multivariate prior for correlated parameters."""

    def __init__(self, mu, sigma, name=None):
        """Initialize a multivariate normal prior.

        Args:
            mu: Mean vector
            sigma: Covariance matrix
            name: Name of the prior
        """
        import zfit

        # Convert to numpy arrays
        mu = np.array(mu)
        sigma = np.array(sigma)

        # Check dimensions
        if len(mu.shape) != 1:
            msg = "mu must be a vector"
            raise ValueError(msg)
        if len(sigma.shape) != 2:
            msg = "sigma must be a matrix"
            raise ValueError(msg)
        if mu.shape[0] != sigma.shape[0] or sigma.shape[0] != sigma.shape[1]:
            msg = "Dimension mismatch between mu and sigma"
            raise ValueError(msg)

        # Currently zfit doesn't directly support multivariate distributions in the Bayesian module,
        # so we'll need to adapt this to work with the existing framework.

        # For demonstration, we'll implement a special case for 2D distributions
        if mu.shape[0] == 2:
            # For 2D, we can use a custom PDF
            obs = zfit.Space(["x", "y"], limits=[(-float("inf"), float("inf")), (-float("inf"), float("inf"))])

            # Convert to TF values
            mu_tf = tf.convert_to_tensor(mu, dtype=tf.float64)
            sigma_tf = tf.convert_to_tensor(sigma, dtype=tf.float64)
            sigma_inv = tf.linalg.inv(sigma_tf)
            sigma_det = tf.linalg.det(sigma_tf)

            # Define the log PDF of a multivariate normal
            def _log_pdf(x):
                x_minus_mu = x - mu_tf
                exponent = -0.5 * tf.reduce_sum(x_minus_mu * tf.linalg.matvec(sigma_inv, x_minus_mu), axis=1)
                norm_const = -0.5 * (2 * tf.math.log(2 * np.pi) + tf.math.log(sigma_det))
                return exponent + norm_const

            # Create custom PDF
            pdf = zfit.pdf.UnbinnedPDF(_log_pdf, obs=obs, name="MultivariateNormal")

            super().__init__(pdf=pdf, name=name)
        else:
            msg = "Multivariate priors with dimension > 2 are not yet implemented"
            raise NotImplementedError(msg)


# Utility functions for working with priors


def create_prior_from_mean_and_std(mean, std, distribution="normal", lower=None, upper=None, name=None):
    """Create a prior from mean and standard deviation.

    Args:
        mean: Prior mean
        std: Prior standard deviation
        distribution: Distribution type ("normal", "student_t", "cauchy")
        lower: Optional lower bound
        upper: Optional upper bound
        name: Name of the prior

    Returns:
        A prior instance
    """
    if distribution == "normal":
        prior = NormalPrior(mu=mean, sigma=std, name=name)
    elif distribution == "student_t":
        prior = StudentTPrior(mu=mean, sigma=std, nu=4, name=name)
    elif distribution == "cauchy":
        prior = CauchyPrior(loc=mean, scale=std, name=name)
    else:
        msg = f"Unknown distribution: {distribution}"
        raise ValueError(msg)

    # Add bounds if specified
    if lower is not None or upper is not None:
        prior = BoundedPrior(prior, lower=lower, upper=upper, name=name)

    return prior


def create_prior_for_scale_parameter(scale=1.0, distribution="half_normal", lower=0.0, name=None):
    """Create a prior appropriate for a scale parameter.

    Args:
        scale: Scale parameter
        distribution: Distribution type ("half_normal", "half_cauchy", "gamma", "inv_gamma", "log_normal")
        lower: Lower bound (default 0)
        name: Name of the prior

    Returns:
        A prior instance
    """
    if distribution == "half_normal":
        prior = HalfNormalPrior(mu=0, sigma=scale, name=name)
    elif distribution == "half_cauchy":
        prior = HalfCauchyPrior(loc=0, scale=scale, name=name)
    elif distribution == "gamma":
        # Parameterized to have mean=scale
        prior = GammaPrior(alpha=2.0, beta=2.0 / scale, name=name)
    elif distribution == "inv_gamma":
        # Parameterized to have mean=scale
        prior = InverseGammaPrior(alpha=3.0, beta=2.0 * scale, name=name)
    elif distribution == "log_normal":
        # Set mu such that median = scale
        prior = LogNormalPrior(mu=np.log(scale), sigma=0.5, name=name)
    elif distribution == "jeffreys":
        # Non-informative Jeffreys prior for scale
        prior = JeffreysPrior(lower=lower, upper=1000 * scale, name=name)
    else:
        msg = f"Unknown distribution: {distribution}"
        raise ValueError(msg)

    return prior


def create_prior_from_data(data, bandwidth=None, name=None):
    """Create an empirical prior from data.

    Args:
        data: Data to create prior from
        bandwidth: Width of kernel for density estimation
        name: Name of the prior

    Returns:
        A prior instance
    """
    return KDEPrior(samples=data, bandwidth=bandwidth, name=name)


def create_regularized_prior(strength=1.0, l1_ratio=0.0, name=None):
    """Create a prior for parameter regularization.

    Args:
        strength: Regularization strength
        l1_ratio: Ratio of L1 vs L2 regularization (0=ridge, 1=lasso)
        name: Name of the prior

    Returns:
        A prior instance
    """
    return RegularizedPrior(strength=strength, l1_ratio=l1_ratio, name=name)


def create_sparsity_inducing_prior(sparsity_weight=0.5, scale=1.0, name=None):
    """Create a prior that encourages sparsity.

    Args:
        sparsity_weight: Weight of the spike at zero
        scale: Scale of the slab component
        name: Name of the prior

    Returns:
        A prior instance
    """
    return SpikeAndSlabPrior(weight_at_zero=sparsity_weight, slab_scale=scale, name=name)


def create_weakly_informative_prior(scale=1.0, name=None):
    """Create a weakly informative prior.

    Args:
        scale: Scale parameter
        name: Name of the prior

    Returns:
        A prior instance
    """
    return WeaklyInformativePrior(scale=scale, name=name, distribution="student_t")


def create_hierarchical_normal_prior(mu, sigma, name=None):
    """Create a hierarchical normal prior.

    Args:
        mu: Mean parameter or distribution
        sigma: Standard deviation parameter or distribution
        name: Name of the prior

    Returns:
        A prior instance
    """
    import zfit

    obs = zfit.Space("prior_space", limits=(-float("inf"), float("inf")))

    # Create a Gaussian PDF
    pdf = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create a hierarchical prior
    hyperparams = {"mu": mu, "sigma": sigma}
    return HierarchicalPrior(pdf, hyperparams, name=name)
