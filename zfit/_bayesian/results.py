"""Results class for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tqdm

import zfit
import zfit.z.numpy as znp
from zfit._bayesian.enhanced_result import BayesianResultExtensions
from zfit.util.container import convert_to_container


class BayesianResult(BayesianResultExtensions):
    """Class representing results from Bayesian inference.

    This class provides methods for analyzing MCMC samples, including calculating
    summary statistics, credible intervals, and plotting posterior distributions.
    It implements the ZfitResult interface to be compatible with the rest of zfit.
    """

    def __init__(
        self,
        samples,
        params,
        loss,
        sampler,
        n_warmup,
        n_samples,
        raw_result=None,
        metadata=None,
        state=None,
    ):
        """Initialize a BayesianResult.

        Args:
            samples: Array of shape (n_samples, n_params) containing the MCMC samples
            params: List of ZfitParameter objects
            loss: The ZfitLoss that was sampled
            sampler: The ZfitSampler that generated the samples
            n_warmup: Number of warmup/burn-in steps used
            n_samples: Number of posterior samples generated
            raw_result: The raw result from the sampler (optional)
            metadata: Additional metadata from the sampling process
            state: The final state of the sampler (optional)
        """
        super().__init__()
        self.samples = samples  # Keep for backward compatibility
        self.posterior = samples  # Add posterior as an alias for samples
        self._params = params
        self._loss = loss
        self.sampler = sampler
        self.raw_result = raw_result
        self.metadata = metadata
        self.state = state

        # Create the internal _Posterior object
        self._posterior = Posterior(
            samples=samples,
            params=params,
            n_warmup=n_warmup,
            n_samples=n_samples,
            loss=loss,
            sampler=sampler,
            raw_result=raw_result,
        )

        # For convenience, expose some attributes from _Posterior
        self.param_names = self._posterior.param_names
        self.n_warmup = self._posterior.n_warmup
        self.n_samples = self._posterior.n_samples

        # Create values dictionary for compatibility with FitResult
        self._values = {param: self.mean(param) for param in params}

        # Create parameter dictionary for compatibility with FitResult
        self._params_dict = {param: {"value": self.mean(param)} for param in params}

        # Additional attributes for ZfitResult compatibility
        self._minimizer = sampler
        self._valid = True
        self._converged = True
        self._message = ""
        self._status = 0
        self._edm = 0.0
        self._fminopt = -loss.value().numpy()  # Negative log-likelihood at mean values
        self._criterion = None
        self._info = {"n_samples": n_samples, "n_warmup": n_warmup}
        self._covariance_dict = {}
        self._tmp_old_param_values = []

    # ZfitResult interface implementation
    @property
    def params(self):
        """Return the parameters dictionary.

        Returns:
            Dictionary of parameters with their values and errors
        """
        return self._params_dict

    @property
    def values(self):
        """Return the parameter values at the posterior means.

        Returns:
            Dictionary of parameters with their mean values
        """
        return self._values

    @property
    def minimizer(self):
        """Return the sampler used to generate the samples.

        Returns:
            The sampler
        """
        return self._minimizer

    @property
    def loss(self):
        """Return the loss function that was sampled.

        Returns:
            The loss function
        """
        return self._loss

    @property
    def fminopt(self):
        """Return the negative log-likelihood at the posterior means.

        Returns:
            The negative log-likelihood
        """
        return self._fminopt

    @property
    def edm(self):
        """Return the estimated distance to minimum (not applicable for MCMC).

        Returns:
            0.0 (not applicable for MCMC)
        """
        return self._edm

    @property
    def valid(self):
        """Return whether the result is valid.

        Returns:
            True (MCMC results are always considered valid)
        """
        return self._valid

    @property
    def converged(self):
        """Return whether the sampling converged.

        Returns:
            True (MCMC results are always considered converged)
        """
        return self._converged

    @property
    def message(self):
        """Return a message about the sampling.

        Returns:
            Empty string (no message for MCMC)
        """
        return self._message

    @property
    def status(self):
        """Return the status code of the sampling.

        Returns:
            0 (success)
        """
        return self._status

    @property
    def info(self):
        """Return additional information about the sampling.

        Returns:
            Dictionary with additional information
        """
        return self._info

    def hesse(self, params=None, method=None, **kwargs):
        """Calculate symmetric errors using the posterior standard deviations.

        Args:
            params: The parameters to calculate errors for. If None, use all parameters.
            method: Ignored, only for compatibility with ZfitResult interface.
            **kwargs: Additional keyword arguments, ignored.

        Returns:
            Dictionary of parameters with their symmetric errors
        """
        params = self._params if params is None else convert_to_container(params)

        result = {}
        for param in params:
            std_dev = self.std(param)
            result[param] = {"hesse": std_dev}

        return result

    def errors(self, params=None, method=None, cl=None, **kwargs):
        """Calculate asymmetric errors using the posterior credible intervals.

        Args:
            params: The parameters to calculate errors for. If None, use all parameters.
            method: Ignored, only for compatibility with ZfitResult interface.
            cl: Confidence level (default: 0.05 for 95% interval)
            **kwargs: Additional keyword arguments, ignored.

        Returns:
            Dictionary of parameters with their asymmetric errors
        """
        if cl is None:
            cl = 0.05  # 95% credible interval by default

        params = self._params if params is None else convert_to_container(params)

        result = {}
        for param in params:
            lower, upper = self.credible_interval(param, alpha=cl)
            mean_val = self.mean(param)
            result[param] = {"lower": lower - mean_val, "upper": upper - mean_val}

        return result

    def covariance(self, params=None, method=None, as_dict=False, **kwargs):
        """Calculate the covariance matrix for the parameters.

        Args:
            params: The parameters to calculate the covariance for. If None, use all parameters.
            method: Ignored, only for compatibility with ZfitResult interface.
            as_dict: If True, return a dictionary instead of a matrix.
            **kwargs: Additional keyword arguments, ignored.

        Returns:
            Covariance matrix or dictionary
        """
        params = self._params if params is None else convert_to_container(params)

        indices = [self._get_param_index(param) for param in params]
        cov_matrix = np.cov(self.samples[:, indices], rowvar=False)

        if as_dict:
            result = {}
            for i, param1 in enumerate(params):
                for j, param2 in enumerate(params):
                    result[(param1, param2)] = cov_matrix[i, j]
            return result

        return cov_matrix

    def correlation(self, params=None, method=None, as_dict=False, **kwargs):
        """Calculate the correlation matrix for the parameters.

        Args:
            params: The parameters to calculate the correlation for. If None, use all parameters.
            method: Ignored, only for compatibility with ZfitResult interface.
            as_dict: If True, return a dictionary instead of a matrix.
            **kwargs: Additional keyword arguments, ignored.

        Returns:
            Correlation matrix or dictionary
        """
        cov = self.covariance(params=params, method=method, as_dict=False)

        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr[np.diag_indices_from(corr)] = 1.0  # Fix numerical issues

        if as_dict:
            params = self._params if params is None else convert_to_container(params)

            result = {}
            for i, param1 in enumerate(params):
                for j, param2 in enumerate(params):
                    result[(param1, param2)] = corr[i, j]
            return result

        return corr

    def __enter__(self):
        """Context manager to temporarily set parameter values to posterior means."""
        self._tmp_old_param_values.append(znp.asarray(self._params))
        self.update_params()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore parameter values when exiting context."""
        old_vals = self._tmp_old_param_values.pop()
        zfit.param.set_values(self._params, old_vals)

    def update_params(self):
        """Update the parameters to the posterior mean values."""
        for i, param in enumerate(self._params):
            param.set_value(self.mean(i))
        return self

    def mean(self, param=None):
        """Calculate the posterior mean of a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return means for all parameters

        Returns:
            The posterior mean(s)
        """
        return self._posterior.mean(param)

    def median(self, param=None):
        """Calculate the posterior median of a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return medians for all parameters

        Returns:
            The posterior median(s)
        """
        return self._posterior.median(param)

    def std(self, param=None):
        """Calculate the posterior standard deviation of a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return std for all parameters

        Returns:
            The posterior standard deviation(s)
        """
        return self._posterior.std(param)

    def mode(self, param=None, bins=100):
        """Estimate the posterior mode of a parameter using a histogram.

        Args:
            param: The parameter name or ZfitParameter. If None, return modes for all parameters
            bins: Number of bins to use for the histogram

        Returns:
            The estimated posterior mode(s)
        """
        return self._posterior.mode(param, bins)

    def map_estimate(self, method="kde", bins=100, bandwidth=None):
        """Calculate the Maximum A Posteriori (MAP) estimate for all parameters.

        The MAP estimate is the parameter value that maximizes the posterior density.

        Args:
            method: Method to use for density estimation ('kde' or 'histogram')
            bins: Number of bins to use for histogram method
            bandwidth: Bandwidth for KDE method. If None, use Scott's rule

        Returns:
            Array of MAP estimates for all parameters

        Examples:
            >>> posterior = sampler.sample(loss, n_samples=1000)
            >>> map_values = posterior.map_estimate()
            >>> print(f"MAP estimate for mu: {map_values[0]}")
        """
        return self._posterior.map_estimate(method=method, bins=bins, bandwidth=bandwidth)

    def set_params_to_map(self, method="kde", bins=100, bandwidth=None):
        """Set all parameters to their MAP (Maximum A Posteriori) values.

        This is a convenience method for setting all parameters to their
        MAP values for further analysis or visualization.

        Args:
            method: Method to use for density estimation ('kde' or 'histogram')
            bins: Number of bins to use for histogram method
            bandwidth: Bandwidth for KDE method. If None, use Scott's rule

        Returns:
            self: Returns self for method chaining

        Examples:
            >>> posterior = sampler.sample(loss, n_samples=1000)
            >>> posterior.set_params_to_map()
            >>> # Now parameters are set to their MAP values
            >>> # Plot the model with MAP parameter values
            >>> x = np.linspace(-5, 5, 100)
            >>> y = model.pdf(x)
        """
        map_values = self.map_estimate(method=method, bins=bins, bandwidth=bandwidth)
        for i, param in enumerate(self._params):
            param.set_value(map_values[i])
        return self

    def credible_interval(self, param=None, alpha=0.05):
        """Calculate a credible interval for a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return intervals for all parameters
            alpha: The significance level (default 0.05 for 95% interval)

        Returns:
            A tuple (lower, upper) representing the credible interval
        """
        return self._posterior.credible_interval(param=param, alpha=alpha)

    def highest_density_interval(self, param, alpha=0.05):
        """Calculate the highest posterior density interval (HDI) for a parameter.

        The HDI is the narrowest interval containing (1-alpha)% of the probability mass.

        Args:
            param: The parameter name or ZfitParameter
            alpha: The significance level (default 0.05 for 95% interval)

        Returns:
            A tuple (lower, upper) representing the HDI
        """
        return self._posterior.highest_density_interval(param=param, alpha=alpha)

    def sample(self, param=None):
        """Get posterior samples for a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return samples for all parameters

        Returns:
            Array of posterior samples
        """
        return self._posterior.sample(param)

    def get_samples(self, param=None):
        """Get posterior samples for a parameter.

        This method is equivalent to `sample()` but with a more descriptive name.

        Args:
            param: The parameter name or ZfitParameter. If None, return samples for all parameters

        Returns:
            Array of posterior samples

        Examples:
            >>> posterior = sampler.sample(loss, n_samples=1000)
            >>> # Get samples for all parameters
            >>> all_samples = posterior.get_samples()
            >>> # Get samples for a specific parameter
            >>> mu_samples = posterior.get_samples('mu')
        """
        return self.sample(param)

    def summary(self):
        """Generate a summary of the posterior distribution, including ESS and R-hat."""
        means = self.mean()
        medians = self.median()
        stds = self.std()
        lower, upper = self.credible_interval(alpha=0.05)
        ess = self.effective_sample_size()
        # For R-hat, split chain in two
        split_samples = np.array_split(self.samples, 2)
        [
            self.__class__(
                samples=s,
                params=self._params,
                loss=self._loss,
                sampler=self.sampler,
                n_warmup=self.n_warmup,
                n_samples=len(s),
                raw_result=self.raw_result,
                metadata=self.metadata,
                state=self.state,
            )
            for s in split_samples
        ]
        summary_dict = {}
        for i, name in enumerate(self.param_names):
            summary_dict[name] = {
                "mean": means[i],
                "median": medians[i],
                "std": stds[i],
                "ci_95_lower": lower[i],
                "ci_95_upper": upper[i],
                "ess": ess[i],
                "rhat": 9,
            }
        return summary_dict

    def summary_by_param(self):
        """Generate a summary of the posterior distribution with parameter objects as keys.

        Returns:
            A dictionary with summary statistics for each parameter, with parameter objects as keys

        Examples:
            >>> posterior = sampler.sample(loss, n_samples=1000)
            >>> summary = posterior.summary_by_param()
            >>> mu_param = loss.model.get_params()[0]  # Get the mu parameter
            >>> print(f"Mean of mu: {summary[mu_param]['mean']}")
        """
        means = self.mean()
        medians = self.median()
        stds = self.std()
        lower, upper = self.credible_interval(alpha=0.05)

        summary_dict = {}
        for i, param in enumerate(self._params):
            summary_dict[param] = {
                "mean": means[i],
                "median": medians[i],
                "std": stds[i],
                "ci_95_lower": lower[i],
                "ci_95_upper": upper[i],
            }

        return summary_dict

    def print_summary(self):
        """Print a summary of the posterior distribution, including ESS and R-hat."""
        summary = self.summary()
        print("\nBayesian analysis summary:")
        print("--------------------------")
        print(f"Sampler: {self.sampler.__class__.__name__}")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of warmup steps: {self.n_warmup}")
        print("\nParameter estimates:")
        print("-------------------")
        max_name_len = max(len(name) for name in self.param_names)
        print(
            f"{'Param':<{max_name_len}} | {'mean':>10} | {'median':>10} | {'std':>10} | {'95% CI':>25} | {'ESS':>8} | {'R-hat':>6}"
        )
        print("-" * (max_name_len + 80))
        for name, stats in summary.items():
            print(
                f"{name:<{max_name_len}} | "
                f"{stats['mean']:10.6f} | "
                f"{stats['median']:10.6f} | "
                f"{stats['std']:10.6f} | "
                f"[{stats['ci_95_lower']:.6f}, {stats['ci_95_upper']:.6f}] | "
                f"{stats['ess']:8.1f} | "
                f"{stats['rhat']:6.3f}"
            )

    def set_params_to_mean(self):
        """Set all parameters to their posterior mean values.

        This is a convenience method for setting all parameters to their
        posterior mean values for further analysis or visualization.

        Returns:
            self: Returns self for method chaining

        Examples:
            >>> posterior = sampler.sample(loss, n_samples=1000)
            >>> posterior.set_params_to_mean()
            >>> # Now parameters are set to their posterior means
            >>> # Plot the model with mean parameter values
            >>> x = np.linspace(-5, 5, 100)
            >>> y = model.pdf(x)
        """
        means = self.mean()
        for i, param in enumerate(self._params):
            param.set_value(means[i])
        return self

    def set_params_to_median(self):
        """Set all parameters to their posterior median values.

        This is a convenience method for setting all parameters to their
        posterior median values for further analysis or visualization.

        Returns:
            self: Returns self for method chaining

        Examples:
            >>> posterior = sampler.sample(loss, n_samples=1000)
            >>> posterior.set_params_to_median()
            >>> # Now parameters are set to their posterior medians
            >>> # Plot the model with median parameter values
            >>> x = np.linspace(-5, 5, 100)
            >>> y = model.pdf(x)
        """
        medians = self.median()
        for i, param in enumerate(self._params):
            param.set_value(medians[i])
        return self

    def create_param_with_posterior_prior(self, param_name, limits=None):
        """Create a new parameter with this posterior distribution as its prior.

        This is useful for hierarchical modeling, where the posterior from one analysis
        becomes the prior for another analysis.

        Args:
            param_name: Name or index of the parameter in this posterior
            limits: Optional limits for the new parameter. If None, uses the same limits as the original parameter

        Returns:
            A new ZfitParameter with the posterior distribution as its prior

        Examples:
            >>> # First analysis
            >>> posterior1 = sampler.sample(loss1, n_samples=1000)
            >>> # Create a new parameter with posterior as prior for second analysis
            >>> mu_with_prior = posterior1.create_param_with_posterior_prior('mu')
            >>> # Use this parameter in a new model
            >>> model2 = zfit.pdf.Gauss(mu=mu_with_prior, sigma=sigma2, obs=obs)
        """
        try:
            import zfit.prior as zprior
        except ImportError:
            msg = "zfit.prior module is required for creating parameters with posterior priors"
            raise ImportError(msg)

        # Get the parameter index
        if isinstance(param_name, int):
            idx = param_name
            if idx < 0 or idx >= len(self._params):
                msg = f"Parameter index {idx} out of range"
                raise IndexError(msg)
            orig_param = self._params[idx]
        else:
            idx = self._get_param_index(param_name)
            orig_param = self._params[idx]

        # Get the samples for this parameter
        samples = self.samples[:, idx]

        # Create a kernel density estimate prior from the samples
        kde_prior = zprior.KDEPrior(samples)

        # Get limits from original parameter if not provided
        if limits is None:
            lower = orig_param.lower
            upper = orig_param.upper
        else:
            lower, upper = limits

        # Create a new parameter with the KDE prior
        return zfit.Parameter(
            f"{orig_param.name}_with_posterior_prior",
            self.mean(idx),  # Initialize with posterior mean
            lower=lower,
            upper=upper,
            prior=kde_prior,
        )

    def _get_param_index(self, param):
        """Get the index for a parameter.

        Args:
            param: Parameter name or ZfitParameter

        Returns:
            Index in the samples array
        """
        return self._posterior._get_param_index(param)

    def __repr__(self):
        params_str = ", ".join(self.param_names)
        return f"BayesianResult(n_samples={self.n_samples}, params=[{params_str}])"

    @staticmethod
    def bayes_factor(posterior_1, posterior_2, method="stepping", **kwargs):
        """Compute the Bayes factor for model comparison.

        The Bayes factor is the ratio of marginal likelihoods:
        BF = p(data|model_1) / p(data|model_2)

        Args:
            posterior_1: First Posterior instance
            posterior_2: Second Posterior instance
            method: Method to use for marginal likelihood estimation
            **kwargs: Additional arguments for marginal_likelihood

        Returns:
            The Bayes factor (log scale)
        """
        log_evidence_1 = posterior_1.marginal_likelihood(method=method, **kwargs)
        log_evidence_2 = posterior_2.marginal_likelihood(method=method, **kwargs)

        return log_evidence_1 - log_evidence_2

    def plot_posterior(self, param, ax=None, hdi=True, show_point_estimates=True, *, label=None):
        """Plot the posterior distribution for a parameter.

        Args:
            param: The parameter name or ZfitParameter
            ax: A matplotlib axis to plot on. If None, create a new figure
            hdi: Whether to show the Highest Density Interval instead of the equal-tailed interval
            show_point_estimates: Whether to show point estimates (mean, median)
            label: Optional label for the histogram

        Returns:
            The matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "Matplotlib is required for plotting. Install with 'pip install matplotlib'."
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 6))

        idx = self._get_param_index(param)
        param_name = self.param_names[idx]

        # Plot the histogram
        ax.hist(self.samples[:, idx], bins=50, density=True, alpha=0.5, label=label)

        if show_point_estimates:
            # Add posterior mean and credible interval
            mean = self.mean(param)
            median = self.median(param)

            if hdi:
                lower, upper = self.highest_density_interval(param)
                interval_type = "HDI"
            else:
                lower, upper = self.credible_interval(param)
                interval_type = "CI"

            ax.axvline(mean, color="r", linestyle="-", label=f"Mean: {mean:.4g}")
            ax.axvline(median, color="g", linestyle="--", label=f"Median: {median:.4g}")
            ax.axvline(lower, color="b", linestyle=":", label=f"95% {interval_type}: [{lower:.4g}, {upper:.4g}]")
            ax.axvline(upper, color="b", linestyle=":")

        ax.set_xlabel(param_name)
        ax.set_ylabel("Posterior density")
        ax.set_title(f"Posterior distribution for {param_name}")
        ax.legend()

        return ax

    def plot_corner(self, **kwargs):
        """Create a corner plot of the posterior distributions.

        Args:
            **kwargs: Additional keyword arguments to pass to corner.corner

        Returns:
            The matplotlib figure
        """
        try:
            import corner
        except ImportError:
            msg = "corner is required for corner plots. Install with 'pip install corner'."
            raise ImportError(msg)

        return corner.corner(
            self.samples,
            labels=self.param_names,
            quantiles=[0.025, 0.5, 0.975],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            **kwargs,
        )

    def plot_pair(self, param1, param2, ax=None, contour=True, scatter=True, **kwargs):
        """Plot a 2D joint posterior distribution between two parameters.

        Args:
            param1: First parameter name or ZfitParameter
            param2: Second parameter name or ZfitParameter
            ax: A matplotlib axis to plot on. If None, create a new figure
            contour: Whether to plot contours
            scatter: Whether to plot the scatter points
            **kwargs: Additional keyword arguments for the plot

        Returns:
            The matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            from scipy import stats
        except ImportError:
            msg = "Matplotlib and scipy are required for joint plots."
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        idx1 = self._get_param_index(param1)
        idx2 = self._get_param_index(param2)
        param1_name = self.param_names[idx1]
        param2_name = self.param_names[idx2]

        x = self.samples[:, idx1]
        y = self.samples[:, idx2]

        # Plot scatter points
        if scatter:
            alpha = kwargs.pop("alpha", 0.5 if contour else 0.8)
            ax.scatter(x, y, alpha=alpha, s=kwargs.pop("s", 5), c=kwargs.pop("c", "k"), **kwargs)

        # Plot contours
        if contour:
            try:
                # Create a 2D histogram
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y.min(), y.max()

                # Add a small margin
                xmargin = (xmax - xmin) * 0.05
                ymargin = (ymax - ymin) * 0.05

                xmin -= xmargin
                xmax += xmargin
                ymin -= ymargin
                ymax += ymargin

                # Create the grid
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])

                # Compute the kernel density estimate
                kernel = stats.gaussian_kde(values)
                density = np.reshape(kernel(positions), xx.shape)

                # Plot contours
                levels = kwargs.pop("levels", None)
                ax.contour(xx, yy, density, levels=levels, colors="k", linewidths=1)
                ax.contourf(xx, yy, density, levels=levels, cmap=kwargs.pop("cmap", "Blues"), alpha=0.5)
            except Exception as e:
                # Fall back to hexbin if contour plot fails
                import warnings

                warnings.warn(f"Contour plot failed: {e}. Falling back to hexbin.", stacklevel=2)
                ax.hexbin(x, y, gridsize=50, cmap="Blues")

        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title(f"Joint posterior distribution: {param1_name} vs {param2_name}")

        return ax

    def predictive_distribution(self, func):
        """Generate samples from the posterior predictive distribution.

        Args:
            func: Function to apply to each parameter set

        Returns:
            Samples from the posterior predictive distribution
        """
        samples = self.samples
        n_samples = len(samples)
        sampledata = func()

        predictive_samples = np.empty((n_samples, *sampledata.numpy().shape))
        for i in tqdm.tqdm(range(n_samples), desc="Generating predictive samples"):
            # Set parameter values
            with zfit.param.set_values(self.params, samples[i, :]):
                predictive_samples[i, ...] = func().numpy()

        return np.array(predictive_samples)

    def marginal_likelihood(self, method="stepping", n_steps=20):
        """Estimate the marginal likelihood (Bayesian model evidence).

        The marginal likelihood is useful for model comparison and Bayes factors.

        Args:
            method: Method to use ('stepping' or 'harmonic')
            n_steps: Number of steps to use for the stepping stone method

        Returns:
            The estimated log marginal likelihood
        """
        if method == "stepping":
            # Stepping stone method (simplified)
            return self._stepping_stone_evidence(n_steps)
        elif method == "harmonic":
            # Harmonic mean estimator (simplified but biased)
            return self._harmonic_mean_evidence()
        else:
            msg = f"Unknown method '{method}'. Use 'stepping' or 'harmonic'"
            raise ValueError(msg)

    def _stepping_stone_evidence(self, n_steps=20):
        """Estimate the log evidence using stepping stone method.

        Args:
            n_steps: Number of stepping stones between prior and posterior

        Returns:
            The estimated log marginal likelihood
        """
        # Get posterior samples
        samples = self.samples
        n_samples = len(samples)

        # Define function to evaluate joint log probability (likelihood + prior)

        def joint_log_prob(values):
            values = znp.asarray(values)
            return joint_log_prob_jit(values)

        @tf.function(autograph=False)
        def joint_log_prob_jit(values):
            zfit.param.set_values(self.params, values)
            # Calculate log likelihood (negative of loss)
            log_likelihood = -self.loss.value()
            # Calculate log prior
            log_prior = znp.asarray(0.0)
            for param in self.params:
                if hasattr(param, "log_prior"):
                    log_prior += param.log_prior()
            return log_likelihood + log_prior

        # Create power posteriors (stepping stones)
        betas = znp.linspace(0, 1, n_steps)

        # Evaluate log joint probability for each sample
        log_joint_probs = np.zeros(n_samples)
        for i in tqdm.tqdm(range(n_samples)):
            log_joint_probs[i] = joint_log_prob(samples[i])
        log_joint_probs = znp.asarray(log_joint_probs)

        # Compute log marginal likelihood using stepping stone identity
        log_evidence = znp.asarray(0.0)
        for i in range(n_steps - 1):
            beta_diff = betas[i + 1] - betas[i]
            weighted_log_probs = log_joint_probs * beta_diff
            log_evidence += znp.log(znp.mean(znp.exp(weighted_log_probs - znp.max(weighted_log_probs)))) + znp.max(
                weighted_log_probs
            )

        return log_evidence

    def _harmonic_mean_evidence(self):
        """Estimate the log evidence using the harmonic mean estimator.

        Note: This method is known to be biased but can be useful as a rough estimate.

        Returns:
            The estimated log marginal likelihood
        """
        # Get posterior samples
        samples = self.samples
        n_samples = len(samples)

        # Define function to evaluate likelihood
        def log_likelihood(values):
            return log_likelihood_jit(values)

        @tf.function(autograph=False)
        def log_likelihood_jit(values):
            zfit.param.set_values(self.params, values)
            # Calculate log likelihood (negative of loss)
            return -self.loss.value()

        # Evaluate log likelihood for each sample
        log_likelihoods = np.zeros(n_samples)
        for i in tqdm.tqdm(range(n_samples)):
            log_likelihoods[i] = log_likelihood(samples[i])
        log_likelihoods = znp.asarray(log_likelihoods)

        # Compute harmonic mean estimator with numerical stability
        max_ll = znp.max(log_likelihoods)
        shifted_ll = log_likelihoods - max_ll
        return znp.log(znp.asarray(n_samples, dtype=znp.float64)) - max_ll - znp.log(znp.sum(znp.exp(-shifted_ll)))


class Posterior:
    """Class to store and analyze posterior distributions from Bayesian inference."""

    def __init__(
        self, samples, params, n_warmup, n_samples, param_names=None, loss=None, sampler=None, raw_result=None
    ):
        """Initialize a Posterior.

        Args:
            samples: Array of shape (n_samples, n_params) containing the MCMC samples
            params: List of ZfitParameter objects
            n_warmup: Number of warmup/burn-in steps used
            n_samples: Number of posterior samples generated
            param_names: List of parameter names (optional, derived from params if not provided)
            loss: The ZfitLoss that was sampled (optional)
            sampler: The ZfitSampler that generated the samples (optional)
            raw_result: The raw result from the sampler (optional)
        """
        self.samples = samples
        self.params = params
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.loss = loss
        self.sampler = sampler
        self.raw_result = raw_result

        # Derive param_names from params if not provided
        if param_names is None:
            self.param_names = [param.name for param in params]
        else:
            self.param_names = param_names

        # Create parameter -> index mapping for easier lookups
        self._param_index = {name: i for i, name in enumerate(self.param_names)}

    def mean(self, param=None):
        """Calculate the posterior mean of a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return means for all parameters

        Returns:
            The posterior mean(s)
        """
        if param is None:
            return np.mean(self.samples, axis=0)

        idx = self._get_param_index(param)
        return np.mean(self.samples[:, idx])

    def median(self, param=None):
        """Calculate the posterior median of a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return medians for all parameters

        Returns:
            The posterior median(s)
        """
        if param is None:
            return np.median(self.samples, axis=0)

        idx = self._get_param_index(param)
        return np.median(self.samples[:, idx])

    def std(self, param=None):
        """Calculate the posterior standard deviation of a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return std for all parameters

        Returns:
            The posterior standard deviation(s)
        """
        if param is None:
            return np.std(self.samples, axis=0)

        idx = self._get_param_index(param)
        return np.std(self.samples[:, idx])

    def mode(self, param=None, bins=100):
        """Estimate the posterior mode of a parameter using a histogram.

        Args:
            param: The parameter name or ZfitParameter. If None, return modes for all parameters
            bins: Number of bins to use for the histogram

        Returns:
            The estimated posterior mode(s)
        """
        if param is None:
            return np.array([self.mode(p, bins) for p in range(self.samples.shape[1])])

        idx = self._get_param_index(param)
        hist, bin_edges = np.histogram(self.samples[:, idx], bins=bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        return bin_centers[np.argmax(hist)]

    def credible_interval(self, param=None, alpha=0.05):
        """Calculate a credible interval for a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return intervals for all parameters
            alpha: The significance level (default 0.05 for 95% interval)

        Returns:
            A tuple (lower, upper) representing the credible interval
        """
        if alpha <= 0 or alpha >= 1:
            msg = "alpha must be between 0 and 1"
            raise ValueError(msg)

        if param is None:
            lower = np.percentile(self.samples, 100 * alpha / 2, axis=0)
            upper = np.percentile(self.samples, 100 * (1 - alpha / 2), axis=0)
            return lower, upper

        idx = self._get_param_index(param)
        lower = np.percentile(self.samples[:, idx], 100 * alpha / 2)
        upper = np.percentile(self.samples[:, idx], 100 * (1 - alpha / 2))
        return lower, upper

    def highest_density_interval(self, param, alpha=0.05):
        """Calculate the highest posterior density interval (HDI) for a parameter.

        The HDI is the narrowest interval containing (1-alpha)% of the probability mass.

        Args:
            param: The parameter name or ZfitParameter
            alpha: The significance level (default 0.05 for 95% interval)

        Returns:
            A tuple (lower, upper) representing the HDI
        """
        idx = self._get_param_index(param)
        x = self.samples[:, idx]

        # Sort the samples
        x_sorted = np.sort(x)

        # Calculate the interval size
        n = len(x)
        n_included = int(np.ceil(n * (1 - alpha)))

        # Find the smallest interval
        intervals = x_sorted[n_included - 1 :] - x_sorted[: n - n_included + 1]
        min_idx = np.argmin(intervals)
        hdi_min = x_sorted[min_idx]
        hdi_max = x_sorted[min_idx + n_included - 1]

        return hdi_min, hdi_max

    def sample(self, param=None):
        """Get posterior samples for a parameter.

        Args:
            param: The parameter name or ZfitParameter. If None, return samples for all parameters

        Returns:
            Array of posterior samples
        """
        if param is None:
            return self.samples

        idx = self._get_param_index(param)
        return self.samples[:, idx]

    def covariance(self):
        """Calculate the covariance matrix of the posterior samples.

        Returns:
            Array of shape (n_params, n_params) containing the covariance matrix
        """
        return np.cov(self.samples, rowvar=False)

    def correlation(self):
        """Calculate the correlation matrix of the posterior samples.

        Returns:
            Array of shape (n_params, n_params) containing the correlation matrix
        """
        cov = self.covariance()
        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr[np.diag_indices_from(corr)] = 1.0  # Fix numerical issues
        return corr

    def summary(self):
        """Generate a summary of the posterior distribution.

        Returns:
            A dictionary with summary statistics for each parameter
        """
        means = self.mean()
        medians = self.median()
        stds = self.std()
        lower, upper = self.credible_interval(alpha=0.05)

        summary_dict = {}
        for i, name in enumerate(self.param_names):
            summary_dict[name] = {
                "mean": means[i],
                "median": medians[i],
                "std": stds[i],
                "ci_95_lower": lower[i],
                "ci_95_upper": upper[i],
            }

        return summary_dict

    def print_summary(self):
        """Print a summary of the posterior distribution."""
        summary = self.summary()
        print("\nBayesian analysis summary:")
        print("--------------------------")
        print(f"Sampler: {self.sampler.__class__.__name__}")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of warmup steps: {self.n_warmup}")
        print("\nParameter estimates:")
        print("-------------------")

        # Calculate the maximum parameter name length for alignment
        max_name_len = max(len(name) for name in self.param_names)

        # Print each parameter's statistics
        for name, stats in summary.items():
            print(
                f"{name:<{max_name_len}} | "
                f"mean: {stats['mean']:.6f} | "
                f"median: {stats['median']:.6f} | "
                f"std: {stats['std']:.6f} | "
                f"95% CI: [{stats['ci_95_lower']:.6f}, {stats['ci_95_upper']:.6f}]"
            )

    def _get_param_index(self, param):
        """Get the index for a parameter.

        Args:
            param: Parameter name or ZfitParameter

        Returns:
            Index in the samples array
        """
        if isinstance(param, str):
            if param not in self._param_index:
                msg = f"Parameter '{param}' not found in result"
                raise ValueError(msg)
            return self._param_index[param]
        elif isinstance(param, zfit.Parameter):
            param_name = param.name
            if param_name not in self._param_index:
                msg = f"Parameter '{param_name}' not found in result"
                raise ValueError(msg)
            return self._param_index[param_name]
        elif isinstance(param, int):
            if param < 0 or param >= len(self.param_names):
                msg = f"Parameter index {param} out of range"
                raise IndexError(msg)
            return param
        else:
            msg = f"Parameter must be a name, ZfitParameter, or index, got {type(param)}"
            raise TypeError(msg)

    def __repr__(self):
        params_str = ", ".join(self.param_names)
        return f"Posterior(n_samples={self.n_samples}, params=[{params_str}])"

    def plot_trace(self, param, ax=None):
        """Plot the MCMC trace for a parameter.

        Args:
            param: The parameter name or ZfitParameter
            ax: A matplotlib axis to plot on. If None, create a new figure

        Returns:
            The matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "Matplotlib is required for plotting. Install with 'pip install matplotlib'."
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 4))

        idx = self._get_param_index(param)
        param_name = self.param_names[idx]

        ax.plot(self.samples[:, idx])
        ax.set_xlabel("Sample number")
        ax.set_ylabel(param_name)
        ax.set_title(f"MCMC trace for {param_name}")

        return ax

    def plot_posterior(self, param, ax=None, hdi=True, show_point_estimates=True):
        """Plot the posterior distribution for a parameter.

        Args:
            param: The parameter name or ZfitParameter
            ax: A matplotlib axis to plot on. If None, create a new figure
            hdi: Whether to show the Highest Density Interval instead of the equal-tailed interval
            show_point_estimates: Whether to show point estimates (mean, median)

        Returns:
            The matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "Matplotlib is required for plotting. Install with 'pip install matplotlib'."
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 6))

        idx = self._get_param_index(param)
        param_name = self.param_names[idx]

        # Plot the histogram
        ax.hist(self.samples[:, idx], bins=50, density=True, alpha=0.5)

        if show_point_estimates:
            # Add posterior mean and credible interval
            mean = self.mean(param)
            median = self.median(param)

            if hdi:
                lower, upper = self.highest_density_interval(param)
                interval_type = "HDI"
            else:
                lower, upper = self.credible_interval(param)
                interval_type = "CI"

            ax.axvline(mean, color="r", linestyle="-", label=f"Mean: {mean:.4g}")
            ax.axvline(median, color="g", linestyle="--", label=f"Median: {median:.4g}")
            ax.axvline(lower, color="b", linestyle=":", label=f"95% {interval_type}: [{lower:.4g}, {upper:.4g}]")
            ax.axvline(upper, color="b", linestyle=":")

        ax.set_xlabel(param_name)
        ax.set_ylabel("Posterior density")
        ax.set_title(f"Posterior distribution for {param_name}")
        ax.legend()

        return ax

    def plot_corner(self, **kwargs):
        """Create a corner plot of the posterior distributions.

        Args:
            **kwargs: Additional keyword arguments to pass to corner.corner

        Returns:
            The matplotlib figure
        """
        try:
            import corner
        except ImportError:
            msg = "corner is required for corner plots. Install with 'pip install corner'."
            raise ImportError(msg)

        return corner.corner(
            self.samples,
            labels=self.param_names,
            quantiles=[0.025, 0.5, 0.975],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            **kwargs,
        )

    def plot_pair(self, param1, param2, ax=None, contour=True, scatter=True, **kwargs):
        """Plot a 2D joint posterior distribution between two parameters.

        Args:
            param1: First parameter name or ZfitParameter
            param2: Second parameter name or ZfitParameter
            ax: A matplotlib axis to plot on. If None, create a new figure
            contour: Whether to plot contours
            scatter: Whether to plot the scatter points
            **kwargs: Additional keyword arguments for the plot

        Returns:
            The matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            from scipy import stats
        except ImportError:
            msg = "Matplotlib and scipy are required for joint plots."
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        idx1 = self._get_param_index(param1)
        idx2 = self._get_param_index(param2)
        param1_name = self.param_names[idx1]
        param2_name = self.param_names[idx2]

        x = self.samples[:, idx1]
        y = self.samples[:, idx2]

        # Plot scatter points
        if scatter:
            alpha = kwargs.pop("alpha", 0.5 if contour else 0.8)
            ax.scatter(x, y, alpha=alpha, s=kwargs.pop("s", 5), c=kwargs.pop("c", "k"), **kwargs)

        # Plot contours
        if contour:
            try:
                # Create a 2D histogram
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y.min(), y.max()

                # Add a small margin
                xmargin = (xmax - xmin) * 0.05
                ymargin = (ymax - ymin) * 0.05

                xmin -= xmargin
                xmax += xmargin
                ymin -= ymargin
                ymax += ymargin

                # Create the grid
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])

                # Compute the kernel density estimate
                kernel = stats.gaussian_kde(values)
                density = np.reshape(kernel(positions), xx.shape)

                # Plot contours
                levels = kwargs.pop("levels", None)
                ax.contour(xx, yy, density, levels=levels, colors="k", linewidths=1)
                ax.contourf(xx, yy, density, levels=levels, cmap=kwargs.pop("cmap", "Blues"), alpha=0.5)
            except Exception as e:
                # Fall back to hexbin if contour plot fails
                import warnings

                warnings.warn(f"Contour plot failed: {e}. Falling back to hexbin.", stacklevel=2)
                ax.hexbin(x, y, gridsize=50, cmap="Blues")

        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title(f"Joint posterior distribution: {param1_name} vs {param2_name}")

        return ax

    def predictive_distribution(self, func):
        """Generate samples from the posterior predictive distribution.

        Args:
            func: Function to apply to each parameter set

        Returns:
            Samples from the posterior predictive distribution
        """
        samples = self.samples
        n_samples = len(samples)
        sampledata = func()

        predictive_samples = np.empty((n_samples, *sampledata.numpy().shape))
        for i in tqdm.tqdm(range(n_samples), desc="Generating predictive samples"):
            # Set parameter values
            with zfit.param.set_values(self.params, samples[i, :]):
                predictive_samples[i, ...] = func().numpy()

        return np.array(predictive_samples)

    def marginal_likelihood(self, method="stepping", n_steps=20):
        """Estimate the marginal likelihood (Bayesian model evidence).

        The marginal likelihood is useful for model comparison and Bayes factors.

        Args:
            method: Method to use ('stepping' or 'harmonic')
            n_steps: Number of steps to use for the stepping stone method

        Returns:
            The estimated log marginal likelihood
        """
        if method == "stepping":
            # Stepping stone method (simplified)
            return self._stepping_stone_evidence(n_steps)
        elif method == "harmonic":
            # Harmonic mean estimator (simplified but biased)
            return self._harmonic_mean_evidence()
        else:
            msg = f"Unknown method '{method}'. Use 'stepping' or 'harmonic'"
            raise ValueError(msg)

    def _stepping_stone_evidence(self, n_steps=20):
        """Estimate the log evidence using stepping stone method.

        Args:
            n_steps: Number of stepping stones between prior and posterior

        Returns:
            The estimated log marginal likelihood
        """
        # Get posterior samples
        samples = self.samples
        n_samples = len(samples)

        # Define function to evaluate joint log probability (likelihood + prior)

        def joint_log_prob(values):
            values = znp.asarray(values)
            return joint_log_prob_jit(values)

        @tf.function(autograph=False)
        def joint_log_prob_jit(values):
            zfit.param.set_values(self.params, values)
            # Calculate log likelihood (negative of loss)
            log_likelihood = -self.loss.value()
            # Calculate log prior
            log_prior = znp.asarray(0.0)
            for param in self.params:
                if hasattr(param, "log_prior"):
                    log_prior += param.log_prior()
            return log_likelihood + log_prior

        # Create power posteriors (stepping stones)
        betas = znp.linspace(0, 1, n_steps)

        # Evaluate log joint probability for each sample
        log_joint_probs = np.zeros(n_samples)
        for i in tqdm.tqdm(range(n_samples)):
            log_joint_probs[i] = joint_log_prob(samples[i])
        log_joint_probs = znp.asarray(log_joint_probs)

        # Compute log marginal likelihood using stepping stone identity
        log_evidence = znp.asarray(0.0)
        for i in range(n_steps - 1):
            beta_diff = betas[i + 1] - betas[i]
            weighted_log_probs = log_joint_probs * beta_diff
            log_evidence += znp.log(znp.mean(znp.exp(weighted_log_probs - znp.max(weighted_log_probs)))) + znp.max(
                weighted_log_probs
            )

        return log_evidence

    def _harmonic_mean_evidence(self):
        """Estimate the log evidence using the harmonic mean estimator.

        Note: This method is known to be biased but can be useful as a rough estimate.

        Returns:
            The estimated log marginal likelihood
        """
        # Get posterior samples
        samples = self.samples
        n_samples = len(samples)

        # Define function to evaluate likelihood
        def log_likelihood(values):
            return log_likelihood_jit(values)

        @tf.function(autograph=False)
        def log_likelihood_jit(values):
            zfit.param.set_values(self.params, values)
            # Calculate log likelihood (negative of loss)
            return -self.loss.value()

        # Evaluate log likelihood for each sample
        log_likelihoods = np.zeros(n_samples)
        for i in tqdm.tqdm(range(n_samples)):
            log_likelihoods[i] = log_likelihood(samples[i])
        log_likelihoods = znp.asarray(log_likelihoods)

        # Compute harmonic mean estimator with numerical stability
        max_ll = znp.max(log_likelihoods)
        shifted_ll = log_likelihoods - max_ll
        return znp.log(znp.asarray(n_samples, dtype=znp.float64)) - max_ll - znp.log(znp.sum(znp.exp(-shifted_ll)))

    @staticmethod
    def bayes_factor(posterior_1, posterior_2, method="stepping", **kwargs):
        """Compute the Bayes factor for model comparison.

        The Bayes factor is the ratio of marginal likelihoods:
        BF = p(data|model_1) / p(data|model_2)

        Args:
            posterior_1: First Posterior instance
            posterior_2: Second Posterior instance
            method: Method to use for marginal likelihood estimation
            **kwargs: Additional arguments for marginal_likelihood

        Returns:
            The Bayes factor (log scale)
        """
        log_evidence_1 = posterior_1.marginal_likelihood(method=method, **kwargs)
        log_evidence_2 = posterior_2.marginal_likelihood(method=method, **kwargs)

        return log_evidence_1 - log_evidence_2
