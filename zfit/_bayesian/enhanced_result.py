#  Copyright (c) 2025 zfit

"""Enhanced Results class for Bayesian inference in zfit."""

from __future__ import annotations

import numpy as np
import tqdm

import zfit


class BayesianResultExtensions:
    """Extension methods for BayesianResult class."""

    def effective_sample_size(self, param=None):
        """Calculate the effective sample size for MCMC samples.

        The effective sample size takes into account autocorrelation in the MCMC chain
        and provides a measure of how many independent samples the chain effectively contains.

        Args:
            param: The parameter name or ZfitParameter. If None, return ESS for all parameters

        Returns:
            The effective sample size(s)

        Examples:
            >>> posterior = sampler.sample(loss, n_samples=1000)
            >>> ess = posterior.effective_sample_size()
            >>> print(f"Effective sample size: {ess}")
            >>> # Check if we have enough effective samples
            >>> if np.min(ess) < 100:
            >>>     print("Warning: Low effective sample size, consider running more iterations")
        """
        try:
            from scipy import signal
        except ImportError:
            msg = "scipy is required for effective sample size calculation"
            raise ImportError(msg)

        if param is None:
            return np.array([self.effective_sample_size(i) for i in range(self.samples.shape[1])])

        idx = self._get_param_index(param)
        chain = self.samples[:, idx]

        # Center the chain
        centered_chain = chain - np.mean(chain)

        # Calculate autocorrelation
        n = len(centered_chain)
        # Use FFT for efficiency
        acf = signal.correlate(centered_chain, centered_chain, mode="full") / np.var(centered_chain) / n
        acf = acf[n - 1 :]  # Only use the positive lags

        # Estimate the autocorrelation time using the initial monotone sequence
        # Following Geyer's initial positive sequence estimator
        max_lag = min(n // 2, 1000)  # Don't use too many lags

        # Find where autocorrelation becomes negative or non-monotonic
        for lag in range(1, max_lag):
            if acf[lag] < 0 or acf[lag] > acf[lag - 1]:
                max_lag = lag
                break

        # Calculate effective sample size
        tau = 1.0 + 2.0 * np.sum(acf[1:max_lag])
        return n / tau

    def autocorrelation(self, param, max_lag=100):
        """Calculate the autocorrelation function for a parameter.

        Args:
            param: The parameter name or ZfitParameter
            max_lag: Maximum lag to calculate

        Returns:
            Array of autocorrelation values
        """
        try:
            from scipy import signal
        except ImportError:
            msg = "scipy is required for autocorrelation calculation"
            raise ImportError(msg)

        idx = self._get_param_index(param)
        chain = self.samples[:, idx]

        # Center the chain
        centered_chain = chain - np.mean(chain)

        # Calculate autocorrelation
        n = len(centered_chain)
        max_lag = min(max_lag, n - 1)

        # Use FFT for efficiency
        acf = signal.correlate(centered_chain, centered_chain, mode="full") / np.var(centered_chain) / n
        return acf[n - 1 : n - 1 + max_lag + 1]  # Only use the positive lags up to max_lag

    def plot_autocorrelation(self, param, max_lag=50, ax=None):
        """Plot the autocorrelation function for a parameter.

        Args:
            param: The parameter name or ZfitParameter
            max_lag: Maximum lag to plot
            ax: A matplotlib axis to plot on. If None, create a new figure

        Returns:
            The matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for plotting"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 4))

        idx = self._get_param_index(param)
        param_name = self.param_names[idx]

        acf = self.autocorrelation(param, max_lag=max_lag)
        lags = np.arange(len(acf))

        ax.plot(lags, acf, "o-")
        ax.axhline(y=0, linestyle="--", color="gray")
        # Plot 95% confidence bands
        n = len(self.samples)
        ci = 1.96 / np.sqrt(n)
        ax.axhline(y=ci, linestyle=":", color="gray")
        ax.axhline(y=-ci, linestyle=":", color="gray")

        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation function for {param_name}")

        return ax

    def geweke_diagnostic(self, param=None, first=0.1, last=0.5):
        """Calculate the Geweke diagnostic for convergence assessment.

        The Geweke diagnostic compares the mean of the first part of the chain
        to the mean of the last part of the chain. If the chain has converged,
        these means should be similar.

        Args:
            param: The parameter name or ZfitParameter. If None, return diagnostics for all parameters
            first: Fraction of chain to use for first part (default: 0.1)
            last: Fraction of chain to use for last part (default: 0.5)

        Returns:
            The Geweke diagnostic(s) (z-scores)
        """
        if param is None:
            return np.array([self.geweke_diagnostic(i, first, last) for i in range(self.samples.shape[1])])

        idx = self._get_param_index(param)
        chain = self.samples[:, idx]

        n = len(chain)
        first_size = int(first * n)
        last_size = int(last * n)

        first_part = chain[:first_size]
        last_part = chain[-last_size:]

        # Calculate means
        first_mean = np.mean(first_part)
        last_mean = np.mean(last_part)

        # Calculate standard errors using spectral density at zero frequency
        first_var = self._spectral_variance(first_part)
        last_var = self._spectral_variance(last_part)

        # Z-score
        return (first_mean - last_mean) / np.sqrt(first_var / first_size + last_var / last_size)

    def _spectral_variance(self, x):
        """Estimate the spectral density at zero frequency for a time series.

        Used by the Geweke diagnostic.

        Args:
            x: The time series

        Returns:
            The spectral variance
        """
        try:
            from scipy import signal
        except ImportError:
            msg = "scipy is required for spectral variance calculation"
            raise ImportError(msg)

        # Center the series
        x = x - np.mean(x)
        n = len(x)

        # Calculate autocorrelation up to half the series length
        max_lag = min(n // 2, 100)
        acf = self._autocorrelation_for_spectral(x, max_lag)

        # Apply Parzen window (triangular window)
        window = 1.0 - np.arange(max_lag) / max_lag
        acf = acf * window

        # Spectral variance is 2π times the sum of ACF values
        # The factor of 2 comes from including both positive and negative frequencies
        # The 2π normalization is usually omitted in practice
        return acf[0] + 2 * np.sum(acf[1:])

    def _autocorrelation_for_spectral(self, x, max_lag):
        """Calculate the autocorrelation function for spectral variance.

        Args:
            x: The time series
            max_lag: Maximum lag to calculate

        Returns:
            Array of autocorrelation values
        """
        n = len(x)
        acf = np.zeros(max_lag)
        variance = np.var(x)

        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.sum(x[lag:] * x[: (n - lag)]) / ((n - lag) * variance)

        return acf

    def raftery_lewis_diagnostic(self, param, q=0.025, r=0.005, s=0.95):
        """Calculate the Raftery-Lewis diagnostic for MCMC convergence and chain length.

        This diagnostic estimates the number of iterations needed to estimate
        the quantile q to within an accuracy of r with probability s.

        Args:
            param: The parameter name or ZfitParameter
            q: The quantile to estimate (default: 0.025, the lower bound of a 95% CI)
            r: The desired accuracy (default: 0.005)
            s: The probability of attaining the accuracy (default: 0.95)

        Returns:
            Dictionary with diagnostic information including:
            - 'burn': Number of burn-in iterations needed
            - 'total': Total number of iterations needed
            - 'thin': Thinning factor
            - 'nmin': Minimum number of iterations required to achieve accuracy
            - 'dependence_factor': How much dependence increases the required iterations
        """
        idx = self._get_param_index(param)
        chain = self.samples[:, idx]

        # Convert chain to binary chain based on quantile
        quantile_value = np.quantile(chain, q)
        binary_chain = (chain <= quantile_value).astype(int)

        # Define parameters for the diagnostic
        epsilon = r

        # Estimate transition probabilities for 2-state Markov chain
        len(binary_chain)
        n_0 = np.sum(binary_chain == 0)
        n_1 = np.sum(binary_chain == 1)

        n_01 = np.sum((binary_chain[:-1] == 0) & (binary_chain[1:] == 1))
        n_10 = np.sum((binary_chain[:-1] == 1) & (binary_chain[1:] == 0))

        # Ensure we don't divide by zero
        if n_0 == 0 or n_1 == 0 or n_01 == 0 or n_10 == 0:
            return {
                "burn": np.nan,
                "total": np.nan,
                "thin": np.nan,
                "nmin": np.nan,
                "dependence_factor": np.nan,
                "warning": "Chain has too few state transitions for reliable estimates",
            }

        p_01 = n_01 / n_0
        p_10 = n_10 / n_1

        # Calculate alpha (probability of staying in state 1)
        alpha = 1 - p_10

        # Calculate beta (probability of staying in state 0)
        beta = 1 - p_01

        # Calculate thinning factor k
        if alpha == 1 or beta == 1:
            k = 1  # Chain doesn't mix well
        else:
            # Calculate approximate thinning factor to achieve independence
            rho = alpha + beta - 1
            k = max(1, int(np.ceil(np.log(epsilon * (1 - rho) / max(alpha, beta)) / np.log(abs(rho)))))

        # Calculate required burn-in
        if alpha == 1 or beta == 1:
            m = 1  # Chain doesn't mix well
        else:
            # Required burn-in to reach equilibrium
            m = max(1, int(np.ceil(np.log(epsilon * (1 - rho) / (2 * max(alpha, beta))) / np.log(abs(rho)))))

        # Calculate required iterations after thinning (using binomial)
        p = (1 - alpha) / (2 - alpha - beta)  # Equilibrium probability
        n_min = int(np.ceil(p * (1 - p) * (1.96 / epsilon) ** 2))  # Normal approximation to binomial

        # Total iterations needed
        n_total = k * n_min + m

        # Dependence factor (how much dependence increases the required iterations)
        dependence_factor = n_total / n_min

        return {"burn": m, "total": n_total, "thin": k, "nmin": n_min, "dependence_factor": dependence_factor}

    def posterior_predictive_check(self, data_function, test_statistic_function, n_samples=100):
        """Perform a posterior predictive check.

        This compares the distribution of a test statistic computed on
        replicated datasets to the test statistic computed on the observed data.

        Args:
            data_function: Function that returns the observed data
            test_statistic_function: Function that computes a test statistic from data
            n_samples: Number of posterior predictive samples to generate

        Returns:
            Dictionary with:
            - 'observed_statistic': Test statistic for observed data
            - 'replicated_statistics': Test statistics for replicated datasets
            - 'p_value': Bayesian p-value (proportion of replicated statistics more extreme than observed)
        """
        # Get observed data
        observed_data = data_function()

        # Compute test statistic on observed data
        observed_statistic = test_statistic_function(observed_data)

        # Generate posterior predictive samples
        n_posterior_samples = min(n_samples, len(self.samples))
        indices = np.random.choice(len(self.samples), n_posterior_samples, replace=False)

        replicated_statistics = []

        for i in tqdm.tqdm(indices, desc="Generating posterior predictive samples"):
            # Set parameter values
            with zfit.param.set_values(self._params, self.samples[i, :]):
                # Generate replicated dataset
                replicated_data = data_function()

                # Compute test statistic on replicated data
                replicated_statistic = test_statistic_function(replicated_data)
                replicated_statistics.append(replicated_statistic)

        replicated_statistics = np.array(replicated_statistics)

        # Compute Bayesian p-value
        p_value = np.mean(replicated_statistics >= observed_statistic)

        return {
            "observed_statistic": observed_statistic,
            "replicated_statistics": replicated_statistics,
            "p_value": p_value,
        }

    def plot_posterior_predictive(self, data_function, ax=None, n_samples=20):
        """Plot posterior predictive distribution against observed data.

        Args:
            data_function: Function that returns the observed data
            ax: A matplotlib axis to plot on. If None, create a new figure
            n_samples: Number of posterior predictive samples to plot

        Returns:
            The matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for plotting"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Get observed data
        observed_data = data_function()

        # Plot observed data
        if hasattr(observed_data, "shape") and len(observed_data.shape) == 1:
            # 1D data - plot histogram
            ax.hist(observed_data, bins=30, density=True, alpha=0.7, label="Observed")
        elif hasattr(observed_data, "shape") and len(observed_data.shape) == 2:
            # 2D data - plot scatter
            ax.scatter(observed_data[:, 0], observed_data[:, 1], alpha=0.7, label="Observed")
        else:
            # Try to plot whatever it is
            ax.plot(observed_data, "o", label="Observed")

        # Generate and plot posterior predictive samples
        n_posterior_samples = min(n_samples, len(self.samples))
        indices = np.random.choice(len(self.samples), n_posterior_samples, replace=False)

        for i, idx in enumerate(indices):
            # Set parameter values
            with zfit.param.set_values(self._params, self.samples[idx, :]):
                # Generate replicated dataset
                replicated_data = data_function()

                # Plot replicated data
                alpha = 0.5 if i == 0 else 0.2
                label = "Posterior predictive" if i == 0 else None

                if hasattr(replicated_data, "shape") and len(replicated_data.shape) == 1:
                    # 1D data - plot histogram
                    ax.hist(replicated_data, bins=30, density=True, alpha=alpha, color="red", label=label)
                elif hasattr(replicated_data, "shape") and len(replicated_data.shape) == 2:
                    # 2D data - plot scatter
                    ax.scatter(replicated_data[:, 0], replicated_data[:, 1], alpha=alpha, color="red", label=label)
                else:
                    # Try to plot whatever it is
                    ax.plot(replicated_data, "-", alpha=alpha, color="red", label=label)

        ax.set_title("Posterior Predictive Check")
        ax.legend()

        return ax

    def compare_models(self, other_results, method="waic"):
        """Compare this model with another model using various information criteria.

        Args:
            other_results: Another BayesianResult to compare with
            method: The comparison method to use ('waic', 'dic', 'bic', 'aic', 'bayes_factor')

        Returns:
            Dictionary with comparison results
        """
        if method.lower() == "waic":
            return self._compare_waic(other_results)
        elif method.lower() == "dic":
            return self._compare_dic(other_results)
        elif method.lower() == "bic":
            return self._compare_bic(other_results)
        elif method.lower() == "aic":
            return self._compare_aic(other_results)
        elif method.lower() == "bayes_factor":
            return self._compare_bayes_factor(other_results)
        else:
            msg = f"Unknown comparison method: {method}. Use 'waic', 'dic', 'bic', 'aic', or 'bayes_factor'."
            raise ValueError(msg)

    def _compare_waic(self, other_results):
        """Compare models using Widely Applicable Information Criterion (WAIC).

        WAIC estimates the out-of-sample prediction error and is fully Bayesian.

        Args:
            other_results: Another BayesianResult to compare with

        Returns:
            Dictionary with comparison results
        """
        waic1 = self.waic()
        waic2 = other_results.waic()

        # Lower WAIC is better
        diff = waic1 - waic2
        se_diff = np.sqrt(waic1["se"] ** 2 + waic2["se"] ** 2)

        # Compare using scale suggested by Burnham & Anderson (2002)
        if abs(diff) < 2:
            interpretation = "No substantial difference between models"
        elif abs(diff) < 10:
            interpretation = "Some evidence in favor of the model with lower WAIC"
        else:
            interpretation = "Strong evidence in favor of the model with lower WAIC"

        return {
            "method": "WAIC",
            "model1": waic1,
            "model2": waic2,
            "difference": diff,
            "se_difference": se_diff,
            "better_model": "model1" if diff < 0 else "model2",
            "interpretation": interpretation,
        }

    def waic(self):
        """Calculate the Widely Applicable Information Criterion (WAIC).

        WAIC estimates the out-of-sample prediction error and is fully Bayesian.

        Returns:
            Dictionary with WAIC results
        """
        # Calculate log-likelihood for each posterior sample
        n_samples = len(self.samples)
        log_likelihoods = np.zeros(n_samples)

        for i in range(n_samples):
            with zfit.param.set_values(self._params, self.samples[i, :]):
                # Calculate log likelihood (negative of loss)
                log_likelihoods[i] = -self._loss.value().numpy()

        # Calculate LPPD (log pointwise predictive density)
        lppd = np.sum(np.log(np.mean(np.exp(log_likelihoods))))

        # Calculate penalty term (effective number of parameters)
        var_log_likelihoods = np.var(log_likelihoods)
        p_waic = var_log_likelihoods

        # WAIC = -2 * (LPPD - p_WAIC)
        waic = -2 * (lppd - p_waic)

        # Standard error
        se = np.sqrt(n_samples * var_log_likelihoods)

        return {"waic": waic, "lppd": lppd, "p_waic": p_waic, "se": se}

    def _compare_dic(self, other_results):
        """Compare models using Deviance Information Criterion (DIC).

        Args:
            other_results: Another BayesianResult to compare with

        Returns:
            Dictionary with comparison results
        """
        dic1 = self.dic()
        dic2 = other_results.dic()

        # Lower DIC is better
        diff = dic1 - dic2

        # Compare using scale suggested by Spiegelhalter et al.
        if abs(diff) < 2:
            interpretation = "No substantial difference between models"
        elif abs(diff) < 10:
            interpretation = "Some evidence in favor of the model with lower DIC"
        else:
            interpretation = "Strong evidence in favor of the model with lower DIC"

        return {
            "method": "DIC",
            "model1": dic1,
            "model2": dic2,
            "difference": diff,
            "better_model": "model1" if diff < 0 else "model2",
            "interpretation": interpretation,
        }

    def dic(self):
        """Calculate the Deviance Information Criterion (DIC).

        Returns:
            Dictionary with DIC results
        """
        # Calculate mean parameter values
        mean_params = self.mean()

        # Set parameters to mean values and calculate deviance
        with zfit.param.set_values(self._params, mean_params):
            # Deviance at mean parameters (D_hat)
            d_hat = -2 * -self._loss.value().numpy()

        # Calculate deviance for each posterior sample
        n_samples = len(self.samples)
        deviances = np.zeros(n_samples)

        for i in range(n_samples):
            with zfit.param.set_values(self._params, self.samples[i, :]):
                # Calculate deviance (-2 * log likelihood)
                deviances[i] = -2 * -self._loss.value().numpy()

        # Calculate mean deviance (D_bar)
        d_bar = np.mean(deviances)

        # Calculate effective number of parameters (p_D)
        p_d = d_bar - d_hat

        # DIC = D_hat + 2 * p_D = D_bar + p_D
        dic = d_bar + p_d

        return {"dic": dic, "d_hat": d_hat, "d_bar": d_bar, "p_d": p_d}

    def _compare_bic(self, other_results):
        """Compare models using Bayesian Information Criterion (BIC).

        Args:
            other_results: Another BayesianResult to compare with

        Returns:
            Dictionary with comparison results
        """
        bic1 = self.bic()
        bic2 = other_results.bic()

        # Lower BIC is better
        diff = bic1 - bic2

        # Compare using scale suggested by Kass & Raftery
        if abs(diff) < 2:
            interpretation = "No substantial difference between models"
        elif abs(diff) < 6:
            interpretation = "Positive evidence in favor of the model with lower BIC"
        elif abs(diff) < 10:
            interpretation = "Strong evidence in favor of the model with lower BIC"
        else:
            interpretation = "Very strong evidence in favor of the model with lower BIC"

        # BIC difference approximates 2 * log(Bayes Factor)
        approx_bf = np.exp(-diff / 2)

        return {
            "method": "BIC",
            "model1": bic1,
            "model2": bic2,
            "difference": diff,
            "approximate_bayes_factor": approx_bf,
            "better_model": "model1" if diff < 0 else "model2",
            "interpretation": interpretation,
        }

    def bic(self):
        """Calculate the Bayesian Information Criterion (BIC).

        BIC = -2 * log(likelihood) + k * log(n)
        where k is the number of parameters and n is the sample size.

        Returns:
            The BIC value
        """
        # Set parameters to mean values and calculate log likelihood
        with zfit.param.set_values(self._params, self.mean()):
            # Log likelihood at mean parameters
            log_likelihood = -self._loss.value().numpy()

        # Number of parameters
        k = len(self._params)

        # Sample size (try to get it from the loss/data, otherwise use a placeholder)
        try:
            n = self._loss.model.n_events
        except:
            # If we can't get the sample size, use a warning
            n = 100  # Placeholder
            import warnings

            warnings.warn("Could not determine sample size for BIC calculation. Using placeholder value.", stacklevel=2)

        # BIC = -2 * log(likelihood) + k * log(n)
        return -2 * log_likelihood + k * np.log(n)

    def _compare_aic(self, other_results):
        """Compare models using Akaike Information Criterion (AIC).

        Args:
            other_results: Another BayesianResult to compare with

        Returns:
            Dictionary with comparison results
        """
        aic1 = self.aic()
        aic2 = other_results.aic()

        # Lower AIC is better
        diff = aic1 - aic2

        # Compare using scale suggested by Burnham & Anderson
        if abs(diff) < 2:
            interpretation = "No substantial difference between models"
        elif abs(diff) < 4:
            interpretation = "Some evidence in favor of the model with lower AIC"
        elif abs(diff) < 7:
            interpretation = "Strong evidence in favor of the model with lower AIC"
        else:
            interpretation = "Very strong evidence in favor of the model with lower AIC"

        return {
            "method": "AIC",
            "model1": aic1,
            "model2": aic2,
            "difference": diff,
            "better_model": "model1" if diff < 0 else "model2",
            "interpretation": interpretation,
        }

    def aic(self):
        """Calculate the Akaike Information Criterion (AIC).

        AIC = -2 * log(likelihood) + 2 * k
        where k is the number of parameters.

        Returns:
            The AIC value
        """
        # Set parameters to mean values and calculate log likelihood
        with zfit.param.set_values(self._params, self.mean()):
            # Log likelihood at mean parameters
            log_likelihood = -self._loss.value().numpy()

        # Number of parameters
        k = len(self._params)

        # AIC = -2 * log(likelihood) + 2 * k
        return -2 * log_likelihood + 2 * k

    def _compare_bayes_factor(self, other_results, method="stepping", **kwargs):
        """Compare models using Bayes factors.

        Args:
            other_results: Another BayesianResult to compare with
            method: Method to use for marginal likelihood estimation
            **kwargs: Additional arguments for marginal_likelihood

        Returns:
            Dictionary with comparison results
        """
        log_bf = self.bayes_factor(self, other_results, method=method, **kwargs)
        bf = np.exp(log_bf)

        # Interpret Bayes factor according to Kass & Raftery
        if bf < 1:
            # Favor model 2
            bf_inv = 1 / bf
            if bf_inv < 3:
                interpretation = "Barely worth mentioning evidence for model 2"
            elif bf_inv < 20:
                interpretation = "Positive evidence for model 2"
            elif bf_inv < 150:
                interpretation = "Strong evidence for model 2"
            else:
                interpretation = "Very strong evidence for model 2"
        # Favor model 1
        elif bf < 3:
            interpretation = "Barely worth mentioning evidence for model 1"
        elif bf < 20:
            interpretation = "Positive evidence for model 1"
        elif bf < 150:
            interpretation = "Strong evidence for model 1"
        else:
            interpretation = "Very strong evidence for model 1"

        return {
            "method": "Bayes Factor",
            "log_bayes_factor": log_bf,
            "bayes_factor": bf,
            "better_model": "model1" if bf > 1 else "model2",
            "interpretation": interpretation,
        }

    def plot_kl_divergence_mh(self, param1, param2, n_bins=20, ax=None):
        """Plot an estimate of local Markov-Hastings algorithm efficiency using Kullback-Leibler divergence.

        This plot helps visualize regions where the sampler might be having trouble exploring the posterior,
        by showing the KL divergence between the proposal distribution and the target distribution.

        Args:
            param1: First parameter name or index for x-axis
            param2: Second parameter name or index for y-axis
            n_bins: Number of bins for the grid
            ax: Matplotlib axis for plotting

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
        except ImportError:
            msg = "matplotlib and scipy are required for KL divergence plots"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Get parameter indices
        idx1 = self._get_param_index(param1)
        idx2 = self._get_param_index(param2)

        # Extract samples for the two parameters
        x = self.samples[:, idx1]
        y = self.samples[:, idx2]

        # Create a 2D histogram
        H, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)

        # Normalize to create a probability distribution
        H = H / np.sum(H)

        # Create a grid for KL divergence values
        kl_grid = np.zeros((n_bins, n_bins))

        # For each bin, calculate KL divergence between local and global distributions
        for i in range(n_bins):
            for j in range(n_bins):
                if H[i, j] > 0:  # Only calculate if there are samples in the bin
                    # Define the center of the bin
                    x_center = (x_edges[i] + x_edges[i + 1]) / 2
                    y_center = (y_edges[j] + y_edges[j + 1]) / 2

                    # Create a small local distribution around this point
                    # This simulates the proposal distribution of MH algorithm
                    x_local = x_center + 0.1 * (x.max() - x.min()) * np.random.randn(1000)
                    y_local = y_center + 0.1 * (y.max() - y.min()) * np.random.randn(1000)

                    # Estimate the probability density of the local and global distributions
                    try:
                        # Create a KDE of the global distribution
                        kde_global = stats.gaussian_kde(np.vstack([x, y]))

                        # Evaluate the global density at the local points
                        global_density = kde_global(np.vstack([x_local, y_local]))

                        # Create a KDE of the local distribution
                        kde_local = stats.gaussian_kde(np.vstack([x_local, y_local]))

                        # Evaluate the local density at the local points
                        local_density = kde_local(np.vstack([x_local, y_local]))

                        # Calculate KL divergence
                        kl = np.mean(np.log(local_density / global_density))
                        kl_grid[i, j] = kl
                    except:
                        # If KDE fails, set KL to a high value
                        kl_grid[i, j] = 0

        # Plot the KL divergence
        im = ax.imshow(
            kl_grid.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="viridis",
            aspect="auto",
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, label="KL Divergence")

        # Add labels
        ax.set_xlabel(self.param_names[idx1])
        ax.set_ylabel(self.param_names[idx2])
        ax.set_title("Local KL Divergence (higher = less efficient sampling)")

        return ax

    def partial_dependence(self, param, n_points=100, fixed_params=None):
        """Calculate the partial dependence of the log-posterior on a parameter.

        This marginalizes over all other parameters to show how this parameter
        affects the posterior when all other parameters are integrated out.

        Args:
            param: Parameter name or index to calculate dependence for
            n_points: Number of points to evaluate
            fixed_params: Dictionary of parameters to fix at specific values

        Returns:
            Dictionary with parameter values and corresponding log-posterior values
        """
        idx = self._get_param_index(param)
        param_name = self.param_names[idx]

        # Define the range to evaluate
        param_min = np.min(self.samples[:, idx])
        param_max = np.max(self.samples[:, idx])
        margin = 0.1 * (param_max - param_min)
        param_min -= margin
        param_max += margin

        # Apply parameter bounds if available
        if hasattr(self._params[idx], "lower") and self._params[idx].lower is not None:
            param_min = max(param_min, self._params[idx].lower)
        if hasattr(self._params[idx], "upper") and self._params[idx].upper is not None:
            param_max = min(param_max, self._params[idx].upper)

        # Create evaluation points
        param_values = np.linspace(param_min, param_max, n_points)
        log_posterior_values = np.zeros(n_points)

        # For each point, marginalize over other parameters by MC integration
        for i, val in enumerate(param_values):
            # Use a subset of posterior samples for efficiency
            n_samples = min(100, len(self.samples))
            log_posterior_sum = 0

            for j in range(n_samples):
                # Create parameter values with current value for target parameter
                # and posterior sample values for other parameters
                param_vals = self.samples[j, :].copy()
                param_vals[idx] = val

                # Fix any specified parameters
                if fixed_params:
                    for fix_param, fix_val in fixed_params.items():
                        fix_idx = self._get_param_index(fix_param)
                        param_vals[fix_idx] = fix_val

                # Evaluate log posterior
                with zfit.param.set_values(self._params, param_vals):
                    # Log likelihood
                    log_likelihood = -self._loss.value().numpy()
                    # Log prior
                    log_prior = 0
                    for k, param in enumerate(self._params):
                        if hasattr(param, "prior") and param.prior is not None:
                            log_prior += param.prior.log_pdf(param_vals[k]).numpy()

                    log_posterior = log_likelihood + log_prior
                    log_posterior_sum += log_posterior

            log_posterior_values[i] = log_posterior_sum / n_samples

        return {"param_name": param_name, "param_values": param_values, "log_posterior_values": log_posterior_values}

    def plot_partial_dependence(self, param, n_points=100, fixed_params=None, ax=None):
        """Plot the partial dependence of the log-posterior on a parameter.

        Args:
            param: Parameter name or index to calculate dependence for
            n_points: Number of points to evaluate
            fixed_params: Dictionary of parameters to fix at specific values
            ax: Matplotlib axis for plotting

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for partial dependence plots"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Calculate partial dependence
        pd = self.partial_dependence(param, n_points, fixed_params)

        # Plot
        ax.plot(pd["param_values"], pd["log_posterior_values"])

        # Add labels
        ax.set_xlabel(pd["param_name"])
        ax.set_ylabel("Log Posterior")
        ax.set_title(f"Partial Dependence of Log Posterior on {pd['param_name']}")

        return ax

    def compare_predictions(self, predictions_func, other_result, data=None):
        """Compare predictive performance with another model.

        Args:
            predictions_func: Function that takes parameters and returns predictions
            other_result: Another BayesianResult to compare with
            data: Observed data for computing metrics

        Returns:
            Dictionary with comparison metrics
        """
        # Generate predictions using both models
        pred1 = self.predictive_distribution(predictions_func)
        pred2 = other_result.predictive_distribution(predictions_func)

        # Calculate summary statistics
        pred1_mean = np.mean(pred1, axis=0)
        pred2_mean = np.mean(pred2, axis=0)

        pred1_std = np.std(pred1, axis=0)
        pred2_std = np.std(pred2, axis=0)

        # Calculate mean absolute difference between predictions
        mean_diff = np.mean(np.abs(pred1_mean - pred2_mean))

        # If data is provided, calculate metrics
        metrics = {}
        if data is not None:
            # Root Mean Squared Error
            rmse1 = np.sqrt(np.mean((pred1_mean - data) ** 2))
            rmse2 = np.sqrt(np.mean((pred2_mean - data) ** 2))

            # Mean Absolute Error
            mae1 = np.mean(np.abs(pred1_mean - data))
            mae2 = np.mean(np.abs(pred2_mean - data))

            # Coverage of 95% posterior predictive intervals
            lower1 = np.percentile(pred1, 2.5, axis=0)
            upper1 = np.percentile(pred1, 97.5, axis=0)
            coverage1 = np.mean((data >= lower1) & (data <= upper1))

            lower2 = np.percentile(pred2, 2.5, axis=0)
            upper2 = np.percentile(pred2, 97.5, axis=0)
            coverage2 = np.mean((data >= lower2) & (data <= upper2))

            metrics = {
                "rmse": {"model1": rmse1, "model2": rmse2},
                "mae": {"model1": mae1, "model2": mae2},
                "coverage_95": {"model1": coverage1, "model2": coverage2},
            }

        return {
            "mean_difference": mean_diff,
            "predictions": {
                "model1": {"mean": pred1_mean, "std": pred1_std},
                "model2": {"mean": pred2_mean, "std": pred2_std},
            },
            "metrics": metrics,
        }

    def sensitivity_analysis(self, param, n_values=10, metric_func=None):
        """Perform a sensitivity analysis for a parameter.

        This fixes a parameter at different values and evaluates how a metric changes.

        Args:
            param: Parameter name or index to analyze
            n_values: Number of values to test
            metric_func: Function that computes a metric given parameter values
                        If None, uses negative log posterior

        Returns:
            Dictionary with parameter values and corresponding metric values
        """
        idx = self._get_param_index(param)
        param_name = self.param_names[idx]

        # Define the range to evaluate
        param_min = np.min(self.samples[:, idx])
        param_max = np.max(self.samples[:, idx])
        margin = 0.1 * (param_max - param_min)
        param_min -= margin
        param_max += margin

        # Apply parameter bounds if available
        if hasattr(self._params[idx], "lower") and self._params[idx].lower is not None:
            param_min = max(param_min, self._params[idx].lower)
        if hasattr(self._params[idx], "upper") and self._params[idx].upper is not None:
            param_max = min(param_max, self._params[idx].upper)

        # Create evaluation points
        param_values = np.linspace(param_min, param_max, n_values)
        metric_values = np.zeros(n_values)

        # Default metric is negative log posterior
        if metric_func is None:

            def metric_func(params):
                with zfit.param.set_values(self._params, params):
                    return self._loss.value().numpy()

        # Get mean values for all parameters
        mean_params = self.mean()

        # For each point, fix the parameter and compute the metric
        for i, val in enumerate(param_values):
            # Set the parameter to the current value
            param_vals = mean_params.copy()
            param_vals[idx] = val

            # Compute metric
            metric_values[i] = metric_func(param_vals)

        return {"param_name": param_name, "param_values": param_values, "metric_values": metric_values}

    def plot_sensitivity_analysis(self, param, n_values=10, metric_func=None, ax=None):
        """Plot a sensitivity analysis for a parameter.

        Args:
            param: Parameter name or index to analyze
            n_values: Number of values to test
            metric_func: Function that computes a metric given parameter values
            ax: Matplotlib axis for plotting

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for sensitivity analysis plots"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Calculate sensitivity
        sa = self.sensitivity_analysis(param, n_values, metric_func)

        # Plot
        ax.plot(sa["param_values"], sa["metric_values"])

        # Add labels
        ax.set_xlabel(sa["param_name"])
        ylabel = "Metric Value" if metric_func else "Negative Log Posterior"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Sensitivity Analysis for {sa['param_name']}")

        return ax

    def plot_prior_posterior(self, param, ax=None):
        """Plot the prior and posterior distributions for a parameter.

        Args:
            param: Parameter name or index
            ax: Matplotlib axis for plotting

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
        except ImportError:
            msg = "matplotlib and scipy are required for prior-posterior plots"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))

        idx = self._get_param_index(param)
        param_name = self.param_names[idx]
        param_obj = self._params[idx]

        # Plot posterior
        posterior_samples = self.samples[:, idx]
        ax.hist(posterior_samples, bins=50, density=True, alpha=0.5, label="Posterior")

        # Plot posterior KDE
        try:
            posterior_kde = stats.gaussian_kde(posterior_samples)
            x = np.linspace(np.min(posterior_samples), np.max(posterior_samples), 1000)
            ax.plot(x, posterior_kde(x), "b-", label="Posterior KDE")
        except:
            # Skip KDE if it fails
            pass

        # Plot prior if available
        if hasattr(param_obj, "prior") and param_obj.prior is not None:
            # Create the same range of x values
            x = np.linspace(np.min(posterior_samples), np.max(posterior_samples), 1000)

            # Evaluate prior PDF
            prior_pdf = np.exp(param_obj.prior.log_pdf(x).numpy())

            # Plot prior
            ax.plot(x, prior_pdf, "r--", label="Prior")

        # Add labels
        ax.set_xlabel(param_name)
        ax.set_ylabel("Density")
        ax.set_title(f"Prior and Posterior for {param_name}")
        ax.legend()

        return ax

    def parameterized_predictions(self, prediction_function, x_values, n_samples=100):
        """Generate parameterized predictions with uncertainty bands.

        Args:
            prediction_function: Function that takes parameters and x_values and returns predictions
            x_values: Input values for predictions
            n_samples: Number of posterior samples to use

        Returns:
            Dictionary with predictions
        """
        # Randomly select samples for efficiency
        n_posterior_samples = min(n_samples, len(self.samples))
        indices = np.random.choice(len(self.samples), n_posterior_samples, replace=False)

        # Generate predictions for each sample
        predictions = []
        for i in indices:
            with zfit.param.set_values(self._params, self.samples[i, :]):
                pred = prediction_function(x_values)
                predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        median_pred = np.median(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Calculate credible intervals
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)

        lower_68 = np.percentile(predictions, 16, axis=0)
        upper_68 = np.percentile(predictions, 84, axis=0)

        return {
            "x_values": x_values,
            "mean": mean_pred,
            "median": median_pred,
            "std": std_pred,
            "ci_95": (lower_95, upper_95),
            "ci_68": (lower_68, upper_68),
            "samples": predictions,
        }

    def plot_parameterized_predictions(self, prediction_function, x_values, data=None, ax=None, n_samples=100):
        """Plot parameterized predictions with uncertainty bands.

        Args:
            prediction_function: Function that takes parameters and x_values and returns predictions
            x_values: Input values for predictions
            data: Observed data to plot (optional)
            ax: Matplotlib axis for plotting
            n_samples: Number of posterior samples to use

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for prediction plots"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Generate predictions
        preds = self.parameterized_predictions(prediction_function, x_values, n_samples)

        # Plot data if provided
        if data is not None:
            ax.scatter(x_values, data, color="black", s=20, alpha=0.7, label="Data")

        # Plot mean prediction
        ax.plot(x_values, preds["mean"], "b-", label="Mean Prediction")

        # Plot 95% credible interval
        ax.fill_between(
            x_values, preds["ci_95"][0], preds["ci_95"][1], color="blue", alpha=0.2, label="95% Credible Interval"
        )

        # Plot 68% credible interval
        ax.fill_between(
            x_values, preds["ci_68"][0], preds["ci_68"][1], color="blue", alpha=0.3, label="68% Credible Interval"
        )

        # Add labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Parameterized Predictions with Uncertainty")
        ax.legend()

        return ax

    def residual_analysis(self, prediction_function, x_values, data):
        """Analyze residuals between predictions and observed data.

        Args:
            prediction_function: Function that takes parameters and x_values and returns predictions
            x_values: Input values for predictions
            data: Observed data

        Returns:
            Dictionary with residual analysis
        """
        # Generate predictions
        preds = self.parameterized_predictions(prediction_function, x_values)

        # Calculate residuals
        residuals = data - preds["mean"]

        # Calculate standardized residuals
        std_residuals = residuals / preds["std"]

        # Test for normality of residuals
        try:
            from scipy import stats

            shapiro_test = stats.shapiro(residuals)
            normality_test = {
                "test": "Shapiro-Wilk",
                "statistic": shapiro_test.statistic,
                "p_value": shapiro_test.pvalue,
            }
        except:
            normality_test = None

        # Check for autocorrelation in residuals
        try:
            acf = self._autocorrelation_for_spectral(residuals, min(20, len(residuals) // 4))
            has_autocorrelation = any(abs(acf[1:]) > 1.96 / np.sqrt(len(residuals)))
        except:
            acf = None
            has_autocorrelation = None

        return {
            "x_values": x_values,
            "data": data,
            "predictions": preds["mean"],
            "residuals": residuals,
            "std_residuals": std_residuals,
            "normality_test": normality_test,
            "autocorrelation": {"values": acf, "has_autocorrelation": has_autocorrelation},
        }

    def plot_residuals(self, prediction_function, x_values, data, ax=None):
        """Plot residuals analysis.

        Args:
            prediction_function: Function that takes parameters and x_values and returns predictions
            x_values: Input values for predictions
            data: Observed data
            ax: Matplotlib axis for plotting

        Returns:
            Tuple of Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for residual plots"
            raise ImportError(msg)

        if ax is None:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            ax1, ax2, ax3, ax4 = axs.flatten()
        else:
            ax1, ax2, ax3, ax4 = ax

        # Analyze residuals
        res = self.residual_analysis(prediction_function, x_values, data)

        # Plot 1: Data vs. Predictions
        ax1.scatter(x_values, data, label="Data", color="black", alpha=0.7)
        ax1.plot(x_values, res["predictions"], label="Predictions", color="blue")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Data vs. Predictions")
        ax1.legend()

        # Plot 2: Residuals vs. x
        ax2.scatter(x_values, res["residuals"], color="red", alpha=0.7)
        ax2.axhline(y=0, color="black", linestyle="--")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs. x")

        # Plot 3: Standardized Residuals
        ax3.scatter(x_values, res["std_residuals"], color="red", alpha=0.7)
        ax3.axhline(y=0, color="black", linestyle="--")
        ax3.axhline(y=1.96, color="gray", linestyle="--")
        ax3.axhline(y=-1.96, color="gray", linestyle="--")
        ax3.set_xlabel("x")
        ax3.set_ylabel("Standardized Residuals")
        ax3.set_title("Standardized Residuals")

        # Plot 4: QQ Plot
        try:
            from scipy import stats

            residuals = res["residuals"]
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title("Normal Q-Q Plot")
        except:
            ax4.text(
                0.5, 0.5, "QQ Plot unavailable\n(scipy required)", ha="center", va="center", transform=ax4.transAxes
            )

        # Add information about normality test if available
        if res["normality_test"]:
            test = res["normality_test"]
            p_value = test["p_value"]
            conclusion = "Normally distributed" if p_value > 0.05 else "Not normally distributed"
            ax4.text(
                0.05,
                0.95,
                f"{test['test']} p-value: {p_value:.4f}\n{conclusion}",
                transform=ax4.transAxes,
                ha="left",
                va="top",
            )

        return (ax1, ax2, ax3, ax4)

    def loo_cv(self, log_likelihood_func):
        """Calculate leave-one-out cross-validation using Pareto smoothed importance sampling.

        Args:
            log_likelihood_func: Function that takes parameters and returns pointwise log likelihoods

        Returns:
            Dictionary with LOO-CV results
        """
        try:
            import arviz
        except ImportError:
            msg = "arviz is required for LOO-CV. Install with 'pip install arviz'."
            raise ImportError(msg)

        # Generate pointwise log-likelihoods for each posterior sample
        n_samples = len(self.samples)
        pointwise_log_liks = []

        for i in range(n_samples):
            with zfit.param.set_values(self._params, self.samples[i, :]):
                # Calculate pointwise log likelihoods
                pointwise_ll = log_likelihood_func()
                pointwise_log_liks.append(pointwise_ll)

        # Convert to appropriate shape for ArviZ
        pointwise_log_liks = np.array(pointwise_log_liks)

        # Calculate LOO using PSIS
        loo = arviz.loo(pointwise_log_liks)

        # Extract results
        loo_results = {
            "loo": loo.loo,
            "loo_se": loo.loo_se,
            "p_loo": loo.p_loo,
            "pareto_k": loo.pareto_k,
            "warnings": [],
        }

        # Add warnings for high Pareto k values
        if np.any(loo.pareto_k > 0.7):
            loo_results["warnings"].append(
                "Very high Pareto k values detected (>0.7). LOO-CV estimates are unreliable."
            )
        elif np.any(loo.pareto_k > 0.5):
            loo_results["warnings"].append("High Pareto k values detected (>0.5). LOO-CV estimates may be unstable.")

        return loo_results

    def compare_loo(self, other_result, log_likelihood_func1, log_likelihood_func2=None):
        """Compare models using leave-one-out cross-validation.

        Args:
            other_result: Another BayesianResult to compare with
            log_likelihood_func1: Function that takes parameters and returns pointwise log likelihoods for this model
            log_likelihood_func2: Function for the other model. If None, assumes the same as log_likelihood_func1

        Returns:
            Dictionary with comparison results
        """
        # If log_likelihood_func2 not provided, use the same function for both models
        if log_likelihood_func2 is None:
            log_likelihood_func2 = log_likelihood_func1

        # Calculate LOO for both models
        loo1 = self.loo_cv(log_likelihood_func1)
        loo2 = other_result.loo_cv(log_likelihood_func2)

        # Calculate difference
        loo_diff = loo1["loo"] - loo2["loo"]
        loo_diff_se = np.sqrt(loo1["loo_se"] ** 2 + loo2["loo_se"] ** 2)

        # Determine better model
        better_model = "model1" if loo_diff > 0 else "model2"

        # Interpret the difference
        if abs(loo_diff) < 2 * loo_diff_se:
            interpretation = "No significant difference between models"
        elif abs(loo_diff) < 10:
            interpretation = "Moderate evidence in favor of the model with higher LOO-CV"
        else:
            interpretation = "Strong evidence in favor of the model with higher LOO-CV"

        return {
            "model1": loo1,
            "model2": loo2,
            "difference": loo_diff,
            "difference_se": loo_diff_se,
            "better_model": better_model,
            "interpretation": interpretation,
            "warnings": loo1["warnings"] + loo2["warnings"],
        }

    def plot_ppc_ecdf(self, prediction_function, data, ax=None, n_samples=20):
        """Plot empirical cumulative distribution function (ECDF) for posterior predictive check.

        Args:
            prediction_function: Function that generates predictive data
            data: Observed data
            ax: Matplotlib axis for plotting
            n_samples: Number of posterior predictive samples to generate

        Returns:
            Matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for ECDF plots"
            raise ImportError(msg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot ECDF for observed data
        observed_data = np.sort(data)
        observed_ecdf = np.arange(1, len(observed_data) + 1) / len(observed_data)

        ax.step(observed_data, observed_ecdf, where="post", color="black", linewidth=2, label="Observed")

        # Generate posterior predictive samples
        n_posterior_samples = min(n_samples, len(self.samples))
        indices = np.random.choice(len(self.samples), n_posterior_samples, replace=False)

        for i, idx in enumerate(indices):
            # Set parameter values
            with zfit.param.set_values(self._params, self.samples[idx, :]):
                # Generate predictive data
                pred_data = prediction_function()

                # Calculate ECDF
                sorted_pred_data = np.sort(pred_data)
                pred_ecdf = np.arange(1, len(sorted_pred_data) + 1) / len(sorted_pred_data)

                # Plot posterior predictive ECDF
                alpha = 0.7 if i == 0 else 0.3
                label = "Posterior predictive" if i == 0 else None
                ax.step(sorted_pred_data, pred_ecdf, where="post", color="blue", alpha=alpha, linewidth=1, label=label)

        ax.set_xlabel("Value")
        ax.set_ylabel("ECDF")
        ax.set_title("Posterior Predictive Check - ECDF")
        ax.legend()

        return ax

    def plot_diagnostics(self, param=None, figsize=(12, 10)):
        """Create a comprehensive diagnostic plot for parameter(s).

        Args:
            param: Parameter name or index. If None, create plots for all parameters
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib is required for diagnostic plots"
            raise ImportError(msg)

        if param is None:
            # Create diagnostics for all parameters (could be a lot of plots)
            # Here we'll do just the first 4 for brevity
            params = self._params[: min(4, len(self._params))]

            # Create a multi-page figure
            fig = plt.figure(figsize=figsize)

            for _i, param in enumerate(params):
                self.plot_diagnostics(param, figsize=figsize)

            return fig

        # Get parameter index
        idx = self._get_param_index(param)
        param_name = self.param_names[idx]

        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Trace plot
        self.plot_trace(param, ax=axs[0, 0])

        # Plot 2: Posterior distribution
        self.plot_posterior(param, ax=axs[0, 1])

        # Plot 3: Autocorrelation
        self.plot_autocorrelation(param, ax=axs[1, 0])

        # Plot 4: Prior-Posterior comparison
        self.plot_prior_posterior(param, ax=axs[1, 1])

        # Add overall title
        fig.suptitle(f"Diagnostics for {param_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        return fig

    # Methods for enhancing result initialization
    def get_hdi_params(self):
        """Set parameters to their Highest Density Interval (HDI) midpoints.

        For each parameter, sets its value to the midpoint of the 95% HDI.
        This can sometimes give better point estimates than the mean for
        skewed or multimodal posteriors.

        Returns:
            Dictionary of parameter values
        """
        param_values = {}

        for _i, param in enumerate(self._params):
            lower, upper = self.highest_density_interval(param)
            # Set to midpoint of HDI
            midpoint = (lower + upper) / 2
            param_values[param] = midpoint

        return param_values

    def set_params_to_hdi_midpoint(self):
        """Set all parameters to the midpoints of their 95% HDI intervals.

        Returns:
            self: Returns self for method chaining
        """
        param_values = self.get_hdi_params()

        for param, value in param_values.items():
            param.set_value(value)

        return self
