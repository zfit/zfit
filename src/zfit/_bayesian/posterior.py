"""Posterior class for Bayesian inference results in zfit."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import zfit
from zfit.util.container import convert_to_container, is_container

if TYPE_CHECKING:
    from collections.abc import Iterable

    import arviz as az
    import numpy.typing as npt
    import pandas as pd

    from zfit.core.interfaces import ZfitLoss, ZfitParameter
    from zfit.mcmc import MCMCSampler

    from .priors import KDE


class PosteriorSamples:
    def __init__(
        self,
        samples: npt.NDArray[np.float64],
        params: Iterable[ZfitParameter],
        loss: ZfitLoss,
        sampler: MCMCSampler,
        n_warmup: int,
        n_samples: int,
        raw_result: object | None = None,
        info: dict | None = None,
    ):
        """Posterior samples from MCMC bayesian inference.

        Args:
            samples: Array of shape (n_samples, n_params) containing MCMC samples.
            params: List of ZfitParameter objects.
            loss: The ZfitLoss that was sampled.
            sampler: The ZfitSampler that generated the samples.
            n_warmup: Number of warmup/burn-in steps.
            n_samples: Number of posterior samples per walker.
            raw_result: Raw result from the sampler.
            info: Additional information dictionary.
        """

        if not isinstance(n_warmup, (int, np.integer)) or n_warmup < 0:
            msg = f"n_warmup must be a non-negative integer, got {n_warmup}"
            raise ValueError(msg)

        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            msg = f"n_samples must be a positive integer, got {n_samples}"
            raise ValueError(msg)

        if info is None:
            info = {}

        self.samples = np.asarray(samples)

        # Validate sample shape
        if self.samples.ndim != 2:
            msg = f"samples must be a 2D array, got shape {self.samples.shape}"
            raise ValueError(msg)

        if len(self.samples) == 0:
            msg = "samples cannot be empty"
            raise ValueError(msg)

        self._params = convert_to_container(params)

        # Validate params
        if not self._params:
            msg = "params cannot be empty"
            raise ValueError(msg)

        # Check samples and params consistency
        if self.samples.shape[1] != len(self._params):
            msg = f"Number of parameters in samples ({self.samples.shape[1]}) does not match number of parameters ({len(self._params)})"
            raise ValueError(msg)

        self._loss = loss
        self._sampler = sampler
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.raw_result = raw_result
        self.info = info

        # Create parameter mappings
        # Map both name->param and param->name for efficient lookup
        self._param_by_name = {param.name: param for param in self._params}
        self._name_by_param = {param: param.name for param in self._params}

        # Create position mapping for internal numpy operations
        self._position_by_name = {param.name: i for i, param in enumerate(self._params)}

        # Compute convergence diagnostics
        self._compute_convergence_diagnostics()

    # Core statistical methods
    def mean(
        self, params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None
    ) -> float | npt.NDArray[np.float64]:
        """Posterior mean(s).

        Args:
            params: Parameter name, object, index, or list thereof. If None, return all means.

        Returns:
            Mean value(s).
            - Single parameter: returns float.
            - Collection of parameters: returns array.
        """
        # Validate that we have samples
        if len(self.samples) == 0:
            msg = "Cannot compute mean of empty samples"
            raise ValueError(msg)

        # If params is None, use all parameters (treat as collection)
        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            # Check if original input was a container before conversion
            was_container = is_container(params)

        indices = self._get_param_positions(params)

        # Validate indices
        if not indices:
            msg = "No valid parameters specified for mean calculation"
            raise ValueError(msg)

        # Select samples for the requested parameters
        samples_np = np.asarray(self.samples)
        selected_samples = samples_np[:, indices]
        means = np.mean(selected_samples, axis=0)

        # Single param not in container -> scalar, collection -> array
        if not was_container:
            return float(means[0])
        return means

    def symerr(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
        *,
        sigma: float | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Symmetric error (standard deviation) of posterior samples.

        Args:
            params: Parameter name, object, or index. If None, return all errors.
            sigma: Number of standard deviations. Default is 1. For example,
                   sigma=1 returns 1 standard deviation,
                   sigma=2 returns 2 standard deviations.

        Returns:
            Symmetric error(s) as float or array.
        """
        if sigma is None:
            sigma = 1

        # Convert to float
        try:
            sigma = float(sigma)
        except (TypeError, ValueError) as error:
            msg = f"sigma must be convertible to float, got {sigma}"
            raise TypeError(msg) from error

        if sigma <= 0:
            msg = f"sigma must be positive, got {sigma}"
            raise ValueError(msg)

        if sigma > 20:
            msg = f"A sigma value of {sigma} is larger than 20. This is not a realistic value and most likely a bug."
            raise ValueError(msg)

        return sigma * self.std(params=params)

    def std(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Standard deviation of posterior samples

        Args:
            params: Parameter name, object, index, or list thereof. If None, return all stds.

        Returns:
            Standard deviation(s).
            - Single parameter: returns float.
            - Collection of parameters: returns array.

        Examples:
            >>> result.std()  # All parameters
            array([0.102, 0.234])

            >>> result.std(['mu', 'sigma'])  # Multiple parameters
            array([0.102, 0.234])

            >>> result.std('mu')  # Single parameter
            0.102

            >>> result.std(['mu'])  # Single parameter in list
            array([0.102])
        """
        # Validate that we have samples
        if len(self.samples) == 0:
            msg = "Cannot compute standard deviation of empty samples"
            raise ValueError(msg)

        # If params is None, use all parameters (treat as collection)
        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            # Check if original input was a container before conversion
            was_container = is_container(params)

        indices = self._get_param_positions(params)

        # Validate indices
        if not indices:
            msg = "No valid parameters specified for std calculation"
            raise ValueError(msg)

        # Select samples for the requested parameters
        samples_np = np.asarray(self.samples)
        selected_samples = samples_np[:, indices]
        stds = np.std(selected_samples, axis=0)

        # Single param not in container -> scalar, collection -> array
        if not was_container:
            return float(stds[0])
        return stds

    def credible_interval(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
        *,
        alpha: float | None = None,
        sigma: float | None = None,
    ) -> tuple[float | npt.NDArray[np.float64], float | npt.NDArray[np.float64]]:
        """Equal-tailed credible interval(s).

        Args:
            params: Parameter name, object, index, or list thereof. If None, return all intervals.
            alpha: Significance level. Default is 0.05 for 95% interval.
            sigma: Number of standard deviations (e.g., 1 for ~68%, 2 for ~95%). Overrides alpha if given.

        Returns:
            Tuple (lower, upper).
            - Single parameter: returns tuple of floats.
            - Collection of parameters: returns tuple of arrays.
        """
        import scipy.stats  # noqa: PLC0415

        # Validate inputs
        if sigma is not None and alpha is not None:
            msg = "Cannot specify both sigma and alpha. Choose one."
            raise ValueError(msg)

        if sigma is not None:
            # Convert to float
            try:
                sigma = float(sigma)
            except (TypeError, ValueError) as error:
                msg = f"sigma must be convertible to float, got {sigma}"
                raise TypeError(msg) from error

            if sigma <= 0:
                msg = f"sigma must be positive, got {sigma}"
                raise ValueError(msg)

            # Convert sigma to two-tailed alpha using normal distribution
            alpha = 2 * (1 - scipy.stats.norm.cdf(sigma))
        elif alpha is None:
            alpha = 0.05
        else:
            # Convert to float
            try:
                alpha = float(alpha)
            except (TypeError, ValueError) as error:
                msg = f"alpha must be convertible to float, got {alpha}"
                raise TypeError(msg) from error

            if not 0 < alpha < 1:
                msg = f"alpha must be between 0 and 1, got {alpha}"
                raise ValueError(msg)

        # Vectorized percentile calculation
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        # If params is None, use all parameters (treat as collection)
        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            # Check if original input was a container before conversion
            was_container = is_container(params)

        # Handle single parameter or list of parameters
        indices = self._get_param_positions(params)
        if not indices:
            msg = "No parameters provided for credible interval calculation"
            raise ValueError(msg)

        # Extract samples for selected parameters
        samples_np = np.asarray(self.samples)
        selected_samples = samples_np[:, indices]

        # Calculate percentiles
        lowers = np.percentile(selected_samples, lower_percentile, axis=0)
        uppers = np.percentile(selected_samples, upper_percentile, axis=0)

        # Single param not in container -> scalar tuple, collection -> array tuple
        if not was_container:
            return float(lowers[0]), float(uppers[0])
        return lowers, uppers

    def get_samples(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Get posterior samples.

        Args:
            params: Parameter name, object, index, or list thereof. If None, return all samples.

        Returns:
            Array of samples.
            - Single parameter: returns 1D array.
            - Collection of parameters: returns 2D array with shape (n_samples, n_params).
        """
        # Validate that we have samples
        if len(self.samples) == 0:
            msg = "Cannot get samples from empty posterior"
            raise ValueError(msg)

        # If params is None, use all parameters (treat as collection)
        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            # Check if original input was a container before conversion
            was_container = is_container(params)

        indices = self._get_param_positions(params)

        # Validate indices
        if not indices:
            msg = "No valid parameters specified"
            raise ValueError(msg)

        # Convert to numpy for indexing
        samples_np = np.asarray(self.samples)

        # Single param not in container -> 1D, collection -> 2D
        if not was_container:
            return samples_np[:, indices[0]]
        return samples_np[:, indices]

    def as_prior(self, param: str | ZfitParameter | int) -> KDE:
        """Get posterior samples as a KDE prior for hierarchical modeling.

        Args:
            param: Parameter name, object, or index to get posterior for.

        Returns:
            KDE prior created from posterior samples.
        """
        # Validate param is not None
        if param is None:
            msg = "param cannot be None. Must specify a single parameter."
            raise ValueError(msg)

        # Validate param is a single parameter (not a list)
        if is_container(param) and len(param) > 1:
            msg = "as_prior() only supports single parameter, not multiple parameters"
            raise ValueError(msg)

        samples = self.get_samples(param)

        # Import here to avoid circular imports
        from .priors import KDE  # noqa: PLC0415

        param_name = param if isinstance(param, str) else param.name if hasattr(param, "name") else f"param_{param}"
        return KDE(samples, name=f"{param_name}_posterior_prior")

    # ArviZ integration
    def to_arviz(self) -> az.InferenceData:
        """Convert to ArviZ InferenceData format.

        Returns:
            ArviZ InferenceData object for advanced analysis.
        """
        try:
            import arviz as az  # noqa: PLC0415
        except ImportError as error:
            msg = "ArviZ is required for to_arviz(). Install with 'pip install arviz'."
            raise ImportError(msg) from error

        # Get samples as numpy array
        samples_np = self.samples

        # Determine chain and draw dimensions
        total_samples = len(samples_np)

        # Try to infer nwalkers from sampler
        if hasattr(self._sampler, "nwalkers") and self._sampler.nwalkers is not None:
            nwalkers = self._sampler.nwalkers
            ndraws = total_samples // nwalkers
        else:
            # Default to single chain
            nwalkers = 1
            ndraws = total_samples

        # Reshape samples for ArviZ (chain, draw, parameter)
        if nwalkers > 1:
            samples_reshaped = np.reshape(samples_np, (nwalkers, ndraws, -1))
        else:
            samples_reshaped = samples_np[np.newaxis, :, :]  # Add chain dimension

        # Use az.from_dict for simpler conversion
        return az.from_dict(
            {param.name: samples_reshaped[:, :, i] for i, param in enumerate(self._params)},
            coords={"chain": range(nwalkers), "draw": range(ndraws)},
        )

    # Parameter management
    def update_params(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
        *,
        what: str | None = None,
    ) -> PosteriorSamples:
        """Set all parameters to their posterior mean values."""
        if what is None:
            what = "mean"

        # Validate 'what' parameter
        valid_options = ["mean"]  # Can be extended in future
        if what not in valid_options:
            msg = f"Invalid 'what' option: {what}. Valid options are: {valid_options}"
            raise ValueError(msg)

        if what == "mean":
            return self._set_params_to_mean(params)
        msg = "This should never be reached, internal error."
        raise AssertionError(msg)

    def _set_params_to_mean(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> PosteriorSamples:
        """Set parameters to their posterior mean values."""
        if params is None:
            params = self._params

        # Convert params to container and get indices
        params = convert_to_container(params)
        indices = self._get_param_positions(params)
        means = self.mean(params)

        # Get actual parameter objects from indices
        param_objects = [self._params[idx] for idx in indices]

        # Set parameter values
        if len(param_objects) == 1:
            param_objects[0].set_value(float(means))
        else:
            for param_obj, mean_val in zip(param_objects, means):
                param_obj.set_value(float(mean_val))

        return self  # todo: improve, refactor?

    def __enter__(self):
        """Context manager: set parameters to posterior means."""
        self._old_values = [param.value() for param in self._params]
        self.update_params()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: restore original parameter values."""
        from zfit.core.parameter import set_values  # noqa: PLC0415

        set_values(self._params, self._old_values)

    # Essential properties
    @property
    def params(self) -> list[ZfitParameter]:
        """Parameters used in the sampling."""
        return self._params

    @property
    def param_names(self) -> list[str]:
        """Names of the parameters used in the sampling."""
        return [param.name for param in self._params]

    @property
    def sampler(self) -> MCMCSampler:
        """Sampler used to generate samples."""
        return self._sampler

    @property
    def loss(self) -> ZfitLoss:
        """Loss function that was sampled."""
        return self._loss

    @property
    def valid(self) -> bool:
        """Whether the MCMC results are valid (no NaN/inf values)."""
        return self._valid

    @property
    def converged(self) -> bool:
        """Whether the MCMC chains have converged based on diagnostics.

        Convergence is determined by:
        - R-hat < 1.1 for all parameters (Gelman-Rubin statistic)
        - Effective sample size > 100 for all parameters
        - No NaN or infinite values
        """
        return self._converged

    def covariance(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Covariance matrix from posterior samples.

        Args:
            params: Parameters to include. If None, use all parameters.

        Returns:
            Covariance matrix as numpy array.
            - Single parameter: returns scalar (variance).
            - Collection of parameters: returns matrix.
        """
        # Validate that we have samples
        if len(self.samples) == 0:
            msg = "Cannot compute covariance of empty samples"
            raise ValueError(msg)

        # Need at least 2 samples for covariance
        if len(self.samples) < 2:
            msg = f"Need at least 2 samples to compute covariance, got {len(self.samples)}"
            raise ValueError(msg)

        # If params is None, use all parameters (treat as collection)
        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            # Check if original input was a container before conversion
            was_container = is_container(params)

        indices = self._get_param_positions(params)

        # Validate indices
        if not indices:
            msg = "No valid parameters specified for covariance calculation"
            raise ValueError(msg)

        # Select columns for specified parameters
        samples_np = np.asarray(self.samples)
        selected_samples = samples_np[:, indices]

        # Single param not in container -> scalar variance, collection -> matrix
        if len(indices) == 1 and not was_container:
            variance = np.var(selected_samples, axis=0, ddof=1)
            return float(variance[0]) if variance.ndim > 0 else float(variance)

        # For collection of parameters, return covariance matrix
        cov_matrix = np.cov(selected_samples, rowvar=False)
        return np.atleast_2d(cov_matrix)

    # Utility methods
    def summary(self, round_to: int | None = None) -> pd.DataFrame:
        """Summary statistics using ArviZ when available.

        Args:
            round_to: Number of decimals to round to. If None, no rounding.

        Returns:
            ArviZ summary DataFrame.
        """
        import arviz as az  # noqa: PLC0415

        idata = self.to_arviz()

        # Use ArviZ summary for comprehensive statistics
        return az.summary(idata, round_to=round_to)

    def _get_param_positions(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter],
    ) -> list[int]:
        """Get parameter positions in the samples array from names or objects.

        Args:
            params: Single parameter or collection of parameters.
                   Can be parameter name(s) or object(s).

        Returns:
            List of parameter positions in the samples array.
        """
        # Check for invalid types before conversion
        if isinstance(params, dict):
            msg = f"Invalid parameter type: {type(params)}"
            raise TypeError(msg)

        # Convert single parameter to list
        params = convert_to_container(params)

        positions = []
        for param in params:
            if isinstance(param, str):
                if param not in self._param_by_name:
                    msg = f"Parameter '{param}' not found"
                    raise ValueError(msg)
                positions.append(self._position_by_name[param])
            elif isinstance(param, zfit.Parameter):
                if param not in self._name_by_param:
                    msg = f"Parameter {param} not found in posterior samples"
                    raise ValueError(msg)
                positions.append(self._position_by_name[param.name])
            else:
                msg = (
                    f"Invalid parameter type: {type(param)}. Expected string (parameter name) or ZfitParameter object."
                )
                raise TypeError(msg)

        return positions

    def __repr__(self) -> str:
        return f"PosteriorSamples(n_samples={len(self.samples)}, params={[param.name for param in self._params]})"

    def __str__(self) -> str:
        """Nice string representation of posterior results."""
        import colored  # noqa: PLC0415
        from colorama import Style  # noqa: PLC0415
        from tabulate import tabulate  # noqa: PLC0415

        # Header
        string = Style.BRIGHT + "PosteriorSamples" + Style.NORMAL + f" from\n{self.loss} \nwith\n{self.sampler}\n\n"

        # Convergence summary table
        def color_on_bool(value, on_true=None, on_false=None):
            """Color boolean values.

            Args:
                value: Boolean value to color
                on_true: Color for True values. Defaults to green background.
                on_false: Color for False values. Defaults to red background.
            """
            if on_true is None:
                on_true = colored.bg("green")
            if on_false is None:
                on_false = colored.bg("red")
            if on_false is False:
                on_false = ""
            text = "True" if value else "False"
            color = on_true if value else on_false
            return f"{color}{text}{Style.RESET_ALL}"

        # Main diagnostics table
        rhat_str = "N/A (single chain)"
        if self._rhat is not None:
            max_rhat = np.max(self._rhat)
            rhat_str = f"{max_rhat:.4f}"
            if max_rhat > 1.1:
                rhat_str = colored.fg("red") + rhat_str + Style.RESET_ALL

        ess_str = "N/A"
        if self._ess is not None:
            min_ess = np.min(self._ess)
            ess_str = f"{min_ess:.0f}"
            if min_ess < 100:
                ess_str = colored.fg("red") + ess_str + Style.RESET_ALL

        string += tabulate(
            [
                [
                    color_on_bool(self.valid),
                    color_on_bool(self.converged, on_true=colored.bg("green"), on_false=colored.bg("yellow")),
                    rhat_str,
                    ess_str,
                    f"{len(self.samples):>13} | {self.n_warmup:>6} | {self.n_samples:>10}",
                ]
            ],
            [
                "valid",
                "converged",
                "max R̂",
                "min ESS",
                "total samples | warmup | per walker",
            ],
            tablefmt="fancy_grid",
            disable_numparse=True,
            colalign=["center", "center", "center", "center", "right"],
        )

        # Parameters table
        string += "\n\n" + Style.BRIGHT + "Parameters\n" + Style.NORMAL

        param_data = []

        # First pass: collect all values to determine optimal formatting widths
        means = self.mean()
        stds = self.std()
        lower, upper = self.credible_interval(alpha=0.05)

        # Determine the width needed for credible intervals
        all_ci_values = []
        for i in range(len(self._params)):
            ci_lower = lower[i] if hasattr(lower, "__len__") else lower
            ci_upper = upper[i] if hasattr(upper, "__len__") else upper
            all_ci_values.extend([ci_lower, ci_upper])

        max_abs_ci = max(abs(val) for val in all_ci_values)
        if max_abs_ci >= 1000:
            ci_width = 10  # For large numbers like n_sig, n_bkg
        elif max_abs_ci >= 10:
            ci_width = 8  # For moderate numbers
        else:
            ci_width = 7  # For small numbers like mu, sigma

        # Second pass: format data with proper alignment
        for i, param in enumerate(self._params):
            param_name = param.name
            mean_val = means[i]
            std_val = stds[i]
            ci_lower = lower[i] if hasattr(lower, "__len__") else lower
            ci_upper = upper[i] if hasattr(upper, "__len__") else upper

            # R-hat and ESS for this parameter
            rhat_param = f"{self._rhat[i]:.3f}" if self._rhat is not None else "N/A"
            ess_param = f"{self._ess[i]:.0f}" if self._ess is not None else "N/A"

            # Format credible interval with proper alignment
            ci_formatted = f"[{ci_lower:>{ci_width}.4f}, {ci_upper:>{ci_width}.4f}]"

            param_data.append(
                [
                    param_name,
                    f"{mean_val:>8.4f}",
                    f"± {std_val:.4f}",
                    ci_formatted,
                    rhat_param,
                    ess_param,
                ]
            )

        string += tabulate(
            param_data,
            headers=["parameter", "mean", "std", "95% CI", "R̂", "ESS"],
            tablefmt="simple",
            floatfmt=".4f",
            colalign=["left", "right", "right", "right", "right", "right"],
        )

        # Additional info
        if hasattr(self._sampler, "nwalkers"):
            string += f"\n\nSampler: {self._sampler.__class__.__name__} with {self._sampler.nwalkers} walkers"

        return string

    def _repr_pretty_(self, p, cycle):
        """IPython/Jupyter pretty display."""
        if cycle:
            p.text(self.__repr__())
            return
        p.text(self.__str__())

    def _compute_convergence_diagnostics(self):
        """Compute MCMC convergence diagnostics."""
        # Check for valid samples (no NaN/inf)
        self._valid = not (np.any(np.isnan(self.samples)) or np.any(np.isinf(self.samples)))

        if not self._valid:
            self._converged = False
            self._rhat = None
            self._ess = None
            return

        # Try to use ArviZ for better diagnostics if available
        import arviz as az  # noqa: PLC0415

        idata = self.to_arviz()

        # Compute R-hat and ESS for all parameters at once
        rhat_data = az.rhat(idata)
        ess_data = az.ess(idata)

        # Extract values for each parameter
        self._rhat = np.array([rhat_data[param.name].values for param in self._params])
        self._ess = np.array([ess_data[param.name].values for param in self._params])

        # Check convergence criteria
        rhat_converged = bool(np.all(self._rhat < 1.1))
        ess_converged = bool(np.all(self._ess > 100))
        self._converged = rhat_converged and ess_converged

    @property
    def rhat(self) -> npt.NDArray[np.float64] | None:
        """Gelman-Rubin R-hat convergence diagnostic.

        Values < 1.1 indicate good convergence.
        Only available when multiple chains are used.
        """
        return self._rhat

    @property
    def ess(self) -> npt.NDArray[np.float64] | None:
        """Effective sample size for each parameter.

        Accounts for autocorrelation in MCMC chains.
        Higher values indicate more independent samples.
        """
        return self._ess

    def convergence_summary(self) -> dict:
        """Summary of convergence diagnostics."""
        import arviz as az  # noqa: PLC0415

        idata = self.to_arviz()

        # Use ArviZ for comprehensive diagnostics
        return {
            "valid": self._valid,
            "converged": self._converged,
            "rhat": az.rhat(idata).to_dict(),
            "ess_bulk": az.ess(idata, method="bulk").to_dict(),
            "ess_tail": az.ess(idata, method="tail").to_dict(),
            "mcse_mean": az.mcse(idata, method="mean").to_dict(),
            "mcse_sd": az.mcse(idata, method="sd").to_dict(),
        }

    def diagnostics(self) -> dict:
        """Comprehensive diagnostics report.

        Returns:
            Dictionary with all available diagnostics.
        """

        import arviz as az  # noqa: PLC0415  # noqa

        idata = self.to_arviz()

        return {
            "valid": self.valid,
            "converged": self.converged,
            "summary": az.summary(idata),
            "rhat": az.rhat(idata),
            "ess_bulk": az.ess(idata, method="bulk"),
            "ess_tail": az.ess(idata, method="tail"),
            "mcse_mean": az.mcse(idata, method="mean"),
            "mcse_sd": az.mcse(idata, method="sd"),
            "loo": None,  # Placeholder for future LOO-CV integration
        }
