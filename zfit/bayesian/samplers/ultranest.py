"""UltraNest sampler for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np

from zfit.core.interfaces import ZfitObject

from ..results import Posteriors


class UltraNestSampler(ZfitObject):
    """Nested sampler using UltraNest (https://johannesbuchner.github.io/UltraNest/)."""

    def __init__(
        self,
        min_num_live_points=400,
        cluster_num_live_points=40,
        dlogz=0.5,
        update_interval_volume_fraction=0.8,
        log_dir=None,
        resume=True,
        name="UltraNestSampler",
    ):
        """Initialize an UltraNest sampler.

        Args:
            min_num_live_points: Minimum number of live points
            cluster_num_live_points: Number of live points per detected cluster
            dlogz: Target evidence uncertainty
            update_interval_volume_fraction: How often to update visualization
            log_dir: Directory for storing logs and output files
            resume: Whether to resume previous runs
            name: Name of the sampler
        """
        self.min_num_live_points = min_num_live_points
        self.cluster_num_live_points = cluster_num_live_points
        self.dlogz = dlogz
        self.update_interval_volume_fraction = update_interval_volume_fraction
        self.log_dir = log_dir
        self.resume = resume
        self.name = name

    def sample(self, loss, params=None, n_samples=None, seed=None, **kwargs):
        """Sample from the posterior distribution using UltraNest.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of posterior samples to draw (not used directly in nested sampling)
            seed: Random seed for reproducibility
            **kwargs: Additional UltraNest-specific arguments

        Returns:
            A Posterior object
        """
        try:
            import ultranest
        except ImportError:
            msg = "UltraNest is required. Install with 'pip install ultranest'."
            raise ImportError(msg)

        # Import here to avoid circular imports
        if params is None:
            params = loss.get_params(floating=True)

        n_dims = len(params)
        param_names = [param.name for param in params]

        # Create parameter ranges
        param_ranges = []
        for param in params:
            if hasattr(param, "has_limits") and param.has_limits:
                lower = param.lower if hasattr(param, "lower") and param.lower is not None else -np.inf
                upper = param.upper if hasattr(param, "upper") and param.upper is not None else np.inf

                # Convert infinite limits to something reasonable
                if np.isinf(lower):
                    val = param.value()
                    lower = val - 10.0 * abs(val) if val != 0 else -10.0

                if np.isinf(upper):
                    val = param.value()
                    upper = val + 10.0 * abs(val) if val != 0 else 10.0

                param_ranges.append([float(lower), float(upper)])
            else:
                # Create a reasonable range around current value
                val = param.value()
                if val != 0:
                    param_ranges.append([val - 10.0 * abs(val), val + 10.0 * abs(val)])
                else:
                    param_ranges.append([-10.0, 10.0])

        # Define prior transform function
        def prior_transform(cube):
            """Transform unit cube to parameter space."""
            params_transformed = np.zeros(n_dims)

            for i, (lower, upper) in enumerate(param_ranges):
                params_transformed[i] = lower + (upper - lower) * cube[i]

            return params_transformed

        # Define log-likelihood function
        def log_likelihood(x):
            """Calculate log likelihood (negative of loss)."""
            # Store original values to restore later
            original_values = [p.value() for p in params]

            try:
                # Set parameter values
                for i, param in enumerate(params):
                    param.set_value(x[i])

                # Calculate log likelihood (negative of loss)
                return -float(loss.value())
            except Exception as e:
                # In case of errors (e.g. out of bounds), return very negative log-likelihood
                print(f"Warning: Error in log_likelihood: {e}")
                return -1e300
            finally:
                # Restore original values
                for param, val in zip(params, original_values):
                    param.set_value(val)

        # Set up and run the sampler
        ultranest_kwargs = {
            # "resume": self.resume,
            # "min_num_live_points": self.min_num_live_points,
            # "cluster_num_live_points": self.cluster_num_live_points,
            # "dlogz": self.dlogz,
            # "update_interval_volume_fraction": self.update_interval_volume_fraction,
        }

        if seed is not None:
            ultranest_kwargs["seed"] = seed

        # Merge user kwargs with defaults
        ultranest_kwargs.update(kwargs)

        # Create and run the sampler
        sampler = ultranest.ReactiveNestedSampler(
            param_names, log_likelihood, prior_transform, log_dir=self.log_dir, **ultranest_kwargs
        )

        print("Running nested sampling with UltraNest...")
        result = sampler.run(**ultranest_kwargs)

        # Get results
        logz = result["logz"]
        logzerr = result["logzerr"]
        print(f"log(Z) = {logz:.2f} ± {logzerr:.2f}")

        # Get weighted posterior samples
        weighted_samples = result["weighted_samples"]
        weights = weighted_samples["weights"]
        samples_raw = weighted_samples["points"]

        # If n_samples is specified, draw that many samples from the posterior
        # Otherwise, use all available samples
        if n_samples is not None:
            # Draw samples according to weights
            indices = np.random.choice(len(weights), size=n_samples, replace=True, p=weights / np.sum(weights))
            samples = samples_raw[indices]
        # Use equal-weight posterior samples from UltraNest
        elif "samples" in result:
            samples = result["samples"]
            n_samples = len(samples)
        else:
            # Resample manually with weights (less accurate)
            indices = np.random.choice(len(weights), size=len(weights), replace=True, p=weights / np.sum(weights))
            samples = samples_raw[indices]
            n_samples = len(samples)

        # Create Posterior object
        return Posteriors(
            samples=samples,
            param_names=param_names,
            params=params,
            loss=loss,
            sampler=self,
            n_warmup=self.min_num_live_points,  # Not exactly warmup but similar concept
            n_samples=n_samples,
            raw_result=result,
            metadata={
                "logz": logz,
                "logzerr": logzerr,
                "param_ranges": param_ranges,
            },
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(min_live_points={self.min_num_live_points}, name='{self.name}')"
