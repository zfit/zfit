"""Dynesty nested sampler for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf

import zfit.param
from zfit.core.interfaces import ZfitObject

from ..results import Posteriors


class DynestySampler(ZfitObject):
    """Nested sampler using Dynesty (https://dynesty.readthedocs.io/)."""

    def __init__(self, nlive=500, bound="multi", samplemethod="auto", dlogz=0.01, dynamic=False, name="DynestySampler"):
        """Initialize a DynestySampler.

        Args:
            nlive: Number of live points
            bound: Method used to approximately bound the prior ("single", "multi", "balls", "cubes")
            samplemethod: Method used to sample uniformly within the likelihood constraint
            dlogz: Iteration will stop when the estimated contribution to the total evidence
                  falls below this threshold
            dynamic: Whether to use dynamic nested sampling
            name: Name of the sampler
        """
        self.nlive = nlive
        self.bound = bound
        self.samplemethod = samplemethod
        self.dlogz = dlogz
        self.dynamic = dynamic
        self.name = name

    def sample(self, loss, params=None, n_samples=None, seed=None, **kwargs):
        """Sample from the posterior distribution using Dynesty.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of samples to draw from the posterior (not used directly
                       in nested sampling, but provided to ensure consistent API)
            seed: Random seed for reproducibility
            **kwargs: Additional Dynesty-specific arguments

        Returns:
            A Posterior object
        """
        try:
            import dynesty
        except ImportError:
            msg = "Dynesty is required. Install with 'pip install dynesty'."
            raise ImportError(msg)

        # Import here to avoid circular imports
        if params is None:
            params = loss.get_params(floating=True)

        n_dims = len(params)
        param_names = [param.name for param in params]

        # Define prior transform function
        def prior_transform(u):
            """Transform unit cube to parameter space according to priors."""
            theta = np.zeros(n_dims)

            for i, param in enumerate(params):
                # Get prior if it exists
                prior = getattr(param, "get_prior", lambda: None)()

                # If no explicit prior, use parameter limits
                if prior is None:
                    if hasattr(param, "has_limits") and param.has_limits:
                        lower = param.lower if hasattr(param, "lower") and param.lower is not None else -np.inf
                        upper = param.upper if hasattr(param, "upper") and param.upper is not None else np.inf

                        # Convert infinite limits to something reasonable
                        if np.isinf(lower):
                            # Default to a reasonable range centered on current value
                            val = param.value()
                            lower = val - 10.0 * abs(val) if val != 0 else -10.0

                        if np.isinf(upper):
                            val = param.value()
                            upper = val + 10.0 * abs(val) if val != 0 else 10.0

                        # Transform from unit cube to parameter space
                        theta[i] = lower + (upper - lower) * u[i]
                    else:
                        # Without limits, default to a normal distribution around current value
                        val = param.value()
                        scale = abs(val) / 5.0 if val != 0 else 1.0
                        # Use inverse CDF of normal distribution
                        from scipy import stats

                        theta[i] = stats.norm.ppf(u[i], loc=val, scale=scale)
                # TODO: Implement proper prior transforms for different prior types
                # Currently just fall back to uniform within limits
                elif hasattr(param, "has_limits") and param.has_limits:
                    lower = param.lower if hasattr(param, "lower") and param.lower is not None else -np.inf
                    upper = param.upper if hasattr(param, "upper") and param.upper is not None else np.inf

                    if np.isinf(lower) or np.isinf(upper):
                        # Default to a reasonable range
                        val = param.value()
                        if np.isinf(lower):
                            lower = val - 10.0 * abs(val) if val != 0 else -10.0
                        if np.isinf(upper):
                            upper = val + 10.0 * abs(val) if val != 0 else 10.0

                    theta[i] = lower + (upper - lower) * u[i]
                else:
                    # Default behavior
                    val = param.value()
                    scale = abs(val) / 5.0 if val != 0 else 1.0
                    from scipy import stats

                    theta[i] = stats.norm.ppf(u[i], loc=val, scale=scale)

            return theta

        # Define log-likelihood function
        def log_likelihood(x):
            """Calculate log likelihood (negative of loss)."""
            # Store original values to restore later
            return log_likelihood_jit(x)

        @tf.function(autograph=False)
        def log_likelihood_jit(x):
            zfit.param.assign_values(params, x)
            return -(loss.value())

        # Set up and run the sampler
        if seed is not None:
            np.random.seed(seed)

        print(f"Running nested sampling with dynesty (nlive={self.nlive})...")

        # Run the nested sampling
        if self.dynamic:
            sampler = dynesty.DynamicNestedSampler(
                log_likelihood,
                prior_transform,
                n_dims,
                nlive=self.nlive,
                bound=self.bound,
                sample=self.sample,
                **kwargs,
            )
            sampler.run_nested(dlogz_init=self.dlogz)
        else:
            sampler = dynesty.NestedSampler(
                log_likelihood,
                prior_transform,
                n_dims,
                nlive=self.nlive,
                bound=self.bound,
                sample=self.sample,
                **kwargs,
            )
            sampler.run_nested(dlogz=self.dlogz)

        # Get results and posterior samples
        results = sampler.results

        # If n_samples is specified, draw that many samples from the posterior
        # Otherwise, use dynesty's default
        if n_samples is not None:
            samples = results.samples_equal(size=n_samples)
        else:
            # Use dynesty's resample_equal method to get posterior samples
            weights = np.exp(results.logwt - results.logz[-1])
            samples = dynesty.utils.resample_equal(results.samples, weights)
            n_samples = len(samples)

        # Calculate evidence and error
        logz = results.logz[-1]
        logzerr = results.logzerr[-1]
        print(f"log(Z) = {logz:.2f} ± {logzerr:.2f}")

        # Create Posterior object
        return Posteriors(
            samples=samples,
            param_names=param_names,
            params=params,
            loss=loss,
            sampler=self,
            n_warmup=self.nlive,  # Not exactly warmup but similar concept
            n_samples=n_samples,
            raw_result=results,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(nlive={self.nlive}, dynamic={self.dynamic}, name='{self.name}')"
