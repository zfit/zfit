"""Parallel Tempering MCMC sampler for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import zfit.param
import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitObject

from ..results import Posteriors


class PTSampler(ZfitObject):
    """Parallel Tempering MCMC sampler using ptemcee (https://github.com/willvousden/ptemcee)."""

    def __init__(
        self,
        nwalkers=None,
        ntemps=10,
        betas=None,
        adaptation_lag=10000,
        adaptation_time=100,
        scale_factor=None,
        name="PTSampler",
    ):
        """Initialize a Parallel Tempering sampler.

        Args:
            nwalkers: Number of walkers per temperature. If None, will use 2 * n_dims
            ntemps: Number of temperature ladders for parallel tempering
            betas: Temperature ladder. If provided, ntemps is ignored
            adapt: Whether to adapt the temperature ladder during sampling
            adaptation_lag: Number of iterations to wait for before starting adaptation
            adaptation_time: Number of iterations to consider for adaptation
            scale_factor: Scale factor for proposal distribution
            name: Name of the sampler
        """
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.betas = betas
        self.adaptation_lag = adaptation_lag
        self.adaptation_time = adaptation_time
        self.scale_factor = scale_factor
        self.name = name

    def sample(self, loss, params=None, n_samples=1000, n_warmup=100, seed=None, **kwargs):
        """Sample from the posterior distribution using Parallel Tempering MCMC.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of posterior samples to generate per walker
            n_warmup: Number of warmup/burn-in steps per walker
            seed: Random seed for reproducibility
            **kwargs: Additional sampler-specific arguments

        Returns:
            A Posterior object
        """
        try:
            import ptemcee
        except ImportError:
            msg = "ptemcee is required. Install with 'pip install ptemcee'."
            raise ImportError(msg)

        # Import here to avoid circular imports
        if params is None:
            params = loss.get_params(floating=True)

        n_dims = len(params)
        param_names = [param.name for param in params]

        # Set number of walkers if not specified
        if self.nwalkers is None:
            self.nwalkers = max(2 * n_dims, 20)  # At least 20 walkers

        # Define the log probability function
        def log_prob(x):
            x = znp.asarray(x)
            return log_prob_jit(x)

            # Calculate log prior

        @tf.function(autograph=False)
        def log_prob_jit(x):
            zfit.param.assign_values(params, x)
            return -(loss.value())

        def log_prior(x):
            x = znp.asarray(x)
            return log_prior_jit(x)

        @tf.function(autograph=False)
        def log_prior_jit(x):
            zfit.param.assign_values(params, x)
            return znp.sum([(getattr(param, "log_prior", lambda: 0.0)()) for param in params])

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize walkers
        initial_position = np.array([param.value() for param in params])

        # Create walker positions with small random displacements
        pos = initial_position + 1e-4 * np.random.randn(self.ntemps, self.nwalkers, n_dims)

        # Apply parameter limits
        for i, param in enumerate(params):
            if hasattr(param, "has_limits") and param.has_limits:
                if hasattr(param, "lower") and param.lower is not None:
                    lower = param.lower
                    pos[:, :, i] = np.maximum(pos[:, :, i], lower + 1e-5)
                if hasattr(param, "upper") and param.upper is not None:
                    upper = param.upper
                    pos[:, :, i] = np.minimum(pos[:, :, i], upper - 1e-5)

        # Create the sampler
        sampler_kwargs = {}
        if self.betas is not None:
            sampler_kwargs["betas"] = self.betas
        if self.scale_factor is not None:
            sampler_kwargs["a"] = self.scale_factor

        # Add any additional kwargs
        sampler_kwargs.update(kwargs)

        # Create parallel tempering sampler
        sampler = ptemcee.Sampler(
            nwalkers=self.nwalkers,
            dim=n_dims,
            logp=log_prob,
            logl=log_prior,
            ntemps=self.ntemps,
            adaptation_lag=self.adaptation_lag,
            adaptation_time=self.adaptation_time,
            **sampler_kwargs,
        )

        # Run burn-in
        if n_warmup > 0:
            print(f"Running burn-in phase with {n_warmup} steps per walker...")
            for _ in tqdm(range(n_warmup), desc="Burn-in"):
                pos, _, _ = sampler.run_mcmc(pos, 1)

            # Reset sampler to clear burn-in samples
            sampler.reset()

        # Run production
        print(f"Running production phase with {n_samples} steps per walker...")
        for _ in tqdm(range(n_samples), desc="Production"):
            pos, _, _ = sampler.run_mcmc(pos, 1)

        # Get samples from the coldest temperature (first index)
        # and flatten the walker dimension
        samples = sampler.chain[0, :, :, :].reshape(-1, n_dims)

        # Calculate acceptance rate
        accept_rate = np.mean(sampler.acceptance_fraction[0])
        print(f"Mean acceptance rate: {accept_rate:.3f}")

        # Create result object
        return Posteriors(
            samples=samples,
            param_names=param_names,
            params=params,
            loss=loss,
            sampler=self,
            n_warmup=n_warmup,
            n_samples=n_samples * self.nwalkers,
            raw_result=sampler,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(nwalkers={self.nwalkers}, ntemps={self.ntemps}, name='{self.name}')"
