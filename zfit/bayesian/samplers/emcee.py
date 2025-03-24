"""MCMC samplers for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import numpy as np

import zfit.param
import zfit.z
from zfit.core.interfaces import ZfitObject


class ZfitSampler(ZfitObject):
    """Base class for MCMC samplers in Bayesian inference."""

    def __init__(self, name=None):
        """Initialize a sampler.

        Args:
            name: Optional name for the sampler
        """
        self.name = name

    def sample(self, loss, params=None, n_samples=1000, n_warmup=100, **kwargs):
        """Sample from the posterior distribution.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of posterior samples to generate
            n_warmup: Number of warmup/burn-in steps
            **kwargs: Additional sampler-specific arguments

        Returns:
            A Posterior object
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


import tensorflow as tf

import zfit.z.numpy as znp


class EmceeSampler(ZfitSampler):
    """MCMC sampler using emcee (https://emcee.readthedocs.io)."""

    def __init__(self, n_walkers=None, name="EmceeSampler"):
        """Initialize an EmceeSampler.

        Args:
            n_walkers: Number of walkers to use. If None, will use 2 * n_dims
            name: Name of the sampler
        """
        super().__init__(name=name)
        self.n_walkers = n_walkers

    def sample(self, loss, params=None, n_samples=1000, n_warmup=100, **kwargs):
        """Sample from the posterior distribution using emcee.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of posterior samples to generate
            n_warmup: Number of warmup/burn-in steps
            **kwargs: Additional emcee-specific arguments

        Returns:
            A Posterior object
        """
        try:
            import emcee
        except ImportError:
            msg = "emcee is required for EmceeSampler. Install with 'pip install emcee'."
            raise ImportError(msg)

        # Import here to avoid circular imports
        from zfit.bayesian.results import Posteriors
        from zfit.core.interfaces import ZfitLoss

        if not isinstance(loss, ZfitLoss):
            msg = f"loss must be a ZfitLoss instance, not {type(loss)}"
            raise TypeError(msg)

        if params is None:
            params = loss.get_params(floating=True)

        n_dims = len(params)
        if self.n_walkers is None:
            self.n_walkers = max(2 * n_dims, 100)  # At least 100 walkers
        params = list(params)

        # @zfit.z.function
        def calculate_priors(x):
            # Calculate log prior
            return sum(getattr(param, "log_prior", lambda: 0.0)() for param in params)

        # Define the log probability function
        def log_prob(x):
            x = znp.asarray(x)
            for i, param in enumerate(params):
                if param.lower is not None and x[i] < param.lower:
                    return -np.inf
                if param.upper is not None and x[i] > param.upper:
                    return -np.inf
            return log_prob_jit(x)

        @tf.function(autograph=False)
        def log_prob_jit(x):
            # with zfit.param.set_values(params, x):
            # Calculate log likelihood (negative of loss)

            zfit.param.assign_values_jit(params, x)
            log_likelihood = -loss.value()
            log_prior = calculate_priors(x)
            return log_likelihood + log_prior

        # Initialize walkers around the current parameter values
        if not all(isinstance(p, zfit.param.Parameter) for p in params):
            msg = "params must be a sequence of ZfitParameter objects"
            raise TypeError(msg)

        initial_positions = np.array([param.value() for param in params])
        pos = initial_positions + 1e-4 * np.random.randn(self.n_walkers, n_dims)

        # For scale parameters, make sure we stay positive
        for i, param in enumerate(params):
            # Check for parameter limits
            if (lower := param.lower) is not None:
                pos[:, i] = np.maximum(pos[:, i], lower + 1e-5)
            if (upper := param.upper) is not None:
                pos[:, i] = np.minimum(pos[:, i], upper - 1e-5)

        # Set up sampler
        sampler = emcee.EnsembleSampler(self.n_walkers, n_dims, log_prob, **kwargs)

        # Run burn-in
        if n_warmup > 0:
            print(f"Running burn-in phase with {n_warmup} steps...")
            state = sampler.run_mcmc(pos, n_warmup, progress=True)
            sampler.reset()
        else:
            state = pos

        # Run production
        print(f"Running production phase with {n_samples} steps...")
        sampler.run_mcmc(state, n_samples, progress=True)

        # Create result object
        samples = sampler.get_chain(flat=True)
        # Take only the requested number of samples
        samples = samples[-n_samples:]
        param_names = [param.name for param in params]

        return Posteriors(
            samples=samples,
            param_names=param_names,
            params=params,
            loss=loss,
            sampler=self,
            n_warmup=n_warmup,
            n_samples=n_samples,
            raw_result=sampler,
        )
