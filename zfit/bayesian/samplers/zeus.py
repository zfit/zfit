"""Zeus sampler for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf

import zfit.param
import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitObject

from ..results import Posteriors


class ZeusSampler(ZfitObject):
    """MCMC sampler using Zeus (https://zeus-mcmc.readthedocs.io/).

    Zeus is a pure-Python implementation of the ensemble slice sampling method.
    """

    def __init__(
        self,
        nwalkers=None,
        tune=True,
        tolerance=0.05,
        patience=5,
        maxsteps=10000,
        mu=1.0,
        sigma_back=1.0e-4,
        vectorize=False,
        check_walkers=True,
        shuffle_ensemble=True,
        name="ZeusSampler",
    ):
        """Initialize a Zeus sampler.

        Args:
            nwalkers: Number of walkers (defaults to 2*ndim)
            tune: Whether to tune the walker ensemble
            tolerance: Tolerance for convergence check
            patience: Number of consecutive iterations below tolerance to consider chain converged
            maxsteps: Maximum steps for tuning
            mu: Hyperparameter controlling the size of reference slice
            sigma_back: Hyperparameter controlling backward jump size
            vectorize: Whether to vectorize the log probability evaluation
            check_walkers: Whether to check for walkers getting stuck
            shuffle_ensemble: Whether to shuffle the ensemble before slice sampling
            name: Name of the sampler
        """
        self.nwalkers = nwalkers
        self.tune = tune
        self.tolerance = tolerance
        self.patience = patience
        self.maxsteps = maxsteps
        self.mu = mu
        self.sigma_back = sigma_back
        self.vectorize = vectorize
        self.check_walkers = check_walkers
        self.shuffle_ensemble = shuffle_ensemble
        self.name = name

    def sample(self, loss, params=None, n_samples=1000, n_warmup=100, seed=None, **kwargs):
        """Sample from the posterior distribution using Zeus.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of posterior samples to generate per walker
            n_warmup: Number of warmup/burn-in steps per walker
            seed: Random seed for reproducibility
            **kwargs: Additional Zeus-specific arguments

        Returns:
            A Posterior object
        """
        try:
            import zeus
        except ImportError:
            msg = "Zeus is required. Install with 'pip install zeus-mcmc'."
            raise ImportError(msg)

        if params is None:
            params = loss.get_params(floating=True)

        n_dims = len(params)
        param_names = [param.name for param in params]

        # Set number of walkers if not specified
        if self.nwalkers is None:
            self.nwalkers = max(2 * n_dims, 100)  # At least 100 walkers

        # Define the log probability function
        def log_prob(x):
            x = znp.asarray(x)

            value = log_prob_jit(x)
            # print(f"NLL = {value} at x = {x}")
            return np.asarray(value)

        @tf.function(autograph=False)
        def log_prob_jit(x):
            zfit.param.set_values(params, x)
            # Calculate log likelihood (negative of loss)
            log_likelihood = -(loss.value())
            # Calculate log prior
            log_prior = znp.sum([getattr(param, "log_prior", lambda: znp.array(0.0))() for param in params])
            return znp.atleast_1d(log_likelihood + log_prior)

        # Create options dict for Zeus
        zeus_kwargs = {
            "tune": self.tune,
            "maxsteps": self.maxsteps,
            "patience": self.patience,
            "mu": self.mu,
            "tolerance": self.tolerance,
            "check_walkers": self.check_walkers,
            "shuffle_ensemble": self.shuffle_ensemble,
        }

        # Add any additional kwargs
        zeus_kwargs.update(kwargs)

        # Initialize walkers
        initial_position = np.array([param.value() for param in params])

        # Create walker positions with small random displacements
        pos = initial_position + 1e-4 * np.random.randn(self.nwalkers, n_dims)

        # Apply parameter limits
        for i, param in enumerate(params):
            if hasattr(param, "has_limits") and param.has_limits:
                if hasattr(param, "lower") and param.lower is not None:
                    lower = param.lower
                    pos[:, i] = np.maximum(pos[:, i], lower + 1e-5)
                if hasattr(param, "upper") and param.upper is not None:
                    upper = param.upper
                    pos[:, i] = np.minimum(pos[:, i], upper - 1e-5)

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Create Zeus sampler
        print("Using Zeus ensemble slice sampler...")
        sampler = zeus.EnsembleSampler(
            self.nwalkers, n_dims, logprob_fn=log_prob, vectorize=self.vectorize, **zeus_kwargs
        )

        # Run burn-in
        if n_warmup > 0:
            print(f"Running burn-in phase with {n_warmup} steps per walker...")
            sampler.run_mcmc(pos, n_warmup)
            pos = sampler.get_last_sample()

            # Reset sampler to clear burn-in samples
            sampler.reset()

        # Run production
        print(f"Running production phase with {n_samples} steps per walker...")
        sampler.run_mcmc(pos, n_samples)

        # Get samples (flatten removes walker dimension)
        samples = sampler.get_chain(flat=True)

        # Calculate acceptance rate
        accept_rate = np.mean(sampler.acceptance_fraction)
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
            metadata={"acceptance_rate": accept_rate, "nwalkers": self.nwalkers},
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(nwalkers={self.nwalkers}, name='{self.name}')"
