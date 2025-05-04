"""MCMC _mcmc for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import numpy as np

from .. import z
from ..core.interfaces import ZfitParameter
from ..util.container import convert_to_container
from ..z import numpy as znp
from .base_sampler import BaseMCMCSampler


class EmceeSampler(BaseMCMCSampler):
    """MCMC sampler using emcee (https://emcee.readthedocs.io)."""

    def __init__(
        self,
        nwalkers=None,
        moves=None,
        backend=None,
        name="EmceeSampler",
        *,
        verbosity=None,
    ):
        """Initialize an EmceeSampler.

        Args:
            *:
            nwalkers: Number of walkers to use. If None, will use 2 * n_dims, at least 5
            moves: The proposal moves to use (emcee-specific)
            backend: The backend to store samples
            pool: Pool object for parallel sampling
            name: Name of the sampler
        """
        try:
            import emcee
        except ImportError:
            msg = "emcee is required for EmceeSampler. Install with 'pip install emcee'."
            raise ImportError(msg)
        super().__init__(name=name, verbosity=verbosity)
        self.nwalkers = nwalkers
        self.moves = moves
        self.backend = backend
        self.pool = None

    def sample(self, loss, params=None, n_samples=1000, n_warmup=100):
        """Sample from the posterior distribution.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of posterior samples to generate
            n_warmup: Number of warmup/burn-in steps
        """
        import emcee

        # Import here to avoid circular imports
        from zfit._bayesian.results import BayesianResult
        from zfit.core.interfaces import ZfitLoss

        if not isinstance(loss, ZfitLoss):
            msg = f"loss must be a ZfitLoss instance, not {type(loss)}"
            raise TypeError(msg)

        if params is None:
            params = loss.get_params(floating=True)
        else:
            params = convert_to_container(params)
            if not all(isinstance(param, ZfitParameter) for param in params):
                msg = "Not all parameters are ZfitParameter"
                raise TypeError(msg)

        n_dims = len(params)
        if (nwalkers := self.nwalkers) is None:
            nwalkers = max(2 * n_dims, 5)  # At least 100 walkers
        params = list(params)
        if noprior := [p for p in params if p.prior is None]:
            msg = f"Parameters {noprior} do not have priors defined"
            raise ValueError(msg)

        # @zfit.z.function
        def calculate_priors(x):
            # Calculate log prior
            return znp.sum([param.prior.log_pdf(x[i]) for i, param in enumerate(params)])

        # Define the log probability function
        def log_prob(x):
            x = znp.asarray(x)
            for i, param in enumerate(params):
                if param.lower is not None and x[i] < param.lower:
                    return -np.inf
                if param.upper is not None and x[i] > param.upper:
                    return -np.inf
            return log_prob_jit(x)

        @z.function(wraps="tensor")
        def log_prob_jit(x):
            # with zfit.param.set_values(params, x):
            # Calculate log likelihood (negative of loss)

            import zfit

            zfit.param.assign_values_jit(params, x)
            log_likelihood = -loss.value()
            log_prior = calculate_priors(x)
            return log_likelihood + log_prior

        # Initialize walkers around the current parameter values
        if not all(isinstance(p, ZfitParameter) for p in params):
            msg = "params must be a sequence of ZfitParameter objects"
            raise TypeError(msg)

        initial_positions = np.array([param.value() for param in params])
        pos = initial_positions + 1e-4 * np.random.randn(nwalkers, n_dims)

        # For scale parameters, make sure we stay positive
        for i, param in enumerate(params):
            # Check for parameter limits
            if (lower := param.lower) is not None:
                pos[:, i] = np.maximum(pos[:, i], lower + 1e-5)
            if (upper := param.upper) is not None:
                pos[:, i] = np.minimum(pos[:, i], upper - 1e-5)

        # Set up sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            n_dims,
            log_prob,
            moves=self.moves,
            backend=self.backend,
            pool=self.pool,
            vectorize=False,
        )

        # Run burn-in
        if n_warmup > 0:
            self._print(f"Running burn-in phase with {n_warmup} steps...", level=7)
            state = sampler.run_mcmc(pos, n_warmup, progress=self.verbosity >= 8)
            sampler.reset()
        else:
            state = pos

        # Run production
        self._print(f"Running production phase with {n_samples} steps...")
        state = sampler.run_mcmc(state, n_samples, progress=self.verbosity >= 8)

        # Create result object
        samples = sampler.get_chain(flat=True)

        return BayesianResult(
            samples=samples,
            params=params,
            loss=loss,
            sampler=self,
            n_warmup=n_warmup,
            n_samples=n_samples,
            raw_result=sampler,
            state=state,
        )
