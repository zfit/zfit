"""MCMC samplers for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import z
from ..core.interfaces import ZfitParameter
from ..z import numpy as znp
from .base_sampler import BaseMCMCSampler

if TYPE_CHECKING:
    import emcee
    import numpy.typing as npt

    from zfit._bayesian.posterior import PosteriorSamples
    from zfit.core.interfaces import ZfitLoss


class EmceeSampler(BaseMCMCSampler):
    """MCMC sampler using emcee (https://emcee.readthedocs.io).

    EmceeSampler is an ensemble sampler that uses multiple 'walkers' to explore
    the posterior distribution. It's particularly good for problems with
    strongly correlated parameters and doesn't require gradients of the
    log-probability function.

    This sampler requires all parameters to have priors defined, as it uses
    the product of likelihood and prior for sampling.

    Examples:
        Basic usage with default settings:

        >>> sampler = zfit.mcmc.EmceeSampler(nwalkers=32)
        >>> result = sampler.sample(loss=nll, params=params, n_samples=1000)

        With custom moves and settings:

        >>> import emcee
        >>> custom_moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        >>> sampler = zfit.mcmc.EmceeSampler(nwalkers=50,moves=custom_moves,verbosity=8)
        >>> result = sampler.sample(loss=nll, params=params,
        ...                        n_samples=2000, n_warmup=500)
    """

    def __init__(
        self,
        nwalkers: int | None = None,
        *,
        n_samples: int | None = None,
        n_warmup: int | None = None,
        moves: list[tuple[emcee.moves.Move, float]] | None = None,
        backend: emcee.backends.Backend | None = None,
        # pool: object | None = None,  # not possible?
        name: str = "EmceeSampler",
        verbosity: int | None = None,
    ):
        """Initialize an EmceeSampler.

        Args:
            nwalkers: Number of walkers to use. If None, will use
                max(2 * n_dims, 5) where n_dims is the number of parameters.
                Must be at least twice the number of dimensions.
            n_warmup: Default value for number of samples for warmup. The number of warmup points that will be
                discarded.
            n_samples: Default value for number of samples. The number of points to sample.
            moves: The proposal moves to use. Can be a single move
                or a list of (move, weight) tuples. If None, uses emcee's default
                StretchMove. See emcee documentation for available moves.
            backend: Backend to store the chain
                state and samples. Useful for checkpointing long runs. If None,
                samples are stored in memory only.
            name: Name of the sampler for identification.
            verbosity: Verbosity level:
                - 0-6: No progress bars
                - 7: Print sampling phases
                - 8+: Show progress bars during sampling

        Raises:
            ImportError: If emcee is not installed.

        Note:
            The number of walkers should be at least 2 * n_params for good performance.
            Larger numbers of walkers can help with difficult posteriors but increase
            computational cost linearly.
        """
        try:
            import emcee
        except ImportError:
            msg = "emcee is required for EmceeSampler. Install with 'pip install emcee'."
            raise ImportError(msg)
        super().__init__(name=name, verbosity=verbosity, n_samples=n_samples, n_warmup=n_warmup)
        self.nwalkers = nwalkers
        self.moves = moves
        self.backend = backend

    def _sample(
        self,
        loss: ZfitLoss,
        params: list[ZfitParameter],
        n_samples: int,
        n_warmup: int,
        init: PosteriorSamples | None,
    ) -> PosteriorSamples:
        """Implementation of emcee sampling.

        Note:
            - The total number of samples in the result is n_samples * nwalkers
            - Sampling time scales linearly with nwalkers, n_samples, and n_warmup
        """
        import emcee

        # Import here to avoid circular imports
        from zfit._bayesian.posterior import PosteriorSamples

        n_dims = len(params)
        if (nwalkers := self.nwalkers) is None:
            nwalkers = max(2 * n_dims, 5)

        # @zfit.z.function
        def calculate_priors(x):
            # Calculate log prior
            return znp.sum([param.prior.log_pdf(x[i]) for i, param in enumerate(params)])

        # Define the log probability function
        def log_prob(x):
            x = znp.asarray(x)
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

        # Initialize walkers

        # Track if we're using emcee state continuation
        using_emcee_state = False
        emcee_state = None
        pos = None  # Initialize pos variable

        if init is not None:
            # Initialize from previous PosteriorSamples
            if not isinstance(init, PosteriorSamples):
                msg = f"init must be a PosteriorSamples instance, not {type(init)}"
                raise TypeError(msg)

            # Check parameter compatibility
            init_param_names = set(init.param_names)
            current_param_names = {p.name for p in params}
            if init_param_names != current_param_names:
                msg = (
                    f"Parameter names don't match. "
                    f"Previous: {sorted(init_param_names)}, "
                    f"Current: {sorted(current_param_names)}"
                )
                raise ValueError(msg)

            # Check if the previous run was also from an emcee sampler
            if hasattr(init, "info") and init.info.get("type") == "emcee":
                # Try to use the stored emcee state for improved continuation
                stored_state = init.info.get("state")
                if stored_state is not None:
                    self._print("Found previous emcee state - attempting optimized continuation", level=7)

                    # Check if state is compatible (has coords attribute and right shape)
                    try:
                        if hasattr(stored_state, "coords") and stored_state.coords is not None:
                            state_nwalkers = stored_state.coords.shape[0]
                            state_ndims = stored_state.coords.shape[1]

                            if state_ndims == n_dims:
                                if state_nwalkers == nwalkers:
                                    # Perfect match - we can use the state directly
                                    pos = stored_state.coords

                                    # Check if parameter order has changed
                                    reorder_indices = []
                                    for param in params:
                                        init_idx = init._position_by_name[param.name]
                                        reorder_indices.append(init_idx)

                                    if reorder_indices != list(range(n_dims)):
                                        # Need to reorder columns from init's order to current order
                                        pos = pos[:, reorder_indices]
                                        # We can't use the full emcee state when reordering
                                        using_emcee_state = False
                                        self._print(
                                            "Parameter order changed, reordering positions but not using full state",
                                            level=7,
                                        )
                                    else:
                                        # Parameter order unchanged, can use full state
                                        emcee_state = stored_state
                                        using_emcee_state = True
                                        self._print(
                                            f"Using exact emcee state continuation with {nwalkers} walkers", level=7
                                        )
                                else:
                                    # Different number of walkers - need to adapt
                                    self._print(
                                        f"Adapting emcee state: {state_nwalkers} -> {nwalkers} walkers", level=7
                                    )
                                    pos = self._adapt_walker_positions(stored_state.coords, nwalkers, n_dims)

                                    # Check if parameter order has changed
                                    reorder_indices = []
                                    for param in params:
                                        init_idx = init._position_by_name[param.name]
                                        reorder_indices.append(init_idx)

                                    if reorder_indices != list(range(n_dims)):
                                        # Need to reorder columns from init's order to current order
                                        pos = pos[:, reorder_indices]
                                        self._print(
                                            "Parameter order changed, reordered positions after adaptation", level=7
                                        )
                                    # Note: We lose the log_prob and random_state when adapting walkers
                            else:
                                self._print(f"State dimension mismatch: {state_ndims} != {n_dims}", level=5)
                                # Fall through to position extraction
                    except Exception as e:
                        self._print(f"Could not use stored emcee state: {e}", level=5)
                        # Fall through to position extraction

            # If we couldn't use the emcee state, extract positions
            if not using_emcee_state and pos is None:
                # Get the last positions from previous sampling
                # Extract positions for the last 'nwalkers' samples
                if hasattr(init, "raw_result") and init.raw_result is not None:
                    # If we have access to the raw emcee sampler, use its last positions
                    try:
                        # Get the last state from the sampler
                        chain = init.raw_result.get_chain()
                        # chain has shape (n_steps, n_walkers, n_params)
                        last_positions = chain[-1, :, :]  # Last step, all walkers

                        # If number of walkers differs, we need to resample
                        prev_nwalkers = last_positions.shape[0]
                        if prev_nwalkers != nwalkers:
                            pos = self._adapt_walker_positions(last_positions, nwalkers, n_dims)
                        else:
                            pos = last_positions

                        # Ensure parameter ordering matches
                        # Create mapping from init's parameter order to current parameter order
                        reorder_indices = []
                        for param in params:
                            init_idx = init._position_by_name[param.name]
                            reorder_indices.append(init_idx)

                        if reorder_indices != list(range(n_dims)):
                            # Need to reorder columns from init's order to current order
                            pos = pos[:, reorder_indices]

                    except Exception as e:
                        self._print(f"Could not extract positions from raw sampler: {e}", level=5)
                        # Fall back to using the flat samples
                        pos = self._extract_positions_from_samples(init, params, nwalkers, n_dims)
                else:
                    # Use flat samples to reconstruct positions
                    pos = self._extract_positions_from_samples(init, params, nwalkers, n_dims)

            if using_emcee_state:
                self._print(f"Continuing from previous emcee state with {nwalkers} walkers", level=7)
            else:
                self._print(f"Initialized {nwalkers} walkers from previous posterior samples", level=7)
        else:
            # Initialize around current parameter values, respecting bounds
            initial_positions = np.array([param.value() for param in params])
            pos = np.zeros((nwalkers, n_dims))

            for i, param in enumerate(params):
                # Use small perturbations around current value, clipped to bounds
                center = initial_positions[i]

                scale = param.stepsize

                # Generate perturbations
                perturbations = center + scale * np.random.randn(nwalkers)

                # Clip to parameter bounds if they exist
                if param.has_limits:
                    if param.lower is not None:
                        perturbations = np.maximum(perturbations, param.lower + 1e-8)
                    if param.upper is not None:
                        perturbations = np.minimum(perturbations, param.upper - 1e-8)

                pos[:, i] = perturbations

        # Set up sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            n_dims,
            log_prob,
            moves=self.moves,
            backend=self.backend,
            vectorize=False,
        )
        oldvals = np.array(params)
        import zfit

        with zfit.param.set_values(params, oldvals):
            # Run burn-in
            if n_warmup > 0:
                self._print(f"Running burn-in phase with {n_warmup} steps...", level=7)
                if using_emcee_state:
                    # Use the full state object which includes log_prob and random state
                    self._print("Starting from previous emcee state", level=7)
                    state = sampler.run_mcmc(emcee_state, n_warmup, progress=self.verbosity >= 8)
                else:
                    state = sampler.run_mcmc(pos, n_warmup, progress=self.verbosity >= 8)
                sampler.reset()
            elif using_emcee_state:
                state = emcee_state
                self._print("Skipping burn-in, continuing from emcee state", level=7)
            else:
                state = pos
                self._print("Skipping burn-in", level=7)

            # Run production
            self._print(f"Running production phase with {n_samples} steps...", level=7)
            state = sampler.run_mcmc(state, n_samples, progress=self.verbosity >= 8)

            # Create result object
            samples = sampler.get_chain(flat=True)

            return PosteriorSamples(
                info={"type": "emcee", "state": state, "sampler": sampler},
                samples=samples,
                params=params,
                loss=loss,
                sampler=self,
                n_warmup=n_warmup,
                n_samples=n_samples,
                raw_result=sampler,
            )

    def _adapt_walker_positions(
        self,
        positions: npt.NDArray[np.float64],
        nwalkers: int,
        n_dims: int,
    ) -> npt.NDArray[np.float64]:
        """Adapt walker positions when number of walkers changes.

        Args:
            positions: Array of shape (prev_nwalkers, n_dims) with walker positions.
            nwalkers: Target number of walkers.
            n_dims: Number of dimensions.

        Returns:
            Array of shape (nwalkers, n_dims) with adapted positions.
        """
        prev_nwalkers = positions.shape[0]

        if nwalkers > prev_nwalkers:
            # Need more walkers: randomly duplicate some
            indices = np.random.choice(prev_nwalkers, nwalkers, replace=True)
            new_positions = positions[indices]

            # Add small noise to duplicated walkers to break symmetry
            noise_mask = np.zeros(nwalkers, dtype=bool)
            unique_indices, counts = np.unique(indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                if count > 1:
                    duplicates = np.where(indices == idx)[0][1:]  # Skip first occurrence
                    noise_mask[duplicates] = True

            # Add scaled noise based on the spread of positions
            if np.sum(noise_mask) > 0:
                pos_std = np.std(positions, axis=0)
                noise_scale = 1e-4 * np.maximum(pos_std, 1e-8)  # Avoid zero scale
                new_positions[noise_mask] += noise_scale * np.random.randn(np.sum(noise_mask), n_dims)
        else:
            # Need fewer walkers: randomly select subset
            indices = np.random.choice(prev_nwalkers, nwalkers, replace=False)
            new_positions = positions[indices]

        return new_positions

    def _extract_positions_from_samples(
        self,
        init: PosteriorSamples,
        params: list[ZfitParameter],
        nwalkers: int,
        n_dims: int,
    ) -> npt.NDArray[np.float64]:
        """Extract walker positions from flat posterior samples.

        Args:
            init: PosteriorSamples instance.
            params: Current parameter list.
            nwalkers: Number of walkers needed.
            n_dims: Number of dimensions.

        Returns:
            Array of shape (nwalkers, n_dims) with initial positions.
        """
        # Get the last nwalkers samples (or resample if needed)
        total_samples = init.samples.shape[0]

        if total_samples >= nwalkers:
            # Use the last nwalkers samples
            last_samples = init.samples[-nwalkers:, :]
        else:
            # Not enough samples, need to resample with replacement
            self._print(f"Resampling from {total_samples} samples to get {nwalkers} walkers", level=7)
            indices = np.random.choice(total_samples, nwalkers, replace=True)
            last_samples = init.samples[indices, :]
            # Add small noise to duplicated samples
            unique_indices, counts = np.unique(indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                if count > 1:
                    duplicates = np.where(indices == idx)[0][1:]  # Skip first occurrence
                    last_samples[duplicates] += 1e-4 * np.random.randn(len(duplicates), n_dims)

        # Ensure parameter ordering matches
        # Create mapping from init's parameter order to current parameter order
        reorder_indices = []
        for param in params:
            init_idx = init._position_by_name[param.name]
            reorder_indices.append(init_idx)

        if reorder_indices != list(range(n_dims)):
            # Need to reorder columns from init's order to current order
            last_samples = last_samples[:, reorder_indices]

        return np.asarray(last_samples)
