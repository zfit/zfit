"""Sequential Monte Carlo sampler for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np

from zfit.core.interfaces import ZfitObject

from ..results import Posteriors


class SMCSampler(ZfitObject):
    """Sequential Monte Carlo sampler for Bayesian inference."""

    def __init__(
        self,
        n_particles=1000,
        n_mcmc_steps=2,
        ess_threshold=0.5,
        resampling_method="systematic",
        proposal_dist=None,
        name="SMCSampler",
    ):
        """Initialize a Sequential Monte Carlo sampler.

        Args:
            n_particles: Number of particles (samples)
            n_mcmc_steps: Number of MCMC steps for each particle
            ess_threshold: Effective sample size threshold for resampling
            resampling_method: Resampling method ('multinomial', 'systematic', 'stratified', 'residual')
            proposal_dist: Custom proposal distribution function
            name: Name of the sampler
        """
        self.n_particles = n_particles
        self.n_mcmc_steps = n_mcmc_steps
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method
        self.proposal_dist = proposal_dist
        self.name = name

    def sample(self, loss, params=None, n_samples=None, seed=None, tempering_schedule=None, **kwargs):
        """Sample from the posterior distribution using Sequential Monte Carlo.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of final samples (default: n_particles)
            seed: Random seed for reproducibility
            tempering_schedule: Tempering schedule for annealing (default: [0, 0.25, 0.5, 0.75, 1.0])
            **kwargs: Additional SMC-specific arguments

        Returns:
            A Posterior object
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Import here to avoid circular imports
        if params is None:
            params = loss.get_params(floating=True)

        n_dims = len(params)
        param_names = [param.name for param in params]

        # Set number of samples if not specified
        if n_samples is None:
            n_samples = self.n_particles

        # Set default tempering schedule if not provided
        if tempering_schedule is None:
            tempering_schedule = np.linspace(0, 1, 5)

        # Define the log probability function
        def log_prob(x, beta=1.0):
            """Calculate log posterior with tempering parameter beta."""
            if x.ndim > 1:
                # Handle batched evaluation
                return np.array([log_prob(xi, beta) for xi in x])

            # Store original values to restore later
            original_values = [p.value() for p in params]

            try:
                # Set parameter values
                for i, param in enumerate(params):
                    param.set_value(x[i])

                # Calculate log likelihood (negative of loss) with tempering
                log_likelihood = -float(loss.value())

                # Calculate log prior
                log_prior = sum(float(getattr(param, "log_prior", lambda: 0.0)()) for param in params)

                # Return tempered posterior
                return log_prior + beta * log_likelihood
            except Exception:
                # In case of errors, return very negative log-likelihood
                return -np.inf
            finally:
                # Restore original values
                for param, val in zip(params, original_values):
                    param.set_value(val)

        # Set up parameter ranges for proposal distribution
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

                param_ranges.append((float(lower), float(upper)))
            else:
                # Create a reasonable range around current value
                val = param.value()
                if val != 0:
                    param_ranges.append((val - 10.0 * abs(val), val + 10.0 * abs(val)))
                else:
                    param_ranges.append((-10.0, 10.0))

        # Define proposal distribution if not provided
        if self.proposal_dist is None:

            def proposal_dist(x, scale=0.1):
                """Default Gaussian proposal distribution."""
                return x + scale * np.random.randn(*x.shape)
        else:
            proposal_dist = self.proposal_dist

        # Initialize particles from prior
        particles = np.zeros((self.n_particles, n_dims))
        for i in range(n_dims):
            lower, upper = param_ranges[i]
            particles[:, i] = np.random.uniform(lower, upper, size=self.n_particles)

        # Initialize weights uniformly
        log_weights = np.zeros(self.n_particles)

        # SMC tempering algorithm
        print(f"Running Sequential Monte Carlo with {self.n_particles} particles...")

        for t, beta in enumerate(tempering_schedule[1:], 1):
            beta_prev = tempering_schedule[t - 1]
            print(f"Tempering step {t}/{len(tempering_schedule) - 1}: beta = {beta:.3f}")

            # Update weights based on tempering
            skipped_nan = 0
            for i in range(self.n_particles):
                logprob_dif = log_prob(particles[i], beta) - log_prob(particles[i], beta_prev)
                if np.isnan(logprob_dif):
                    skipped_nan += 1
                    continue
                log_weights[i] += logprob_dif
            if skipped_nan > 0:
                print(f"  Skipped {skipped_nan} particles due to NaN log probability difference")

            # Normalize weights
            max_log_weight = np.max(log_weights)
            log_weights -= max_log_weight
            weights = np.exp(log_weights)
            weights /= np.sum(weights)

            # Calculate effective sample size
            ess = 1.0 / np.sum(weights**2)
            ess_ratio = ess / self.n_particles
            print(f"  Effective sample size: {ess:.1f} ({ess_ratio:.2%} of particles)")

            # Resample if ESS is below threshold
            if ess_ratio < self.ess_threshold:
                print("  Resampling particles...")

                # Perform resampling
                if self.resampling_method == "multinomial":
                    # Multinomial resampling
                    indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights, replace=True)

                elif self.resampling_method == "systematic":
                    # Systematic resampling
                    positions = (np.arange(self.n_particles) + np.random.rand()) / self.n_particles
                    cumulative_sum = np.cumsum(weights)
                    indices = np.zeros(self.n_particles, dtype=int)
                    i, j = 0, 0
                    while i < self.n_particles:
                        if positions[i] < cumulative_sum[j]:
                            indices[i] = j
                            i += 1
                        else:
                            j += 1

                elif self.resampling_method == "stratified":
                    # Stratified resampling
                    positions = (np.arange(self.n_particles) + np.random.rand(self.n_particles)) / self.n_particles
                    cumulative_sum = np.cumsum(weights)
                    indices = np.zeros(self.n_particles, dtype=int)
                    j = 0
                    for i in range(self.n_particles):
                        while j < self.n_particles - 1 and positions[i] > cumulative_sum[j]:
                            j += 1
                        indices[i] = j

                elif self.resampling_method == "residual":
                    # Residual resampling
                    indices = []
                    # Deterministic part
                    n_copies = np.floor(self.n_particles * weights).astype(int)
                    for i, n in enumerate(n_copies):
                        indices.extend([i] * n)
                    # Random part for remaining particles
                    n_remaining = self.n_particles - len(indices)
                    if n_remaining > 0:
                        residual_weights = self.n_particles * weights - n_copies
                        residual_weights /= np.sum(residual_weights)
                        residual_indices = np.random.choice(
                            self.n_particles, size=n_remaining, p=residual_weights, replace=True
                        )
                        indices.extend(residual_indices)
                    indices = np.array(indices)

                else:
                    msg = f"Unknown resampling method: {self.resampling_method}"
                    raise ValueError(msg)

                # Copy particles according to resampling
                particles = particles[indices].copy()

                # Reset weights
                log_weights = np.zeros(self.n_particles)
                weights = np.ones(self.n_particles) / self.n_particles

            # MCMC moves to diversity particles
            print(f"  Performing {self.n_mcmc_steps} MCMC steps per particle...")

            acceptance_count = 0

            # Adapt proposal scale to target ~20-40% acceptance rate
            proposal_scale = 0.1  # initial scale

            for step in range(self.n_mcmc_steps):
                for i in range(self.n_particles):
                    # Propose new particle
                    proposal = proposal_dist(particles[i], scale=proposal_scale)

                    # Check parameter bounds
                    valid_proposal = True
                    for j, (lower, upper) in enumerate(param_ranges):
                        if proposal[j] < lower or proposal[j] > upper:
                            valid_proposal = False
                            break

                    if not valid_proposal:
                        continue

                    # Calculate acceptance probability
                    log_prob_current = log_prob(particles[i], beta)
                    log_prob_proposal = log_prob(proposal, beta)

                    # Metropolis acceptance criterion
                    log_alpha = log_prob_proposal - log_prob_current
                    accept = log_alpha >= 0 or np.log(np.random.rand()) < log_alpha

                    if accept:
                        particles[i] = proposal
                        acceptance_count += 1

                # Adjust proposal scale based on acceptance rate
                if step > 0 and step % 5 == 0:
                    acceptance_rate = acceptance_count / (self.n_particles * 5)
                    if acceptance_rate < 0.2:
                        proposal_scale *= 0.9  # Reduce scale if acceptance rate is too low
                    elif acceptance_rate > 0.4:
                        proposal_scale *= 1.1  # Increase scale if acceptance rate is too high
                    acceptance_count = 0

        # Final set of particles and weights
        if n_samples < self.n_particles:
            # Randomly select n_samples particles according to weights
            indices = np.random.choice(self.n_particles, size=n_samples, p=weights, replace=True)
            final_samples = particles[indices]
        else:
            final_samples = particles

        # Create Posterior object
        return Posteriors(
            samples=final_samples,
            param_names=param_names,
            params=params,
            loss=loss,
            sampler=self,
            n_warmup=0,  # SMC doesn't have traditional warmup
            n_samples=len(final_samples),
            raw_result={"particles": particles, "weights": weights, "tempering_schedule": tempering_schedule},
            metadata={"final_ess": 1.0 / np.sum(weights**2), "param_ranges": param_ranges},
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(n_particles={self.n_particles}, name='{self.name}')"
