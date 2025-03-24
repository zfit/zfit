"""NUTS (No-U-Turn Sampler) for Bayesian inference in zfit."""

#  Copyright (c) 2025 zfit
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import zfit.param
import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitObject

from ..results import Posteriors


class NUTSSampler(ZfitObject):
    """MCMC sampler using NUTS with custom gradient function support."""

    def __init__(self, step_size=0.1, adapt_step_size=True, target_accept=0.8, max_tree_depth=10, name="NUTSSampler"):
        """Initialize a NUTS sampler.

        Args:
            step_size: Initial step size
            adapt_step_size: Whether to adapt the step size during warmup
            target_accept: Target acceptance rate
            max_tree_depth: Maximum tree depth for NUTS algorithm
            name: Name of the sampler
        """
        self.step_size = step_size
        self.adapt_step_size = adapt_step_size
        self.target_accept = target_accept
        self.max_tree_depth = max_tree_depth
        self.name = name

    def sample(
        self, loss, params=None, n_samples=1000, n_warmup=1000, grad_func=None, use_hessian=False, seed=None, **kwargs
    ):
        """Sample from the posterior distribution using NUTS.

        Args:
            loss: The ZfitLoss to sample from
            params: The parameters to sample. If None, use all free parameters
            n_samples: Number of posterior samples to generate
            n_warmup: Number of warmup/burn-in steps
            grad_func: Custom gradient function. If None, use zfit loss.gradient
            use_hessian: If True, use the inverse Hessian for mass matrix adaptation
            seed: Random seed for reproducibility
            **kwargs: Additional sampler-specific arguments

        Returns:
            A Posterior object
        """
        try:
            import pymc
        except ImportError:
            try:
                # Fall back to older version if available
                import pymc3 as pymc
            except ImportError:
                msg = "PyMC is required. Install with 'pip install pymc'."
                raise ImportError(msg)

        # Import here to avoid circular imports
        if params is None:
            params = loss.get_params(floating=True)

        n_dims = len(params)
        param_names = [param.name for param in params]

        # Setup gradient function if not provided
        if grad_func is None:

            def grad_func(x):
                x = znp.asarray(x)
                return grad_func_jit(x)

            @tf.function(autograph=False)
            def grad_func_jit(x):
                zfit.param.assign_values(params, x)
                return loss.gradient(params=params)

        # Define the log probability function
        def log_prob(x):
            x = znp.asarray(x)
            return log_prob_jit(x)

        @tf.function(autograph=False)
        def log_prob_jit(x):
            zfit.param.assign_values(params, x)
            log_likelihood = -loss.value()
            log_prior = znp.sum([getattr(param, "log_prior", lambda: 0.0)() for param in params])
            return log_likelihood + log_prior

        # Custom implementation of NUTS using NumPy and provided gradient function

        # Handle randomness
        if seed is not None:
            np.random.seed(seed)

        # Starting point
        initial_position = np.array([param.value() for param in params])

        # Ensure initial position is within bounds
        for i, param in enumerate(params):
            if hasattr(param, "has_limits") and param.has_limits:
                if hasattr(param, "lower") and param.lower is not None:
                    initial_position[i] = max(initial_position[i], param.lower + 1e-6)
                if hasattr(param, "upper") and param.upper is not None:
                    initial_position[i] = min(initial_position[i], param.upper - 1e-6)

        # Adapt mass matrix if using Hessian
        if use_hessian:
            # Get Hessian at current point
            def hessian_at_point():
                # Store original values to restore later

                # Calculate hessian using zfit loss
                eigvals, hess = hessian_at_point_jit()
                if np.any(eigvals <= 0):
                    # Add regularization
                    min_eigval = np.min(eigvals)
                    if min_eigval <= 0:
                        hess += (-min_eigval + 1e-6) * np.eye(n_dims)

                # Return inverse for mass matrix
                try:
                    return np.linalg.inv(hess)
                except np.linalg.LinAlgError:
                    print("Warning: Hessian inversion failed, using identity matrix")
                    return np.eye(n_dims)

            @tf.function(autograph=False)
            def hessian_at_point_jit():
                hess = loss.hessian(
                    params=params,
                    numgrad=False,
                )
                # Ensure hessian is positive definite by regularizing if needed
                eigvals = tf.linalg.eigvalsh(hess)
                return eigvals, hess

            # Get initial mass matrix from Hessian
            mass_matrix = hessian_at_point()
        else:
            # Use identity mass matrix
            mass_matrix = np.eye(n_dims)

        # Create NUTS sampler with custom gradient function
        def nuts_kernel(position, step_size, log_prob_fn, grad_fn, mass_matrix, max_depth=10):
            """Simplified NUTS kernel with custom gradient"""
            position = np.array(position, dtype=np.float64)
            current_log_prob = log_prob_fn(position)
            grad_fn(position)

            # Initial momentum
            momentum = np.asarray(np.random.multivariate_normal(mean=np.zeros(n_dims), cov=mass_matrix))

            # Hamiltonian at start
            H0 = current_log_prob - 0.5 * np.dot(momentum, np.linalg.solve(mass_matrix, momentum))

            # Leapfrog integrator
            def leapfrog(position, momentum, step_size):
                position = znp.asarray(position)
                momentum = znp.asarray(momentum)
                step_size = znp.asarray(step_size)
                position, momentum = leapfrog_jit(momentum, position, step_size)
                return np.asarray(position), np.asarray(momentum)

            @tf.function(autograph=False, reduce_retracing=True)
            def leapfrog_jit(momentum, position, step_size):
                momentum = momentum + 0.5 * step_size * grad_fn(position)
                position = position + step_size * znp.linalg.solve(mass_matrix, momentum[:, None])[:, 0]
                momentum = momentum + 0.5 * step_size * grad_fn(position)
                return position, momentum

            # Build tree recursively
            def build_tree(position, momentum, u, direction, depth, step_size):
                if depth == 0:
                    # Base case - take one leapfrog step
                    position_new, momentum_new = leapfrog(position, momentum, direction * step_size)
                    log_prob_new = log_prob_fn(position_new)
                    H_new = log_prob_new - 0.5 * np.dot(momentum_new, (np.linalg.solve(mass_matrix, momentum_new)))

                    # Check for acceptance
                    accept = u <= np.exp(H_new - H0)

                    return position_new, momentum_new, position_new, momentum_new, position_new, accept, 1, accept
                else:
                    # Recursively build subtrees
                    position_m, momentum_m, position_p, momentum_p, position_prop, accept1, n1, valid1 = build_tree(
                        position, momentum, u, direction, depth - 1, step_size
                    )

                    if valid1:
                        if direction == -1:
                            position_m2, momentum_m2, _, _, position_prop2, accept2, n2, valid2 = build_tree(
                                position_m, momentum_m, u, direction, depth - 1, step_size
                            )
                            position_m, momentum_m = position_m2, momentum_m2
                        else:
                            _, _, position_p2, momentum_p2, position_prop2, accept2, n2, valid2 = build_tree(
                                position_p, momentum_p, u, direction, depth - 1, step_size
                            )
                            position_p, momentum_p = position_p2, momentum_p2

                        # Update proposal with 50% chance if valid
                        if valid2 and np.random.random() < n2 / (n1 + n2):
                            position_prop = position_prop2

                        # Check validity of combined tree (no-U-turn condition)
                        valid = valid2 and is_valid_trajectory(position_m, position_p, momentum_m, momentum_p)
                        accept = accept1 + accept2
                        n = n1 + n2
                    else:
                        valid = False
                        accept = accept1
                        n = n1

                    return position_m, momentum_m, position_p, momentum_p, position_prop, accept, n, valid

            # Check if trajectory makes a U-turn
            def is_valid_trajectory(position_m, position_p, momentum_m, momentum_p):
                # No-U-turn condition
                position_diff = position_p - position_m
                return position_diff.dot(momentum_m) >= 0 and position_diff.dot(momentum_p) >= 0

            # Build tree to maximum depth
            u = np.random.uniform(
                0, np.exp(current_log_prob - 0.5 * momentum.dot(np.linalg.solve(mass_matrix, momentum)))
            )

            # Start with depth 0 and increase
            position_next = position.copy()
            for depth in range(max_depth):
                # Randomly select direction
                direction = 1 if np.random.random() < 0.5 else -1

                # Initialize one of the edge positions/momenta to current
                if direction == 1:
                    _, _, position_p, momentum_p, position_prop, accept, n, valid = build_tree(
                        position, momentum, u, direction, depth, step_size
                    )
                    position_m, momentum_m = position, momentum
                else:
                    position_m, momentum_m, _, _, position_prop, accept, n, valid = build_tree(
                        position, momentum, u, direction, depth, step_size
                    )
                    position_p, momentum_p = position, momentum

                # Accept or reject proposal
                if valid and np.random.random() < min(1, accept / n):
                    position_next = position_prop

                # Check for U-turn
                if not is_valid_trajectory(position_m, position_p, momentum_m, momentum_p):
                    break

            # Return new position and acceptance info
            return position_next, log_prob_fn(position_next), grad_fn(position_next)

        # Step size adaptation during warmup
        step_size = self.step_size
        if self.adapt_step_size and n_warmup > 0:
            # Dual averaging step size adaptation
            target_accept_rate = self.target_accept
            gamma = 0.05
            t0 = 10
            kappa = 0.75
            mu = np.log(10 * step_size)  # Target log step size

            log_step_size = np.log(step_size)
            log_step_size_bar = 0
            h_bar = 0

            # Run adaptation
            position = initial_position.copy()
            current_log_prob = log_prob(position)
            grad_func(position)

            print(f"Running warmup with step size adaptation ({n_warmup} steps)...")
            for i in tqdm(range(1, n_warmup + 1), desc="Warmup"):
                # Run NUTS with current step size
                position_new, log_prob_new, grad_new = nuts_kernel(
                    position, np.exp(log_step_size), log_prob, grad_func, mass_matrix, max_depth=self.max_tree_depth
                )

                # Metropolis acceptance probability
                alpha = min(1.0, np.exp(log_prob_new - current_log_prob))

                # Update parameters for dual averaging
                h_bar = (1 - 1 / (i + t0)) * h_bar + 1 / (i + t0) * (target_accept_rate - alpha)
                log_step_size = mu - h_bar * np.sqrt(i) / gamma
                log_step_size_bar = i ** (-kappa) * log_step_size + (1 - i ** (-kappa)) * log_step_size_bar

                # Update position
                position = position_new
                current_log_prob = log_prob_new

                # Progress update
                if i % 100 == 0 or i == n_warmup:
                    print(f"  Warmup step {i}/{n_warmup}: step_size = {np.exp(log_step_size):.5f}")

            # Set adapted step size
            step_size = np.exp(log_step_size_bar)
            print(f"Final adapted step size: {step_size:.5f}")

            # Update mass matrix if using Hessian
            if use_hessian:
                mass_matrix = hessian_at_point()

        # Run sampling
        samples = np.zeros((n_samples, n_dims))
        position = initial_position.copy()
        current_log_prob = log_prob(position)
        grad_func(position)

        acceptance_count = 0

        print(f"Running production sampling ({n_samples} steps)...")
        for i in range(n_samples):
            # Run NUTS with final step size
            position_new, log_prob_new, grad_new = nuts_kernel(
                position, step_size, log_prob, grad_func, mass_matrix, max_depth=self.max_tree_depth
            )

            # Metropolis accept/reject step
            alpha = min(1.0, np.exp(log_prob_new - current_log_prob))
            if np.random.random() < alpha:
                position = position_new
                current_log_prob = log_prob_new
                acceptance_count += 1

            # Store sample
            samples[i] = position

            # Progress update
            if (i + 1) % 100 == 0 or i == n_samples - 1:
                accept_rate = acceptance_count / (i + 1)
                print(f"  Sample {i + 1}/{n_samples}: acceptance rate = {accept_rate:.3f}")

        # Create Posterior object
        return Posteriors(
            samples=samples,
            param_names=param_names,
            params=params,
            loss=loss,
            sampler=self,
            n_warmup=n_warmup,
            n_samples=n_samples,
            raw_result={
                "step_size": step_size,
                "mass_matrix": mass_matrix,
                "accept_rate": acceptance_count / n_samples,
            },
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(step_size={self.step_size}, name='{self.name}')"
