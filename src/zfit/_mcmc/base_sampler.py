#  Copyright (c) 2025 zfit
from __future__ import annotations

from typing import TYPE_CHECKING

from zfit.core.interfaces import ZfitParameter, ZfitSampler
from zfit.util.container import convert_to_container

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from zfit._bayesian.posterior import PosteriorSamples
    from zfit.core.interfaces import ZfitLoss


class BaseMCMCSampler(ZfitSampler):
    """Base class for MCMC samplers in zfit.

    This abstract base class provides common functionality for all MCMC samplers,
    including verbosity control and utility methods.
    """

    def __init__(
        self,
        *args: Any,
        n_samples: int | None = None,
        n_warmup: int | None,
        verbosity: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the base MCMC sampler.

        Args:
            n_samples: Default number of samples to draw in the sample method.
            verbosity: Verbosity level for output. Higher values produce more output.
                Default is 0 (minimal output).
            *args: Positional arguments passed to parent class.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(
            *args,
            **kwargs,
        )
        if verbosity is None:
            verbosity = 0
        self.verbosity = verbosity
        if n_warmup is None:
            n_warmup = 200
        self._default_n_warmup = n_warmup
        if n_samples is None:
            n_samples = 1000
        self._default_n_samples = n_samples

    def _print(self, *args: Any, level: int = 7, **kwargs: Any) -> None:
        if self.verbosity >= level:
            print(*args, **kwargs)

    def sample(
        self,
        loss: ZfitLoss,
        params: Iterable[ZfitParameter] | None = None,
        n_samples: int | None = None,
        n_warmup: int | None = None,
        init: PosteriorSamples | None = None,
    ) -> PosteriorSamples:
        """Sample from the posterior distribution using MCMC.

        This method runs the MCMC sampler to generate samples from the
        posterior distribution. The sampling process consists of two phases:
        1. Warmup/burn-in: Allow chains to find the typical set
        2. Production: Generate samples for analysis

        All parameters must have priors defined before sampling.

        Args:
            loss: The loss function to sample from. Must be a ZfitLoss
                instance (e.g., UnbinnedNLL, ExtendedUnbinnedNLL).
            params: List of ZfitParameter objects to sample.
                If None, uses all floating parameters from the loss function.
            n_samples: Number of production samples to generate.
                If None, uses the default value specified in the constructor.
            n_warmup: Number of burn-in steps. These samples
                are discarded and used only to allow the chains to converge.
                If init is provided with PosteriorSamples, n_warmup can be 0
                to skip burn-in (default behavior).
            init: Previous posterior samples to initialize
                the sampler from. This allows warm-starting from a previous
                run, potentially skipping burn-in.

        Returns:
            Object containing posterior samples and analysis methods.

        Raises:
            TypeError: If loss is not a ZfitLoss or params contain non-ZfitParameter objects.
            ValueError: If any parameter lacks a prior distribution.

        Examples:
            Basic sampling:

            >>> # Setup model and data
            >>> nll = zfit.loss.UnbinnedNLL(model=model, data=data)
            >>> sampler = zfit.mcmc.EmceeSampler(nwalkers=32)
            >>> result = sampler.sample(loss=nll, n_samples=1000, n_warmup=500)

            Sampling specific parameters:

            >>> # Only sample mu and sigma, keep other parameters fixed
            >>> params_to_sample = [mu, sigma]
            >>> result = sampler.sample(loss=nll, params=params_to_sample,
            ...                        n_samples=2000, n_warmup=1000)

            Warm-start from previous sampling:

            >>> # First run
            >>> result1 = sampler.sample(loss=nll, n_samples=1000, n_warmup=500)
            >>>
            >>> # Continue sampling from where we left off
            >>> result2 = sampler.sample(loss=nll, init=result1,
            ...                         n_samples=2000, n_warmup=0)

        Note:
            - The exact behavior depends on the specific sampler implementation
            - Use verbosity >= 7 to see progress information
            - Consider using larger n_warmup for difficult posteriors
            - When using init, ensure the parameters match between runs
        """
        # Import here to avoid circular imports
        from zfit.core.interfaces import ZfitLoss

        # Validate inputs
        if not isinstance(loss, ZfitLoss):
            msg = f"loss must be a ZfitLoss instance, not {type(loss)}"
            raise TypeError(msg)

        # Handle default n_samples
        if n_samples is None:
            n_samples = self._default_n_samples

        if n_warmup is None:
            n_warmup = self._default_n_warmup if init is None else 0

        # Get and validate parameters
        if params is None:
            params = loss.get_params(floating=True)
        else:
            params = convert_to_container(params)
            if not all(isinstance(param, ZfitParameter) for param in params):
                msg = "Not all parameters are ZfitParameter"
                raise TypeError(msg)

        params = list(params)

        # Check that all parameters have priors
        if noprior := [p for p in params if p.prior is None]:
            msg = f"Parameters {noprior} do not have priors defined"
            raise ValueError(msg)

        # Delegate to concrete implementation
        return self._sample(loss=loss, params=params, n_samples=n_samples, n_warmup=n_warmup, init=init)

    def _sample(
        self,
        loss: ZfitLoss,
        params: list[ZfitParameter],
        n_samples: int,
        n_warmup: int,
        init: PosteriorSamples | None,
    ) -> PosteriorSamples:
        """Concrete implementation of the sampling method.

        This method should be implemented by subclasses to provide the
        actual sampling logic. At this point, all inputs have been
        validated and normalized.

        Args:
            loss: The loss function to sample from.
            params: List of ZfitParameter objects to sample (validated).
            n_samples: Number of production samples to generate.
            n_warmup: Number of burn-in steps.
            init: Previous posterior samples to initialize from.

        Returns:
            PosteriorSamples object.
        """
        msg = "_sample method not implemented, needs to be implemented in subclass. Don't call this method directly."
        raise RuntimeError(msg)
