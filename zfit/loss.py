#  Copyright (c) 2020 zfit

from .core.loss import ExtendedUnbinnedNLL, UnbinnedNLL, BaseLoss, SimpleLoss

__all__ = ['ExtendedUnbinnedNLL', "UnbinnedNLL", "BaseLoss", "SimpleLoss", 'experimental_enable_loss_penalty']

from .util.warnings import warn_experimental_feature


@warn_experimental_feature
def experimental_enable_loss_penalty(enable=True):
    """EXPERIMENTAL! Enable a loss penalty if the loss is NaN, which can push back the minimizer.

    Won't work with every minimizer

    Args:
        enable: If True, enable this feature.
    """
    global _experimental_loss_penalty_nan
    _experimental_loss_penalty_nan = enable


_experimental_loss_penalty_nan = False
