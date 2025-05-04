#  Copyright (c) 2025 zfit
from __future__ import annotations

from zfit.core.interfaces import ZfitSampler


class BaseMCMCSampler(ZfitSampler):
    def __init__(
        self,
        *args,
        verbosity=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        if verbosity is None:
            verbosity = 0
        self.verbosity = verbosity

    def _print(self, *args, level=7, **kwargs):
        if self.verbosity >= level:
            print(*args, **kwargs)
