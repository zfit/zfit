#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Iterable

from zfit import z
from zfit.core.interfaces import ZfitPDF
from zfit.util.exception import NotExtendedPDFError
from zfit.z.random import counts_multinomial, sample_with_replacement

from .util.container import convert_to_container

__all__ = ["poisson", "sample_with_replacement", "counts_multinomial"]


def poisson(n=None, pdfs: Iterable[ZfitPDF] | None = None):
    if n and pdfs:
        msg = "Cannot specify both, `n`, and `pdfs`, at the same time."
        raise ValueError(msg)

    if pdfs:
        pdfs = convert_to_container(pdfs)
        not_extended = [pdf.is_extended for pdf in pdfs]
        if not_extended:
            msg = f"The following pdfs are not extended but need to be: {not_extended}"
            raise NotExtendedPDFError(msg)
        if len(pdfs) > 1:
            msg = "More than one model (currently) not supported."
            raise ValueError(msg)

        # single pdf only implementation here
        yield_ = pdfs[0].get_yield()
    else:
        yield_ = n

    return z.random.poisson(lam=yield_)
