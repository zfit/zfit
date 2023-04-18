#  Copyright (c) 2023 zfit

from __future__ import annotations

from collections.abc import Iterable

from zfit import z
from zfit.core.interfaces import ZfitPDF
from zfit.util.exception import NotExtendedPDFError
from zfit.z.random import sample_with_replacement, counts_multinomial
from .util.container import convert_to_container

__all__ = ["poisson", "sample_with_replacement", "counts_multinomial"]


def poisson(n=None, pdfs: Iterable[ZfitPDF] = None):
    if n and pdfs:
        raise ValueError("Cannot specify both, `n`, and `pdfs`, at the same time.")

    if pdfs:
        pdfs = convert_to_container(pdfs)
        not_extended = [pdf.is_extended for pdf in pdfs]
        if not_extended:
            raise NotExtendedPDFError(
                f"The following pdfs are not extended but need to be: {not_extended}"
            )
        if len(pdfs) > 1:
            raise ValueError("More than one model (currently) not supported.")

        # single pdf only implementation here
        yield_ = pdfs[0].get_yield()
    else:
        yield_ = n

    poisson_term = z.random.poisson(lam=yield_)
    return poisson_term
