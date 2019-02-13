from typing import Iterable

from zfit import ztf
from zfit.core.interfaces import ZfitModel, ZfitPDF
from zfit.util.exception import NotExtendedPDFError
from .util.container import convert_to_container


def poisson(n=None, pdfs: Iterable[ZfitPDF] = None):
    if n and pdfs:
        raise ValueError("Cannot specify both, `n`, and `pdfs`, at the same time.")

    if pdfs:
        pdfs = convert_to_container(pdfs)
        not_extended = [pdf.is_extended for pdf in pdfs]
        if not_extended:
            raise NotExtendedPDFError("The following pdfs are not extended but need to be: {}".format(not_extended))
        if len(pdfs) > 1:
            raise ValueError("More then one model (currently) not supported.")

        # single pdf only implementation here
        yield_ = pdfs[0].get_yield()
    else:
        yield_ = n

    poisson_term = ztf.random_poisson(lam=yield_)
    return poisson_term
