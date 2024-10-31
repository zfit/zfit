#  Copyright (c) 2024 zfit
from __future__ import annotations

import functools
import operator
from collections.abc import Iterable

import numpy as np
import tensorflow as tf

import zfit
import zfit.z.numpy as znp

from .. import ComposedParameter, Space
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitData, ZfitPDF


class ZfitNLL:
    pass


class ZfitNLLs:
    pass


def to_nlls(nll):
    if isinstance(nll, NLL):
        nll = [nll]
    if isinstance(nll, Iterable):
        return NLLs(nlls=list(nll))
    elif isinstance(nll, NLLs):
        return nll
    else:
        msg = f"Cannot convert {nll} to NLLs"
        raise TypeError(msg)


class NLL(ZfitNLL):
    def __init__(self, dist: ZfitPDF, data: ZfitData):
        self.dist = dist
        self.data = data

    def __call__(self, params=None, *, options=None):
        logpdf = self.logpfds(params=params)
        return -self.summation(logpdf, options=options)

    def __add__(self, other):
        return to_nlls(self) + to_nlls(other)

    def logpfds(self, params=None):
        return self.dist.log_pdf(self.data, params=params)

    def summation(self, logpdfs, *, options=None, full=True):
        if options is None:
            options = {}
        if not full and (offset := options.get("offset", False)):
            logpdfs -= offset
        return znp.sum(logpdfs)


class NLLs(ZfitNLLs):
    def __init__(self, nlls, *, options=None):
        self.nlls = nlls
        self.dist = functools.reduce(
            operator.iadd, [nll.dist if isinstance(nll.dist, list) else [nll.dist] for nll in nlls], []
        )
        self.data = functools.reduce(
            operator.iadd, [nll.data if isinstance(nll.data, list) else [nll.data] for nll in nlls], []
        )
        self.options = {"offset": 0} if options is None else options

    def __call__(self, params=None):
        return znp.sum([nll(params) for nll in self.nlls])

    def __add__(self, other):
        return type(self)(nlls=self.nlls + other.nlls)

    def logpfds(self, params=None):
        return znp.concatenate([nll.logpfds(params=params) for nll in self.nlls])

    def summation(self, logpdfs, *, options=None, full=True):
        if options is None:
            options = self.options
        if not full:
            logpdfs -= options["offset"]
        return znp.sum(logpdfs)


class BinnedNLL(NLLs):
    def __init__(self, expected: ZfitBinnedPDF, observed: ZfitBinnedData):
        # TODO: check if expected and observed are compatible with obs
        self.expected = expected
        self.observed = observed
        # create unique name using random number
        data_array = znp.flatten(observed.values())
        expectedparams = [
            ComposedParameter(
                name=f"{np.random.randint(1e15)}_expected",
                params=expected.get_params(),
                func=lambda params, i=i: znp.flatten(
                    expected.rel_counts(params={p: p.value() for p in params}) * znp.sum(data_array)
                )[i],
            )
            for i in range(znp.flatten(expected.rel_counts()).shape[0])
        ]

        nlls = []
        for dataarr, eparam in zip(data_array, expectedparams):
            nll = ExtendedTerm(param=eparam, observed=dataarr)
            nlls.append(nll)

        super().__init__(nlls=nlls)

    def __call__(self, params=None, *, options=None, full=True):
        values = self.observed.values()
        nvalues = znp.sum(values)
        expected = self.expected.rel_counts(params=params) * nvalues
        logexpected = znp.log(expected)
        pterms = tf.nn.log_poisson_loss(targets=values, log_input=logexpected, compute_full_loss=full)
        return self.summation(pterms, full=full, options=options)


class ExtendedTerm(NLL):
    def __init__(self, param, observed):
        space = Space(f"LOSS_{np.random.randint(1e15)}", 0, 1e100)
        data = zfit.Data(data=observed, obs=space)
        dist = zfit.pdf.Poisson(param, obs=space)
        super().__init__(dist=dist, data=data)


#
# class BinNLL(NLL):
#     def __init__(self, expected, observed):
#         space = Space(f"LOSS_{np.random.randint(1e15)}", 0, 1e100)
#         data = zfit.Data(data=dataarr, obs=space)
#         dist = zfit.pdf.Poisson(eparam, obs=space)
#         super().__init__(dist=dist, data=data)
