"""Module for testing of the zfit components.

Contains a singleton instance to register new PDFs and let them be tested.
"""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import scipy.stats

from ..util.container import convert_to_container
from zfit._interfaces import ZfitPDF

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = ["tester"]

import scipy.integrate


def check_integrate(func, limits, norm):
    if norm is not False:
        return check_integrate(func, limits, False) / check_integrate(
            func, norm, False
        )
    lower, upper = limits.limid1d
    return scipy.integrate.quad(func, lower, upper)


class AutoTester:
    def __init__(self):
        self.pdfs = []

    def register_pdf(
            self,
            pdf_class: ZfitPDF,
            params_factories: Callable | Iterable[Callable],
            scipy_dist: scipy.stats.rv_continuous = None,
            analytic_int_axes: None | int | list[tuple[int, ...]] = None,
    ):
        # if not isinstance(pdf_class, ZfitPDF):
        #     raise TypeError(f"PDF {pdf_class} is not a ZfitPDF.")
        params_factories = convert_to_container(params_factories)

        if isinstance(analytic_int_axes, tuple):
            raise TypeError(
                "`analytic_int_axes` is either a number or a list of tuples."
            )
        analytic_int_axes = convert_to_container(
            analytic_int_axes, non_containers=[tuple]
        )
        if analytic_int_axes is not None:
            isinstance(analytic_int_axes)
        registration = {
            "pdf_class": pdf_class,
            "params": params_factories,
            "scipy_dist": scipy_dist,

        }
        self.pdfs.append(registration)

    def create_parameterized_pdfs(self):
        if len(self.pdfs) == 0:
            return [], []

        argnames = list(self.pdfs[0].keys())
        argvals = [[] * len(argnames)]

        for pdf in self.pdfs:
            for i, param in enumerate(pdf.values()):
                argvals[i].append(param)

        return argnames, argvals


tester = AutoTester()
