#  Copyright (c) 2024 zfit

import pytest
import numpy as np

import time


def benchmark_pdf(pdf, arr_val):
    start = None
    for i, val in enumerate(arr_val):
        pdf.pdf(val)
        if i == 5:  # warmup
            start = time.time()
    assert start is not None, "Too few trials"
    return time.time() - start


# --------------------------------------------------
def get_dscb(obs, prefix=None):
    import zfit

    mu = zfit.Parameter(f"mu_{prefix}", 2.4, -1, 5)
    sg = zfit.Parameter(f"sg_{prefix}", 1.3, 0, 5)

    al = zfit.Parameter(f"al_{prefix}", -2.4, -3, -1)
    nl = zfit.Parameter(f"nl_{prefix}", 1.3, 0, 5)

    ar = zfit.Parameter(f"ar_{prefix}", +2.4, +1, +3)
    nr = zfit.Parameter(f"nr_{prefix}", 1.3, 0, 5)

    pdf = zfit.pdf.DoubleCB(
        obs=obs, mu=mu, sigma=sg, alphal=al, nl=nl, alphar=ar, nr=nr
    )
    nev = zfit.Parameter(f"nev_{prefix}", 100, 0, 1000)

    pdf = pdf.create_extended(nev, name=prefix)

    return pdf


# --------------------------------------------------
def get_gaus(obs, prefix=None):
    import zfit

    mu = zfit.Parameter(f"mu_{prefix}", 2.4, -1, 5)
    sg = zfit.Parameter(f"sg_{prefix}", 1.3, 0, 5)

    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sg)
    nev = zfit.Parameter(f"nev_{prefix}", 100, 0, 1000)

    pdf = pdf.create_extended(nev, name=prefix)

    return pdf


def test_sum_cb_speed():
    import zfit

    arr_val = np.linspace(-10, 10, 25)
    obs = zfit.Space("x", -10, 10)
    m1 = get_dscb(obs, prefix="g1")
    m2 = get_dscb(obs, prefix="g2")
    sm = zfit.pdf.SumPDF([m1, m2], name="g1 + g2")
    res1 = benchmark_pdf(m1, arr_val)
    res2 = benchmark_pdf(m2, arr_val)
    ressum = benchmark_pdf(sm, arr_val)

    assert ressum < 3 * (res1 + res2)
