#  Copyright (c) 2024 zfit
import numpy as np
import pytest

import zfit


def test_get_dependents_is_deterministic():
    parameters = [zfit.Parameter(f"param{i}", i) for i in range(4)]
    obs = zfit.Space("obs1", (-3, 2))

    def create_pdf(params):
        param1, param2, param3, param4 = params
        gauss1 = zfit.pdf.Gauss(param1, param2, obs=obs)
        gauss2 = zfit.pdf.Gauss(param2, param2, obs=obs)
        gauss3 = zfit.pdf.Gauss(param2, param4, obs=obs)
        gauss4 = zfit.pdf.Gauss(param3, param1, obs=obs)
        sum_pdf = zfit.pdf.SumPDF(
            [gauss1, gauss2, gauss3, gauss4], fracs=[0.1, 0.3, 0.4]
        )
        return sum_pdf

    for params in (parameters, reversed(parameters)):
        pdf = create_pdf(params)
        assert (
            pdf.get_params(floating=None) == pdf.get_params(floating=None)
        ), "get_dependents() not deterministic"


def test_get_params():
    obs = zfit.Space("obs", (-4, 5))
    mu_nofloat = zfit.Parameter("mu_nofloat", 1, floating=False)
    mu2 = zfit.Parameter("mu2", 1)
    sigma2 = zfit.Parameter("sigma2", 2)
    sigma_comp = zfit.ComposedParameter("sigma_comp", lambda s: s * 0.7, params=sigma2)

    yield1 = zfit.Parameter("yield1", 10)
    yield2 = zfit.Parameter("yield2", 200)
    yield2_comp = zfit.ComposedParameter(
        "yield2_comp", lambda y: y * 0.9, params=yield2
    )

    gauss = zfit.pdf.Gauss(mu_nofloat, sigma_comp, obs)
    gauss2 = zfit.pdf.Gauss(mu2, sigma2, obs)
    gauss_ext = gauss.create_extended(yield1)
    gauss2_ext = gauss2.create_extended(yield2_comp)

    frac = zfit.Parameter("frac", 0.4)
    sum1 = zfit.pdf.SumPDF([gauss, gauss2], fracs=frac)
    sum1_ext = zfit.pdf.SumPDF([gauss_ext, gauss2_ext])

    assert set(gauss.get_params()) == {sigma2}

    with pytest.raises(ValueError):
        set(gauss.get_params(floating=True, is_yield=False, extract_independent=False))

    with pytest.raises(ValueError):
        set(
            gauss_ext.get_params(floating=True, is_yield=False, extract_independent=False)
        )
    assert set(
        gauss_ext.get_params(floating=None, is_yield=False, extract_independent=False)
    ) == {mu_nofloat, sigma_comp}
    with pytest.raises(ValueError):
        set(
            gauss_ext.get_params(floating=False, is_yield=False, extract_independent=False)
        )

    with pytest.raises(ValueError):
        set(
            gauss_ext.get_params(floating=True, is_yield=None, extract_independent=False)
        )

    assert set(
        gauss_ext.get_params(floating=None, is_yield=None, extract_independent=False)
    ) == {mu_nofloat, sigma_comp, yield1}
    with pytest.raises(ValueError):
        set(
            gauss_ext.get_params(floating=False, is_yield=None, extract_independent=False)
        )

    assert (
        set(
            gauss_ext.get_params(floating=False, is_yield=True, extract_independent=False)
        )
        == set()
    )
    assert set(
        gauss_ext.get_params(floating=None, is_yield=True, extract_independent=False)
    ) == {yield1}
    assert set(
        gauss_ext.get_params(floating=True, is_yield=True, extract_independent=False)
    ) == {yield1}

    # with extract deps
    assert set(
        gauss_ext.get_params(floating=True, is_yield=False, extract_independent=True)
    ) == {sigma2}
    assert set(
        gauss_ext.get_params(floating=None, is_yield=False, extract_independent=True)
    ) == {mu_nofloat, sigma2}
    assert set(
        gauss_ext.get_params(floating=False, is_yield=False, extract_independent=True)
    ) == {mu_nofloat}

    assert set(
        gauss_ext.get_params(floating=True, is_yield=None, extract_independent=True)
    ) == {sigma2, yield1}
    assert set(
        gauss_ext.get_params(floating=None, is_yield=None, extract_independent=True)
    ) == {mu_nofloat, sigma2, yield1}
    assert set(
        gauss_ext.get_params(floating=False, is_yield=None, extract_independent=True)
    ) == {mu_nofloat}

    assert set(
        gauss_ext.get_params(floating=True, is_yield=True, extract_independent=True)
    ) == {yield1}
    assert set(
        gauss_ext.get_params(floating=None, is_yield=True, extract_independent=True)
    ) == {yield1}
    assert (
        set(
            gauss_ext.get_params(floating=False, is_yield=True, extract_independent=True)
        )
        == set()
    )

    # Gauss ext2
    assert set(
        gauss2_ext.get_params(floating=True, is_yield=False, extract_independent=False)
    ) == {mu2, sigma2}
    assert set(
        gauss2_ext.get_params(floating=None, is_yield=False, extract_independent=False)
    ) == {mu2, sigma2}
    assert (
        set(
            gauss2_ext.get_params(floating=False, is_yield=False, extract_independent=False)
        )
        == set()
    )

    with pytest.raises(ValueError):
        set(
            gauss2_ext.get_params(floating=True, is_yield=None, extract_independent=False)
        )
    assert set(
        gauss2_ext.get_params(floating=None, is_yield=None, extract_independent=False)
    ) == {mu2, sigma2, yield2_comp}
    with pytest.raises(ValueError):
        set(
            gauss2_ext.get_params(floating=False, is_yield=None, extract_independent=False)
        )

    with pytest.raises(ValueError):
        set(
            gauss2_ext.get_params(floating=False, is_yield=True, extract_independent=False)
        )
    assert set(
        gauss2_ext.get_params(floating=None, is_yield=True, extract_independent=False)
    ) == {yield2_comp}
    with pytest.raises(ValueError):
        set(
            gauss2_ext.get_params(floating=True, is_yield=True, extract_independent=False)
        )

    # with extract deps
    assert set(
        gauss2_ext.get_params(floating=True, is_yield=False, extract_independent=True)
    ) == {mu2, sigma2}
    assert set(
        gauss2_ext.get_params(floating=None, is_yield=False, extract_independent=True)
    ) == {mu2, sigma2}
    yield2.floating = False
    assert (
        set(
            gauss2_ext.get_params(floating=False, is_yield=False, extract_independent=True)
        )
        == set()
    )
    yield2.floating = True
    assert (
        set(
            gauss2_ext.get_params(floating=False, is_yield=False, extract_independent=True)
        )
        == set()
    )

    assert set(
        gauss2_ext.get_params(floating=True, is_yield=None, extract_independent=True)
    ) == {mu2, sigma2, yield2}
    assert set(
        gauss2_ext.get_params(floating=None, is_yield=None, extract_independent=True)
    ) == {mu2, sigma2, yield2}
    assert (
        set(
            gauss2_ext.get_params(floating=False, is_yield=None, extract_independent=True)
        )
        == set()
    )

    assert set(
        gauss2_ext.get_params(floating=True, is_yield=True, extract_independent=True)
    ) == {yield2}
    assert set(
        gauss2_ext.get_params(floating=None, is_yield=True, extract_independent=True)
    ) == {yield2}
    assert (
        set(
            gauss2_ext.get_params(floating=False, is_yield=True, extract_independent=True)
        )
        == set()
    )

    # sum extended
    with pytest.raises(ValueError):
        assert set(
            sum1_ext.get_params(floating=True, is_yield=False, extract_independent=False)
        )
    frac0 = sum1_ext.fracs[0]
    frac1 = sum1_ext.fracs[1]
    assert set(
        sum1_ext.get_params(floating=None, is_yield=False, extract_independent=False)
    ) == {mu_nofloat, sigma_comp, mu2, sigma2, frac0, frac1}
    with pytest.raises(ValueError):
        assert set(
            sum1_ext.get_params(floating=False, is_yield=False, extract_independent=False)
        )

    with pytest.raises(ValueError):
        set(
            sum1_ext.get_params(floating=True, is_yield=None, extract_independent=False)
        )
    assert set(
        sum1_ext.get_params(floating=None, is_yield=None, extract_independent=False)
    ) == {mu_nofloat, sigma_comp, mu2, sigma2, frac0, frac1, sum1_ext.get_yield()}
    with pytest.raises(ValueError):
        set(
            sum1_ext.get_params(floating=False, is_yield=None, extract_independent=False)
        )

    with pytest.raises(ValueError):
        set(
            sum1_ext.get_params(floating=False, is_yield=True, extract_independent=False)
        )
    assert set(
        sum1_ext.get_params(floating=None, is_yield=True, extract_independent=False)
    ) == {sum1_ext.get_yield()}
    with pytest.raises(ValueError):
        set(
            sum1_ext.get_params(floating=True, is_yield=True, extract_independent=False)
        )

    # with extract deps
    assert set(
        sum1_ext.get_params(floating=True, is_yield=False, extract_independent=True)
    ) == {
        mu2,
        sigma2,
        yield1,
        yield2,  # fracs depend on them
    }
    assert set(
        sum1_ext.get_params(floating=None, is_yield=False, extract_independent=True)
    ) == {mu_nofloat, mu2, sigma2, yield1, yield2}
    assert set(
        sum1_ext.get_params(floating=False, is_yield=False, extract_independent=True)
    ) == {mu_nofloat}

    assert set(
        sum1_ext.get_params(floating=True, is_yield=None, extract_independent=True)
    ) == {mu2, sigma2, yield1, yield2, yield1, yield2}
    assert set(
        sum1_ext.get_params(floating=None, is_yield=None, extract_independent=True)
    ) == {mu_nofloat, mu2, sigma2, yield1, yield2}
    assert set(
        sum1_ext.get_params(floating=False, is_yield=None, extract_independent=True)
    ) == {mu_nofloat}
    yield1.floating = False
    assert set(
        sum1_ext.get_params(floating=False, is_yield=None, extract_independent=True)
    ) == {mu_nofloat, yield1}
    yield1.floating = True

    assert set(
        sum1_ext.get_params(floating=True, is_yield=True, extract_independent=True)
    ) == {yield1, yield2}
    assert set(
        sum1_ext.get_params(floating=None, is_yield=True, extract_independent=True)
    ) == {yield1, yield2}
    assert (
        set(
            sum1_ext.get_params(floating=False, is_yield=True, extract_independent=True)
        )
        == set()
    )


def test_unique_param_names():
    obs = zfit.Space("obs1", (-1, 2))

    param_a1 = zfit.Parameter("a", 1)
    param_a2 = zfit.Parameter("a", 2)
    param_a3 = zfit.Parameter("a", 5)
    param_a4fix = zfit.Parameter("a", 5, floating=False)
    param_b1 = zfit.Parameter("b", 7)
    param_b2 = zfit.Parameter("b", 3)
    param_c = zfit.Parameter("c", 0.42)

    compparam_a1 = zfit.ComposedParameter("a", lambda x: x, params=param_a1)
    compparam_a2 = zfit.ComposedParameter("a", lambda x: x, params=param_a2)

    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.pdf.Gauss(mu=param_a1, sigma=param_a2, obs=obs)

    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.pdf.Gauss(mu=param_a1, sigma=param_a4fix, obs=obs)

    gauss1 = zfit.pdf.Gauss(mu=param_a1, sigma=param_b1, obs=obs)
    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        gauss1.create_extended(param_b2)
    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        gauss1.set_yield(param_b2)

    gauss1 = zfit.pdf.Gauss(
        mu=param_a1, sigma=param_b1, obs=obs
    )  # recreate, it is now maybe broken
    gauss2 = zfit.pdf.Gauss(mu=param_a2, sigma=param_b1, obs=obs)
    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.pdf.SumPDF([gauss1, gauss2], fracs=param_c)

    data = zfit.Data.from_numpy(obs=obs, array=np.random.normal(0, 1, size=100))
    gauss3 = zfit.pdf.Gauss(mu=param_a1, sigma=param_c, obs=obs)
    sumpdf = zfit.pdf.SumPDF([gauss1, gauss3], fracs=0.7)
    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        sumpdf.create_extended(param_a3)

    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.loss.UnbinnedNLL(
            model=[sumpdf, gauss2], data=[data, data]
        )  # gauss1 and gauss2 don't go with each other

    sumpdfext = sumpdf.create_extended(param_b1)
    gaussext = gauss2.create_extended(15)
    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.loss.ExtendedUnbinnedNLL(model=[sumpdfext, gaussext], data=[data, data])

    # composed parameters
    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.pdf.Gauss(mu=compparam_a1, sigma=param_a2, obs=obs)

    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.pdf.Gauss(mu=compparam_a1, sigma=param_a4fix, obs=obs)

    with pytest.raises(zfit.exception.ParamNameNotUniqueError):
        zfit.ComposedParameter("ue", lambda x: x, params=[param_a1, param_a2])
