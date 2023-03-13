#  Copyright (c) 2023 zfit
import json

import asdf
import numpy as np
import pytest


def test_dumpload_hs3_pdf(request):
    import zfit

    mu = zfit.Parameter("mu", 1.2, -4.0, 5.0)
    sigma3 = zfit.Parameter("sigma3", 0.1, 0.0, 1.0)
    sigma2 = zfit.Parameter("sigma2", 42, 0.0, 100.0)
    sigma1_free = zfit.Parameter("sigma1_free", 422, 0.0, 1002.0)
    sigma1 = zfit.ComposedParameter("sigma1", lambda x: x + 1, params=sigma1_free)
    mu3 = zfit.Parameter("mu3", 2, -2, 3, step_size=0.1)
    frac1 = zfit.param.ConstantParameter("frac1", 0.2)
    frac2 = zfit.Parameter("frac2", 0.3, 0.0, 1.0)

    obs = zfit.Space("obs1", limits=(-4, 5))
    gauss1 = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs)
    gauss2 = zfit.pdf.Gauss(mu=mu, sigma=sigma2, obs=obs)
    gauss3 = zfit.pdf.Gauss(mu=mu3, sigma=sigma3, obs=obs)
    model = zfit.pdf.SumPDF([gauss1, gauss2, gauss3], fracs=[frac1, frac2])

    hs3model = zfit.hs3.dump(model)
    hs3model["metadata"]["serializer"]["version"] = "ZFIT_ARBITRARY_VALUE"
    hs3model_true = pytest.helpers.get_truth("hs3", "sum3gauss.json", request, hs3model)
    hs3model_cleaned, hs3model_true = pytest.helpers.cleanup_hs3(
        hs3model, hs3model_true
    )
    assert hs3model_cleaned == hs3model_true

    model_loaded = zfit.hs3.load(hs3model)
    assert model_loaded is not model
    hs3model = zfit.hs3.dump(model_loaded["pdfs"].values())
    hs3model_cleaned, hs3model_true = pytest.helpers.cleanup_hs3(
        hs3model, hs3model_true
    )
    assert hs3model_cleaned == hs3model_true
    # second time, no loop to better check if failure in unittest
    model_loaded = zfit.hs3.load(hs3model)
    assert model_loaded is not model
    hs3model = zfit.hs3.dump(model_loaded["pdfs"].values())

    hs3model_cleaned, hs3model_true = pytest.helpers.cleanup_hs3(
        hs3model, hs3model_true
    )
    assert hs3model_cleaned == hs3model_true

    hs3model_false_truth = hs3model_true.copy()
    hs3model_false_truth["variables"]["mu3"] = mu


def test_dumpload_hs3_loss(request):
    import zfit

    mu = zfit.Parameter("mu", 1.2, -4.0, 5.0)
    sigma3 = zfit.Parameter("sigma3", 0.1, 0.0, 1.0)
    sigma2 = zfit.Parameter("sigma2", 42, 0.0, 100.0)
    sigma1_free = zfit.Parameter("sigma1_free", 422, 0.0, 1002.0)
    sigma1 = zfit.ComposedParameter("sigma1", lambda x: x + 1, params=sigma1_free)
    mu3 = zfit.Parameter("mu3", 2, -2, 3, step_size=0.1)
    frac1 = zfit.param.ConstantParameter("frac1", 0.2)
    frac2 = zfit.Parameter("frac2", 0.3, 0.0, 1.0)

    obs = zfit.Space("obs1", limits=(-4, 5))
    gauss1 = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs)
    gauss2 = zfit.pdf.Gauss(mu=mu, sigma=sigma2, obs=obs)
    gauss3 = zfit.pdf.Gauss(mu=mu3, sigma=sigma3, obs=obs)
    model = zfit.pdf.SumPDF([gauss1, gauss2, gauss3], fracs=[frac1, frac2])

    data1 = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 1, 17))
    data2 = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.31, 1, 9))
    constraint = zfit.constraint.GaussianConstraint(
        params=mu, observation=1.2, uncertainty=0.14
    )
    loss = zfit.loss.UnbinnedNLL(
        model=[model, gauss2], data=[data1, data2], constraints=constraint
    )

    hs3model = zfit.hs3.dump(loss)
    print(hs3model)
    hs3model["metadata"]["serializer"]["version"] = "ZFIT_ARBITRARY_VALUE"
    hs3model_asdf = asdf.AsdfFile(hs3model)
    hs3model_true = pytest.helpers.get_truth(
        "hs3", "UnbinnedNLL_simple.asdf", request, hs3model_asdf
    ).tree
    hs3model_cleaned, hs3model_true = pytest.helpers.cleanup_hs3(
        hs3model, hs3model_true
    )
    hs3model_cleaned.pop("asdf_library", None)
    hs3model_true.pop("asdf_library", None)
    hs3model_cleaned.pop("history", None)
    hs3model_true.pop("history", None)

    for key, val in hs3model_cleaned.items():
        if isinstance(val, np.ndarray):
            np.testing.assert_equal(val, hs3model_true[key])
        else:
            for k, v in val.items():
                if isinstance(v, np.ndarray):
                    np.testing.assert_equal(v, val[k])
                else:
                    try:
                        is_equal = v == val[k]
                    except ValueError:
                        pass
                    else:
                        if not is_equal:
                            assert v == val[k]  # make sure the diff is shown by pytest
    model_loaded = zfit.hs3.load(hs3model)
    loss = list(model_loaded["loss"].values())[0]
    loss.value()
    # np.testing.assert_equal(hs3model_cleaned, hs3model_true)

    # assert hs3model_cleaned == hs3model_true

    # model_loaded = zfit.hs3.load(hs3model)
    # assert model_loaded is not model
    # hs3model = zfit.hs3.dump(model_loaded["pdfs"].values())
    # hs3model_cleaned, hs3model_true = pytest.helpers.cleanup_hs3(
    #     hs3model, hs3model_true
    # )
    # assert hs3model_cleaned == hs3model_true
    # # second time, no loop to better check if failure in unittest
    # model_loaded = zfit.hs3.load(hs3model)
    # assert model_loaded is not model
    # hs3model = zfit.hs3.dump(model_loaded["pdfs"].values())
    #
    # hs3model_cleaned, hs3model_true = pytest.helpers.cleanup_hs3(
    #     hs3model, hs3model_true
    # )
    # assert hs3model_cleaned == hs3model_true
