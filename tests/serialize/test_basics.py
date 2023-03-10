#  Copyright (c) 2023 zfit
import copy

import numpy as np
import pytest
from frozendict import frozendict

import zfit


def test_serial_space():
    import zfit

    space = zfit.Space("x", (-1, 1))
    json1 = space.to_json()
    space2 = zfit.Space.get_repr().parse_raw(json1).to_orm()
    space2_direct = zfit.Space.hs3.from_json(json1)

    assert space == space2
    assert space == space2_direct
    json2 = space2.to_json()
    assert json1 == json2
    space3 = zfit.Space.get_repr().parse_raw(json2).to_orm()
    assert space2 == space3
    assert space == space3


def test_serial_param():
    import zfit

    param = zfit.Parameter("mu", 0.1, -1, 1)

    json1 = param.to_json()
    param2 = zfit.Parameter.get_repr().parse_raw(json1).to_orm()

    assert param == param2
    json2 = param2.to_json()
    assert json1 == json2
    param3 = zfit.Parameter.get_repr().parse_raw(json2).to_orm()
    assert param2 == param3
    assert param == param3


def gauss(**kwargs):
    import zfit

    mu = zfit.Parameter("mu_gauss", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_gauss", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)


def prod2dgauss(extended=None, **kwargs):
    import zfit

    mu = zfit.Parameter("mu_gauss", 0.1, -1, 1)
    sigma1 = zfit.Parameter("sigma_gauss", 0.1, 0, 1)
    obs1 = zfit.Space("obs1", (-3, 7))
    sigma2 = zfit.Parameter("sigma_gauss2", 0.1, 0, 1)
    obs2 = zfit.Space("obs2", (-13, 5))
    gauss1 = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs1)
    gauss2 = zfit.pdf.Gauss(mu=mu, sigma=sigma2, obs=obs2)
    prod = zfit.pdf.ProductPDF([gauss1, gauss2], extended=extended)
    return prod


def cauchy(**kwargs):
    import zfit

    m = zfit.Parameter("m_cauchy", 0.1, -1, 1)
    gamma = zfit.Parameter("gamma_cauchy", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.Cauchy(m=m, gamma=gamma, obs=obs)


def exponential(**kwargs):
    import zfit

    lam = zfit.Parameter("lambda_exp", 0.1, -1, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.Exponential(lam=lam, obs=obs)


def crystalball(**kwargs):
    import zfit

    alpha = zfit.Parameter("alphacb", 0.1, -1, 1)
    n = zfit.Parameter("ncb", 0.1, 0, 1)
    mu = zfit.Parameter("mucb", 0.1, -1, 1)
    sigma = zfit.Parameter("sigmacb", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.CrystalBall(alpha=alpha, n=n, mu=mu, sigma=sigma, obs=obs)


def doublecb(**kwargs):
    import zfit

    alphaL = zfit.Parameter("alphaL_dcb", 0.1, -1, 1)
    nL = zfit.Parameter("nL_dcb", 0.1, 0, 1)
    alphaR = zfit.Parameter("alphaR_dcb", 0.1, -1, 1)
    nR = zfit.Parameter("nR_dcb", 0.1, 0, 1)
    mu = zfit.Parameter("mu_dcb", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_dcb", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.DoubleCB(
        alphal=alphaL, nl=nL, alphar=alphaR, nr=nR, mu=mu, sigma=sigma, obs=obs
    )


def legendre(**kwargs):
    import zfit

    obs = zfit.Space("obs", (-3, 3))
    coeffs = [zfit.Parameter(f"coeff{i}_legendre", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Legendre(obs=obs, coeffs=coeffs)


def chebyshev(**kwargs):
    import zfit

    obs = zfit.Space("obs", (-3, 3))
    coeffs = [zfit.Parameter(f"coeff{i}_cheby", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs)


def chebyshev2(**kwargs):
    import zfit

    obs = zfit.Space("obs", (-3, 3))
    coeffs = [zfit.Parameter(f"coeff{i}_cheby2", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Chebyshev2(obs=obs, coeffs=coeffs)


def laguerre(**kwargs):
    import zfit

    obs = zfit.Space("obs", (-3, 3))
    coeffs = [zfit.Parameter(f"coeff{i}_laguerre", 0.1) for i in range(5)]
    return zfit.pdf.Laguerre(obs=obs, coeffs=coeffs)


def hermite(**kwargs):
    import zfit

    obs = zfit.Space("obs", (-3, 3))
    coeffs = [zfit.Parameter(f"coeff{i}_hermite", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Hermite(obs=obs, coeffs=coeffs)


basic_pdfs = [
    gauss,
    cauchy,
    exponential,
    crystalball,
    doublecb,
    legendre,
    chebyshev,
    chebyshev2,
    laguerre,
    hermite,
]
basic_pdfs.reverse()


def sumpdf(pdfs=None, fracs=True, extended=None, **kwargs):
    if pdfs is None:
        pdfs = [pdf() for pdf in basic_pdfs]
    import zfit

    if fracs:
        fracs = [zfit.Parameter(f"frac{i}", 0.1, -1, 1) for i in range(len(pdfs))]
    return zfit.pdf.SumPDF(pdfs=pdfs, fracs=fracs, extended=extended)


def productpdf(pdfs=None, extended=None, **kwargs):
    if pdfs is None:
        pdfs = [pdf() for pdf in basic_pdfs[:3]]
    import zfit

    return zfit.pdf.ProductPDF(pdfs=pdfs, extended=extended)


def complicatedpdf(pdfs=None, extended=None, **kwargs):
    import zfit

    basic_pdfs_made = [pdf() for pdf in basic_pdfs]
    pdfs1 = basic_pdfs_made[:3]
    pdfs2 = basic_pdfs_made[3:5]
    pdfs3 = basic_pdfs_made[5:]
    pdfs4 = basic_pdfs_made[1:4]

    sum1 = zfit.pdf.SumPDF(
        pdfs=pdfs1,
        fracs=[
            zfit.Parameter(f"frac_sum1_{i}", 0.1, -1, 1) for i in range(len(pdfs1) - 1)
        ],
    )
    pdfs2.append(sum1)
    sum2 = zfit.pdf.SumPDF(
        pdfs=pdfs2,
        fracs=[
            zfit.Parameter(f"frac_sum2_{i}", 0.1, -1, 1) for i in range(len(pdfs2) - 1)
        ],
    )
    pdfs3.append(sum2)
    prod1 = zfit.pdf.ProductPDF(pdfs=pdfs3)
    pdfs4.append(prod1)
    sum3 = zfit.pdf.SumPDF(
        pdfs=pdfs4,
        extended=extended,
        fracs=[
            zfit.Parameter(f"frac_sum3_{i}", 0.1, -1, 1) for i in range(len(pdfs4) - 1)
        ],
    )
    return sum3


def convolutionpdf(pdf=None, extended=None, **kwargs):
    kernel = gauss()
    func = cauchy()
    import zfit

    return zfit.pdf.FFTConvPDFV1(kernel=kernel, func=func, extended=extended)


def conditionalpdf(pdf=None, extended=None, **kwargs):
    if pdf is None:
        pdf = gauss()
    import zfit

    return zfit.pdf.ConditionalPDFV1(
        pdf=pdf, cond={pdf.params["sigma"]: pdf.space}, extended=extended
    )


all_pdfs = (
    basic_pdfs
    + [
        sumpdf,
        productpdf,
        convolutionpdf,
        complicatedpdf,
    ]
    + [prod2dgauss]
)


# TODO(serialization): add to serializer
# def test_loss_serialization():
#     import zfit
#
#     for pdf_factory in all_pdfs:
#         pdf = pdf_factory()
#         data = zfit.Data.from_numpy(obs=pdf.space, array=np.random.normal(size=1000))
#         # print(pdf)
#         loss = zfit.loss.UnbinnedNLL(pdf, data)
#         loss_json = loss.to_json()
#         print(loss_json)
#         break


@pytest.mark.parametrize("pdf", all_pdfs, ids=lambda x: x.__name__)
@pytest.mark.parametrize("extended", [True, None], ids=["extended", "not extended"])
def test_serial_hs3_pdfs(pdf, extended):
    import zfit
    import zfit.z.numpy as znp
    import zfit.serialization as zserial

    pdf = pdf(extended=extended)
    scale = zfit.Parameter("yield", 100.0, 0, 1000)
    if extended and not pdf.is_extended:
        pdf.set_yield(scale)

    hs3json = zserial.Serializer.to_hs3(pdf)
    loaded = zserial.Serializer.from_hs3(hs3json)

    loaded_pdf = list(loaded["pdfs"].values())[0]
    assert str(pdf) == str(loaded_pdf)
    x = znp.random.uniform(-3, 3, size=(107, pdf.n_obs))
    assert np.allclose(pdf.pdf(x), loaded_pdf.pdf(x))
    if extended:
        scale.set_value(0.6)
        assert np.allclose(pdf.ext_pdf(x), loaded_pdf.ext_pdf(x))


def test_replace_matching():
    import zfit
    import zfit.serialization as zserial

    original_dict = {
        "metadata": {
            "HS3": {"version": "experimental"},
            "serializer": {"lib": "zfit", "version": str(zfit.__version__)},
        },
        "pdfs": {
            "Gauss": {
                "mu": {
                    "floating": True,
                    "max": 1.0,
                    "min": -1.0,
                    "name": "mu",
                    "step_size": 0.001,
                    "type": "Parameter",
                    "value": 0.10000000149011612,
                },
                "sigma": {
                    "floating": True,
                    "max": 1.0,
                    "min": 0.0,
                    "name": "sigma",
                    "step_size": 0.001,
                    "type": "Parameter",
                    "value": 0.10000000149011612,
                },
                "type": "Gauss",
                "x": {"max": 3.0, "min": -3.0, "name": "obs", "type": "Space"},
            }
        },
        "variables": {
            "mu": {
                "floating": True,
                "max": 1.0,
                "min": -1.0,
                "name": "mu",
                "step_size": 0.001,
                "type": "Parameter",
                "value": 0.10000000149011612,
            },
            "obs": {"max": 3.0, "min": -3.0, "name": "obs", "type": "Space"},
            "sigma": {
                "floating": True,
                "max": 1.0,
                "min": 0.0,
                "name": "sigma",
                "step_size": 0.001,
                "type": "Parameter",
                "value": 0.10000000149011612,
            },
        },
    }

    target_dict = {
        "metadata": {
            "HS3": {"version": "experimental"},
            "serializer": {"lib": "zfit", "version": str(zfit.__version__)},
        },
        "pdfs": {"Gauss": {"mu": "mu", "sigma": "sigma", "type": "Gauss", "x": "obs"}},
        "variables": {
            "mu": {
                "floating": True,
                "max": 1.0,
                "min": -1.0,
                "name": "mu",
                "step_size": 0.001,
                "type": "Parameter",
                "value": 0.10000000149011612,
            },
            "obs": {"max": 3.0, "min": -3.0, "name": "obs", "type": "Space"},
            "sigma": {
                "floating": True,
                "max": 1.0,
                "min": 0.0,
                "name": "sigma",
                "step_size": 0.001,
                "type": "Parameter",
                "value": 0.10000000149011612,
            },
        },
    }

    parameter = frozendict({"name": None, "min": None, "max": None})
    replace_forward = {parameter: lambda x: x["name"]}
    target_test = copy.deepcopy(original_dict)
    target_test["pdfs"] = zserial.serializer.replace_matching(
        target_test["pdfs"], replace_forward
    )
    assert target_test == target_dict
    replace_backward = {
        k: lambda x=k: target_dict["variables"][x]
        for k in target_dict["variables"].keys()
    }
    original_test = copy.deepcopy(target_dict)
    original_test["pdfs"] = zserial.serializer.replace_matching(
        original_test["pdfs"], replace_backward
    )

    assert original_test == original_dict


@pytest.mark.parametrize("pdfcreator", all_pdfs, ids=lambda x: x.__name__)
def test_dumpload_pdf(pdfcreator):
    import zfit.z.numpy as znp

    pdf = pdfcreator()
    param1 = list(pdf.get_params())[0]
    json1 = pdf.to_json()
    gauss2_raw = pdf.__class__.get_repr().parse_raw(json1)
    gauss2 = gauss2_raw.to_orm()
    assert str(pdf) == str(gauss2)
    json2 = gauss2.to_json()
    json1cleaned = json1
    json2cleaned = json2
    for i in range(1000, 0, -1):
        json2cleaned = json2cleaned.replace(f"autoparam_{i}", "autoparam_ANY")
        json1cleaned = json2cleaned.replace(f"autoparam_{i}", "autoparam_ANY")
    assert json1cleaned == json2cleaned  # Just a technicality
    gauss3 = pdf.__class__.get_repr().parse_raw(json2).to_orm()
    x = znp.random.uniform(-3, 3, size=(100, pdf.n_obs))
    assert np.allclose(pdf.pdf(x), gauss3.pdf(x))
    assert np.allclose(gauss2.pdf(x), gauss3.pdf(x))
    param1.set_value(0.6)
    assert np.allclose(pdf.pdf(x), gauss3.pdf(x))
    assert np.allclose(gauss2.pdf(x), gauss3.pdf(x))


param_factories = [
    lambda: zfit.param.ComposedParameter(
        "test", lambda x: x, [zfit.Parameter("x", 1.0, 0.5, 1.4)]
    ),
    lambda: zfit.Parameter("test1", 5.0, 3.0, 10.0),
    lambda: zfit.Parameter("test1", 5.0, 3.0, 10.0, step_size=1.1),
    lambda: zfit.Parameter("test1", 5.0, step_size=1.1),
    lambda: zfit.Parameter("test1", 5.0, 3.0, 10.0, step_size=1.1, floating=False),
    lambda: zfit.Parameter("test1", 5.0, 3.0, 10.0, floating=False),
    lambda: zfit.Parameter("test1", 5.0, floating=True),
    lambda: zfit.param.ConstantParameter("const1", 5.0),
    lambda: zfit.Space("obs", (-3.0, 5.0)),
]


@pytest.mark.parametrize("param_factory", param_factories)
def test_params_dumpload(param_factory):
    param = param_factory()
    json = param.to_json()
    param_loaded = param.get_repr().parse_raw(json).to_orm()
    assert param == param_loaded
    assert json == param_loaded.to_json()


data_factories = [
    lambda: zfit.data.Data.from_numpy(
        obs=zfit.Space("obs1", (-3.0, 5.0)), array=np.random.normal(size=(100, 1))
    ),
    lambda: zfit.data.Data.from_numpy(
        obs=zfit.Space("obs1", (-3.0, 5.0)),
        array=np.random.normal(size=(100, 1)),
        weights=np.random.normal(size=(100, 1)),
    ),
    lambda: zfit.data.Data.from_numpy(
        obs=zfit.Space("obs1", (-3.0, 5.0)) * zfit.Space("obs2", (-13.0, 15.0)),
        array=np.random.normal(size=(100, 2)),
        weights=np.random.normal(size=(100, 1)),
    ),
    lambda: zfit.data.Data.from_numpy(
        obs=zfit.Space("obs1", (-3.0, 5.0)) * zfit.Space("obs2", (-13.0, 15.0)),
        array=np.random.normal(size=(100, 2)),
    ),
]


@pytest.mark.parametrize("data_factory", data_factories)
def test_data_dumpload(data_factory, request):
    data = data_factory()
    asdf_dumped = data.to_asdf()
    data_truth_dumped = pytest.helpers.get_truth(
        "hs3_data", "basic_data_dumpload1.asdf", request, newval=asdf_dumped
    )
    data_loaded1 = data.__class__.from_asdf(
        asdf_dumped
    )  # class just to make sure we don't use the instance method
    asdf_dumped2 = (
        data_loaded1.to_asdf()
    )  # dump and load two times to check consistency
    data_loaded2 = data.__class__.from_asdf(
        asdf_dumped2
    )  # class just to make sure we don't use the instance method
    data_truth_tree = dict(data_truth_dumped.tree)
    data_tree = dict(asdf_dumped.tree)
    data_tree2 = dict(asdf_dumped2.tree)

    for d in [data_truth_tree, data_tree, data_tree2]:
        d.pop("asdf_library", None)
        d.pop("history", None)

    # test and pop all the arrays in the dict, then compare the rest
    true_weigths = data_truth_tree.pop("weights", 1)
    np.testing.assert_allclose(true_weigths, data_tree.pop("weights", 1))
    np.testing.assert_allclose(true_weigths, data_tree2.pop("weights", 1))

    true_data = data_truth_tree.pop("data")
    np.testing.assert_allclose(true_data, data_tree.pop("data"))
    np.testing.assert_allclose(true_data, data_tree2.pop("data"))
    np.testing.assert_allclose(data.value(), data_loaded1.value())
    np.testing.assert_allclose(data.value(), data_loaded2.value())
    assert (data.weights is None) == (data_loaded1.weights is None)
    assert (data.weights is None) == (data_loaded2.weights is None)
    if data.weights is not None:
        np.testing.assert_allclose(data.weights, data_loaded1.weights)
        np.testing.assert_allclose(data.weights, data_loaded2.weights)
    with pytest.raises(
        TypeError, match="The object you are trying to serialize contains numpy arrays."
    ):
        data.to_json()
    with pytest.raises(
        TypeError, match="The object you are trying to serialize contains numpy arrays."
    ):
        data.to_yaml()
    with pytest.raises(
        TypeError, match="The object you are trying to serialize contains numpy arrays."
    ):
        data_loaded2.to_json()
    with pytest.raises(
        TypeError, match="The object you are trying to serialize contains numpy arrays."
    ):
        data_loaded2.to_yaml()
