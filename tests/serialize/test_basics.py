#  Copyright (c) 2024 zfit
import copy

import numpy as np
import pytest
from frozendict import frozendict

import zfit

rndgen = np.random.Generator(np.random.PCG64(8213))
rndgens = [np.random.Generator(np.random.PCG64(8213 + i)) for i in range(10)]

default_limits = (-2.1, 3)
positive_limits = (0.5, 4)


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
    param2 = zfit.Parameter.get_repr().parse_raw(json1).to_orm(reuse_params=[param])

    assert param == param2
    json2 = param2.to_json()
    assert json1 == json2
    param3 = zfit.Parameter.get_repr().parse_raw(json2).to_orm(reuse_params=[param])
    assert param2 == param3
    assert param == param3


def gauss(extended=None, **kwargs):
    import zfit

    mu = zfit.Parameter("mu_gauss", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_gauss", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.Gauss(
        mu=mu, sigma=sigma, obs=obs, extended=extended, name="MyGaussName"
    )


def prod2dgauss(extended=None, **kwargs):
    import zfit

    mu = zfit.Parameter("mu_gauss", 0.1, -1, 1)
    sigma1 = zfit.Parameter("sigma_gauss", 0.1, 0, 1)
    obs1 = zfit.Space("obs1", (-3, 7))
    sigma2 = zfit.Parameter("sigma_gauss2", 0.1, 0, 1)
    obs2 = zfit.Space("obs2", (-13, 5))
    gauss1 = zfit.pdf.Gauss(mu=mu, sigma=sigma1, obs=obs1)
    gauss2 = zfit.pdf.Gauss(mu=mu, sigma=sigma2, obs=obs2)
    prod = zfit.pdf.ProductPDF([gauss1, gauss2], extended=extended, name="prod2dgauss")
    return prod


def cauchy(extended=None, **kwargs):
    import zfit

    m = zfit.Parameter("m_cauchy", 0.1, -1, 1)
    gamma = zfit.Parameter("gamma_cauchy", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.Cauchy(m=m, gamma=gamma, obs=obs, extended=extended)


def voigt(extended=None, **kwargs):
    import zfit

    m = zfit.Parameter("m_voigt", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_voigt", 0.1, 0, 1)
    gamma = zfit.Parameter("gamma_voigt", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.Voigt(m=m, sigma=sigma, gamma=gamma, obs=obs, extended=extended)


def exponential(extended=None, **kwargs):
    import zfit

    lam = zfit.Parameter("lambda_exp", 0.1, -1, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.Exponential(lam=lam, obs=obs, extended=extended)


def crystalball(extended=None, **kwargs):
    import zfit

    alpha = zfit.Parameter("alphacb", 0.1, -1, 1)
    n = zfit.Parameter("ncb", 0.1, 0, 1)
    mu = zfit.Parameter("mucb", 0.1, -1, 1)
    sigma = zfit.Parameter("sigmacb", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.CrystalBall(
        alpha=alpha, n=n, mu=mu, sigma=sigma, obs=obs, extended=extended
    )


def doublecb(extended=None, **kwargs):
    import zfit

    alphaL = zfit.Parameter("alphaL_dcb", 0.1, -1, 1)
    nL = zfit.Parameter("nL_dcb", 0.1, 0, 1)
    alphaR = zfit.Parameter("alphaR_dcb", 0.1, -1, 1)
    nR = zfit.Parameter("nR_dcb", 0.1, 0, 1)
    mu = zfit.Parameter("mu_dcb", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_dcb", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.DoubleCB(
        alphal=alphaL,
        nl=nL,
        alphar=alphaR,
        nr=nR,
        mu=mu,
        sigma=sigma,
        obs=obs,
        extended=extended,
    )


def generalizedcb(extended=None, **kwargs):
    import zfit

    sigmaL = zfit.Parameter("sigmaL_gcb", 0.1, 0, 1)
    alphaL = zfit.Parameter("alphaL_gcb", 0.1, -1, 1)
    nL = zfit.Parameter("nL_gcb", 0.1, 0, 1)
    sigmaR = zfit.Parameter("sigmaR_gcb", 0.1, 0, 1)
    alphaR = zfit.Parameter("alphaR_gcb", 0.1, -1, 1)
    nR = zfit.Parameter("nR_gcb", 0.1, 0, 1)
    mu = zfit.Parameter("mu_gcb", 0.1, -1, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.GeneralizedCB(
        sigmal=sigmaL,
        alphal=alphaL,
        nl=nL,
        sigmar=sigmaR,
        alphar=alphaR,
        nr=nR,
        mu=mu,
        obs=obs,
        extended=extended,
    )

def gaussexptail(extended=None, **kwargs):
    import zfit

    alpha = zfit.Parameter("alpha_gaussexptail", 0.1, -1, 1)
    mu = zfit.Parameter("mu_gaussexptail", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_gaussexptail", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.GaussExpTail(
        alpha=alpha, mu=mu, sigma=sigma, obs=obs, extended=extended
    )

def generalizedgaussexptail(extended=None, **kwargs):
    import zfit

    sigmaL = zfit.Parameter("sigmaL_generalizedgaussexptail", 0.1, 0, 1)
    alphaL = zfit.Parameter("alphaL_generalizedgaussexptail", 0.1, -1, 1)
    sigmaR = zfit.Parameter("sigmaR_generalizedgaussexptail", 0.1, 0, 1)
    alphaR = zfit.Parameter("alphaR_generalizedgaussexptail", 0.1, -1, 1)
    mu = zfit.Parameter("mu_generalizedgaussexptail", 0.1, -1, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.GeneralizedGaussExpTail(
        sigmal=sigmaL,
        alphal=alphaL,
        sigmar=sigmaR,
        alphar=alphaR,
        mu=mu,
        obs=obs,
        extended=extended,
    )

def legendre(extended=None, **kwargs):
    import zfit

    obs = zfit.Space("obs", default_limits)
    coeffs = [zfit.Parameter(f"coeff{i}_legendre", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Legendre(obs=obs, coeffs=coeffs, extended=extended)


def chebyshev(extended=None, **kwargs):
    import zfit

    obs = zfit.Space("obs", default_limits)
    coeffs = [zfit.Parameter(f"coeff{i}_cheby", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs, extended=extended)


def chebyshev2(extended=None, **kwargs):
    import zfit

    obs = zfit.Space("obs", default_limits)
    coeffs = [zfit.Parameter(f"coeff{i}_cheby2", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Chebyshev2(obs=obs, coeffs=coeffs, extended=extended)


def laguerre(extended=None, **kwargs):
    import zfit

    obs = zfit.Space("obs", default_limits)
    coeffs = [zfit.Parameter(f"coeff{i}_laguerre", 0.1) for i in range(5)]
    return zfit.pdf.Laguerre(obs=obs, coeffs=coeffs, extended=extended)


def hermite(extended=None, **kwargs):
    import zfit

    obs = zfit.Space("obs", default_limits)
    coeffs = [zfit.Parameter(f"coeff{i}_hermite", 0.1, -1, 1) for i in range(5)]
    return zfit.pdf.Hermite(obs=obs, coeffs=coeffs, extended=extended)


def bernstein(extended=None, **kwargs):
    import zfit

    obs = zfit.Space("obs", default_limits)
    coeffs = [zfit.Parameter(f"coeff{i}_bernstein", 0.1, 0, 1) for i in range(5)]
    return zfit.pdf.Bernstein(obs=obs, coeffs=coeffs, extended=extended)


def poisson(extended=None, **kwargs):
    import zfit

    lambda_ = zfit.Parameter("lambda_poisson", 0.1, 0, 1)
    obs = zfit.Space("obs", positive_limits)
    return zfit.pdf.Poisson(lamb=lambda_, obs=obs, extended=extended)


def lognormal(extended=None, **kwargs):
    import zfit

    mu = zfit.Parameter("mu_lognormal", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_lognormal", 0.1, 0, 1)
    obs = zfit.Space("obs", positive_limits)
    return zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs, extended=extended)


def bifurgauss(extended=None, **kwargs):
    import zfit

    mu = zfit.Parameter("mu_bifurgauss", 0.1, -1, 1)
    sigmaL = zfit.Parameter("sigmaL_bifurgauss", 0.1, 0, 1)
    sigmaR = zfit.Parameter("sigmaR_bifurgauss", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.BifurGauss(
        mu=mu, sigmal=sigmaL, sigmar=sigmaR, obs=obs, extended=extended
    )

def qgauss(extended=None, **kwargs):
    import zfit

    q = zfit.Parameter("q_qgauss", 2, 1, 3)
    mu = zfit.Parameter("mu_qgauss", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_qgauss", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.QGauss(q=q, mu=mu, sigma=sigma, obs=obs, extended=extended)


def chisquared(extended=None, **kwargs):
    import zfit

    ndof = zfit.Parameter("ndof_chisquared", 4, 1, 10)
    obs = zfit.Space("obs", positive_limits)
    return zfit.pdf.ChiSquared(ndof=ndof, obs=obs, extended=extended)


def studentt(extended=None, **kwarfs):
    import zfit

    ndof = zfit.Parameter("ndof_studentt", 4, 1, 10)
    mu = zfit.Parameter("mu_studentt", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma_studentt", 0.1, 0, 1)
    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.StudentT(ndof=ndof, mu=mu, sigma=sigma, obs=obs, extended=extended)


def gamma(extended=None, **kwargs):
    import zfit

    gamma = zfit.Parameter("gamma_gamma", 4, 1, 10)
    beta = zfit.Parameter("beta_gamma", 0.1, 0, 1)
    mu = zfit.Parameter("mu_gamma", -1, -3, -0.1)
    obs = zfit.Space("obs", positive_limits)
    return zfit.pdf.Gamma(gamma=gamma, beta=beta, mu=mu, obs=obs, extended=extended)

def kde1dimexact(pdfs=None, extended=None, **kwargs):
    data = np.array(
        [
            0.5510963,
            0.46542518,
            -1.3578327,
            -1.1137842,
            0.30369744,
            -0.49643811,
            0.68624974,
            0.2091461,
            -0.32975698,
            0.71040372,
            1.95709914,
            -0.0430226,
            1.28575821,
            -0.37646677,
            -0.25665507,
            1.67085858,
            0.05659317,
            0.52634881,
            0.58759272,
            -0.24556029,
            -0.74070669,
        ]
    )
    import zfit

    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.KDE1DimExact(data, obs=obs, extended=extended)


def kde1dgrid(pdfs=None, extended=None, **kwargs):
    data = np.array(
        [
            0.5510963,
            0.46542518,
            -1.3578327,
            -1.1137842,
            0.30369744,
            -0.49643811,
            0.68624974,
            0.2091461,
            -0.32975698,
            0.71040372,
            1.95709914,
            -0.0430226,
            1.28575821,
            -0.37646677,
            -0.25665507,
            1.67085858,
            0.05659317,
            0.52634881,
            0.58759272,
            -0.24556029,
            -0.74070669,
        ]
    )
    import zfit

    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.KDE1DimGrid(data, obs=obs, extended=extended, num_grid_points=512)


def kde1dfft(pdfs=None, extended=None, **kwargs):
    data = np.array(
        [
            0.5510963,
            0.46542518,
            -1.3578327,
            -1.1137842,
            0.30369744,
            -0.49643811,
            0.68624974,
            0.2091461,
            -0.32975698,
            0.71040372,
            1.95709914,
            -0.0430226,
            1.28575821,
            -0.37646677,
            -0.25665507,
            1.67085858,
            0.05659317,
            0.52634881,
            0.58759272,
            -0.24556029,
            -0.74070669,
        ]
    )
    import zfit

    obs = zfit.Space("obs", default_limits)
    return zfit.pdf.KDE1DimFFT(data=data, obs=obs, extended=extended, weights=data**2)


def kde1disj(pdfs=None, extended=None, **kwargs):
    data = np.array(
        [
            0.5510963,
            0.46542518,
            -1.3578327,
            -1.1137842,
            0.30369744,
            -0.49643811,
            0.68624974,
            0.2091461,
            -0.32975698,
            0.71040372,
            1.95709914,
            -0.0430226,
            1.28575821,
            -0.37646677,
            -0.25665507,
            1.67085858,
            0.05659317,
            0.52634881,
            0.58759272,
            -0.24556029,
            -0.74070669,
        ]
    )
    import zfit

    obs = zfit.Space("obs", default_limits)
    data = zfit.data.Data.from_numpy(obs=obs, array=data)
    return zfit.pdf.KDE1DimFFT(data=data, extended=extended)


basic_pdfs = [
    gauss,
    qgauss,
    bifurgauss,
    cauchy,
    voigt,
    exponential,
    studentt,
    crystalball,
    doublecb,
    generalizedcb,
    gaussexptail,
    generalizedgaussexptail,
    legendre,
    bernstein,
    chebyshev,
    chebyshev2,
    laguerre,
    hermite,
    kde1dimexact,
    kde1dgrid,
    kde1dfft,
    kde1disj,
]
basic_pdfs.reverse()
positive_pdfs = [poisson, lognormal, chisquared, gamma]


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


def cachedpdf(pdfs=None, extended=None, **kwargs):
    if pdfs is None:
        pdf = basic_pdfs[0]()
    else:
        pdf = pdfs[0]
    import zfit

    return zfit.pdf.CachedPDF(pdf=pdf, extended=extended)


def truncpdf(pdfs=None, extended=None, **kwargs):
    if pdfs is None:
        pdfs = basic_pdfs
    pdf = pdfs[0]()
    obs = pdf.obs[0]
    space1 = zfit.Space(obs, (-1, 1))
    space2 = zfit.Space(obs, (2, 3))
    return zfit.pdf.TruncatedPDF(pdf=pdf, extended=extended, limits=[space1, space2])


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
        name="complicatedpdf",

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
    + positive_pdfs
    + [
        sumpdf,
        productpdf,
        convolutionpdf,
        # conditionalpdf,  # not working currently, see implementation
        truncpdf,
        complicatedpdf,
        cachedpdf,
    ]
    + [prod2dgauss]
)

all_constraints = [
    lambda: zfit.constraint.GaussianConstraint(
        params=[zfit.Parameter("mu", 0.0, -1, 1), zfit.Parameter("sigma", 1.0, 0, 10)],
        observation=[0.0, 1.0],
        uncertainty=[0.1, 0.5],
    ),
    lambda: zfit.constraint.PoissonConstraint(
        params=[zfit.Parameter("mu", 0.1, -1, 1), zfit.Parameter("sigma", 1.0, 0, 10)],
        observation=[0.1, 1.2],
    ),
    lambda: zfit.constraint.LogNormalConstraint(
        params=[zfit.Parameter("mu", 0.1, -1, 1), zfit.Parameter("sigma", 1.0, 0, 10)],
        observation=[0.1, 1.2],
        uncertainty=[0.1, 0.5],
    ),
]


@pytest.mark.parametrize(
    "ext_Loss",
    [(False, zfit.loss.UnbinnedNLL), (True, zfit.loss.ExtendedUnbinnedNLL)],
    ids=lambda x: x[1].__name__,
)
@pytest.mark.parametrize("pdf_factory", all_pdfs, ids=lambda x: x.__name__)
@pytest.mark.parametrize(
    "Constraint", [None] + all_constraints, ids=lambda x: (x is not None) and x.__name__
)
def test_loss_serialization(ext_Loss, pdf_factory, Constraint, request):
    import zfit

    extended, Loss = ext_Loss
    pdf = pdf_factory(extended=extended)
    assert pdf.is_extended == extended, "Error in testing setup"
    data = zfit.Data.from_numpy(
        obs=pdf.space, array=rndgens[0].normal(size=(1000, pdf.n_obs))
    )
    constraint = Constraint() if Constraint is not None else None
    loss = Loss(pdf, data, constraints=constraint)
    loss_dict = loss.to_dict()
    loss_asdf = loss.to_asdf()

    loss_truth_dumped = pytest.helpers.get_truth(
        f"hs3_loss/{loss.name}",
        f"{loss.name}_{pdf.name}_{'unconstr' if constraint is None else constraint.name}.asdf",
        request,
        newval=loss_asdf,
    )
    loss_truth_tree = loss_truth_dumped.tree
    loss_asdf_tree = loss_asdf.tree
    for l in [loss_asdf_tree, loss_truth_tree]:
        l.pop("asdf_library", None)
        l.pop("history", None)

    # assert loss_asdf_tree == loss_truth_tree
    loss_loaded = loss.from_asdf(loss_asdf)
    # assert loss_loaded == loss
    # TODO: improve tests!

    print(loss_loaded)


@pytest.mark.parametrize("Constraint", all_constraints, ids=lambda x: x.__name__)
def test_constraint_dumpload(Constraint, request):
    constraint = Constraint()
    constraint_dict = constraint.to_dict()
    constraint_json = constraint.to_json()
    constraint_loaded_truth = pytest.helpers.get_truth(
        "hs3_constraint", f"{constraint.name}.json", request, newval=constraint_json
    )
    constraint_loaded = constraint.__class__.from_json(constraint_json).to_dict()
    constraint_loaded_truth["library"] = "ZFIT_ARBITRARY_VALUE"
    (
        constraint_loaded_cleaned,
        constraint_loaded_truth_cleaned,
    ) = pytest.helpers.cleanup_hs3(constraint_loaded, constraint_loaded_truth)
    assert constraint_loaded_cleaned == constraint_loaded_truth_cleaned


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
    loaded = zserial.Serializer.from_hs3(hs3json, reuse_params=pdf.get_params())

    loaded_pdf = list(loaded["distributions"].values())[0]
    assert str(pdf) == str(loaded_pdf)
    lower, upper = pdf.space.v1.lower, pdf.space.v1.upper
    x = znp.random.uniform(lower, upper, size=(107, pdf.n_obs))
    np.testing.assert_allclose(pdf.pdf(x), loaded_pdf.pdf(x))
    if extended:
        scale.set_value(0.6)
        np.testing.assert_allclose(pdf.ext_pdf(x), loaded_pdf.ext_pdf(x))


def test_replace_matching():
    import zfit
    import zfit.serialization as zserial

    original_dict = {
        "metadata": {
            "HS3": {"version": "experimental"},
            "serializer": {"lib": "zfit", "version": str(zfit.__version__)},
        },
        "distributions": {
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
        "distributions": {
            "Gauss": {"mu": "mu", "sigma": "sigma", "type": "Gauss", "x": "obs"}
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

    parameter = frozendict({"name": None, "min": None, "max": None})
    replace_forward = {parameter: lambda x: x["name"]}
    target_test = copy.deepcopy(original_dict)
    target_test["distributions"] = zserial.serializer.replace_matching(
        target_test["distributions"], replace_forward
    )
    assert target_test == target_dict
    replace_backward = {
        k: lambda x=k: target_dict["variables"][x]
        for k in target_dict["variables"].keys()
    }
    original_test = copy.deepcopy(target_dict)
    original_test["distributions"] = zserial.serializer.replace_matching(
        original_test["distributions"], replace_backward
    )

    assert original_test == original_dict


@pytest.mark.parametrize("pdfcreator", all_pdfs, ids=lambda x: x.__name__)
@pytest.mark.parametrize("reuse_params", [True, False])
def test_dumpload_pdf(pdfcreator, reuse_params):
    import zfit.z.numpy as znp

    pdf = pdfcreator()
    params = list(pdf.get_params())
    if len(params) > 0:
        param1 = params[0]
    else:
        param1 = None
    json1 = pdf.to_dict()
    gauss2 = type(pdf).from_dict(json1, reuse_params=params)
    gauss2noshared = type(pdf).from_dict(json1)
    try:
        json1 = pdf.to_json()
    except zfit.exception.NumpyArrayNotSerializableError:
        pass  # KDEs or similar
    else:
        gauss2 = pdf.get_repr().parse_raw(json1).to_orm(reuse_params=params)

    assert str(pdf) == str(gauss2)

    json2 = gauss2.to_dict()
    gauss3 = type(pdf).from_dict(json2, reuse_params=params)
    try:
        json1 = pdf.to_json()
        json2 = gauss2.to_json()
    except zfit.exception.NumpyArrayNotSerializableError:
        pass  # KDEs or similarjson1
    else:
        gauss3 = pdf.get_repr().parse_raw(json2).to_orm(reuse_params=params)

        json1cleaned = json1
        json2cleaned = json2
        for i in range(1000, 0, -1):
            json2cleaned = json2cleaned.replace(f"autoparam_{i}", "autoparam_ANY")
            json1cleaned = json2cleaned.replace(f"autoparam_{i}", "autoparam_ANY")
        assert json1cleaned == json2cleaned  # Just a technicality

    lower, upper = pdf.space.v1.lower, pdf.space.v1.upper
    x = znp.random.uniform(lower, upper, size=(1000, pdf.n_obs))
    true_y = pdf.pdf(x)
    gauss3_y = gauss3.pdf(x)
    np.testing.assert_allclose(true_y, gauss3_y)
    np.testing.assert_allclose(gauss2.pdf(x), gauss3_y)
    if param1 is not None:
        with param1.set_value(param1.value() + 0.3):
            gauss3_y = gauss3.pdf(x)
            np.testing.assert_allclose(pdf.pdf(x), gauss3_y)
            gauss2_y = gauss2.pdf(x)
            np.testing.assert_allclose(gauss2_y, gauss3_y)
            assert not np.allclose(
                gauss2noshared.pdf(x), gauss2_y
            )  # param1 changed only for gauss2
            np.testing.assert_allclose(gauss2noshared.pdf(x), true_y)


param_factories = [
    lambda: zfit.param.ComposedParameter(
        "test", func=lambda x: x, params=[zfit.Parameter("x", 1.0, 0.5, 1.4)]
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
    assert param.name == param_loaded.name
    if isinstance(param, zfit.Space):
        assert pytest.approx(param.lower) == param_loaded.lower
        assert pytest.approx(param.upper) == param_loaded.upper
    else:
        assert pytest.approx(param.value()) == param_loaded.value()
        if isinstance(param, zfit.Parameter):
            assert param.floating == param_loaded.floating
            assert pytest.approx(param.lower) == param_loaded.lower
            assert pytest.approx(param.upper) == param_loaded.upper
            assert pytest.approx(param.step_size) == param_loaded.step_size
    assert json == param_loaded.to_json()


data_factories = enumerate(
    [
        lambda: zfit.data.Data.from_numpy(
            obs=zfit.Space("obs1", (-3.0, 5.0)), array=rndgens[1].normal(size=(100, 1))
        ),
        lambda: zfit.data.Data.from_numpy(
            obs=zfit.Space("obs1", (-3.0, 5.0)),
            array=rndgens[2].normal(size=(100, 1)),
            weights=rndgens[2].normal(size=(100,)),
        ),
        lambda: zfit.data.Data.from_numpy(
            obs=zfit.Space("obs1", (-3.0, 5.0)) * zfit.Space("obs2", (-13.0, 15.0)),
            array=rndgens[3].normal(size=(100, 2)),
            weights=rndgens[3].normal(size=(100,)),
        ),
        lambda: zfit.data.Data.from_numpy(
            obs=zfit.Space("obs1", (-3.0, 5.0)) * zfit.Space("obs2", (-13.0, 15.0)),
            array=rndgens[4].normal(size=(100, 2)),
        ),
    ]
)


@pytest.mark.parametrize("data_factory", data_factories)
def test_data_dumpload(data_factory, request):
    i, factory = data_factory
    data = factory()
    asdf_dumped = data.to_asdf()
    data_truth_dumped = pytest.helpers.get_truth(
        "hs3_data", f"basic_data_dumpload_{i}.asdf", request, newval=asdf_dumped
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


def test_multiple_obs_serialization():
    lower1, upper1 = -3.0, 5.0
    obs1 = zfit.Space("obs", (lower1, upper1))
    lower2, upper2 = -13.0, 15.0
    obs2 = zfit.Space("obs", (lower2, upper2))

    mu = zfit.Parameter("mu", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma", 1)
    sigma2 = zfit.Parameter("sigma2", 1)

    gauss1 = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs1)
    gauss2 = zfit.pdf.Gauss(mu=mu, sigma=sigma2, obs=obs2)
    sumpdf = zfit.pdf.SumPDF([gauss1, gauss2], fracs=0.3, obs=obs1, name="SumPDF")

    hs3pdf = zfit.hs3.dumps(sumpdf)
    pdf = zfit.hs3.loads(hs3pdf)["distributions"]["SumPDF"]
    assert pdf.pdfs[0].space.limit1d == (lower1, upper1)
    assert pdf.pdfs[1].space.limit1d == (lower2, upper2)
    assert pdf.space.limit1d == (lower1, upper1)
