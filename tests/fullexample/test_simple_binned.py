#  Copyright (c) 2024 zfit
import json
import time

# In[2]:
import zfit
import zfit.z.numpy as znp
from zfit.models.binned_functor import BinnedSumPDF
from zfit.models.template import BinnedTemplatePDFV1


def test_simple_examples_1D():
    import zfit.data
    import zfit.z.numpy as znp

    bkgnp = [50.0, 60.0]
    signp = [5.0, 10.0]
    datanp = [60.0, 80.0]
    uncnp = [5.0, 12.0]

    serialized = (
            """{
                                "channels": [
                                    { "name": "singlechannel",
                                      "samples": [
                                        { "name": "signal",
                                        """
            + f"""

              "data": {signp},
              """
              """
                                                                    "modifiers": [ { "name": "mu", "type": "normfactor", "data": null} ]
                                                                  },
                                                                  { "name": "background",
                                                                  """
              f'"data": {bkgnp},'
              """
                                                                    "modifiers": [ {"name": "uncorr_bkguncrt", "type": "shapesys",
                                                                    """
              f'"data": {uncnp}'
              """
                                                                } ]
                                                              }
                                                            ]
                                                          }
                                                      ],
                                                      "observations": [
                                                          {
                                                          """
              f'"name": "singlechannel", "data": {datanp}'
              """
                                                                  }
                                                              ],
                                                              "measurements": [
                                                                  { "name": "Measurement", "config": {"poi": "mu", "parameters": []} }
                                                              ],
                                                              "version": "1.0.0"
                                                              }"""
    )

    obs = zfit.Space(
        "signal", binning=zfit.binned.RegularBinning(2, 0, 2, name="signal")
    )
    zdata = zfit.data.BinnedData.from_tensor(obs, datanp)
    zmcsig = zfit.data.BinnedData.from_tensor(obs, signp)
    zmcbkg = zfit.data.BinnedData.from_tensor(obs, bkgnp)

    shapesys = {
        f"shapesys_{i}": zfit.Parameter(f"shapesys_{i}", 1, 0.1, 10) for i in range(2)
    }
    bkgmodel = BinnedTemplatePDFV1(zmcbkg, sysshape=shapesys)
    # sigyield = zfit.Parameter('sigyield', znp.sum(zmcsig.values()))
    mu = zfit.Parameter("mu", 1, 0.1, 10)
    # sigmodeltmp = BinnedTemplatePDFV1(zmcsig)
    sigyield = zfit.ComposedParameter(
        "sigyield",
        lambda params: params["mu"] * znp.sum(zmcsig.values()),
        params={"mu": mu},
    )
    sigmodel = BinnedTemplatePDFV1(zmcsig, extended=sigyield)
    zmodel = BinnedSumPDF([sigmodel, bkgmodel])
    unc = np.array(uncnp) / np.array(bkgnp)
    nll = zfit.loss.ExtendedBinnedNLL(
        zmodel,
        zdata,
        constraints=zfit.constraint.GaussianConstraint(
            list(shapesys.values()), [1, 1], sigma=unc
        ),
    )
    # print(nll.value())
    # print(nll.gradient())
    # minimizer = zfit.minimize.ScipyLBFGSBV1()
    # minimizer = zfit.minimize.IpyoptV1()
    minimizer = zfit.minimize.Minuit(tol=1e-5, gradient=False)
    result = minimizer.minimize(nll)
    result.hesse(method="hesse_np")
    # result.errors()
    _ = str(result)
    # mu_z = sigmodel.get_yield() / znp.sum(zmcsig.values())
    zbestfit = np.asarray(result.params)
    errors = np.array([p["hesse"]["error"] for p in result.params.values()])
    # print('minval actual:', nll.value(), nll.gradient())
    # errors = np.ones(3) * 0.1
    # print('mu:', mu_z)

    spec = json.loads(serialized)

    workspace = pyhf.Workspace(spec)
    model = workspace.model(poi_name="mu")

    pars = model.config.suggested_init()
    data = workspace.data(model)

    model.logpdf(pars, data)

    bestfit_pars, twice_nll = pyhf.infer.mle.fit(data, model, return_fitted_val=True)
    diff = (bestfit_pars - zbestfit) / errors
    # print(bestfit_pars)
    np.testing.assert_allclose(diff, 0, atol=1e-3)

    # print(-2 * model.logpdf(bestfit_pars, data), twice_nll)


import pyhf
from pyhf.simplemodels import uncorrelated_background
import numpy as np
import pytest


def generate_source_static(n_bins):
    """Create the source structure for the given number of bins.

    Args:
        n_bins: `list` of number of bins

    Returns:
        source
    """
    scale = 10
    binning = [n_bins, -0.5, n_bins + 0.5]
    data = np.random.poisson(size=n_bins, lam=scale * 120).tolist()
    bkg = np.random.poisson(size=n_bins, lam=scale * 100).tolist()
    bkgerr = np.random.normal(size=n_bins, loc=scale * 10.0, scale=3).tolist()
    sig = np.random.poisson(size=n_bins, lam=scale * 30).tolist()

    source = {
        "binning": binning,
        "bindata": {"data": data, "bkg": bkg, "bkgerr": bkgerr, "sig": sig},
    }
    return source


def hypotest_pyhf(pdf, data):
    return pyhf.infer.mle.fit(
        data, pdf, pdf.config.suggested_init(), pdf.config.suggested_bounds()
    )


def hypotest_zfit(minimizer, nll):
    params = nll.get_params()
    init_vals = np.array(params)
    _ = minimizer.minimize(nll)
    zfit.param.set_values(params, init_vals)


bins = [
    # 1,
    3,
    # 10,
    30,
    # 100,
    # 300,
    # 400,
    # 700,
    # 1000,
    # 3000,
]
bin_ids = [f"{n_bins}_bins" for n_bins in bins]


class CachedNLLConstructor:

    def __init__(self):
        self.nlls = {}

    def __call__(self, n_bins, *args, **kwargs):
        if n_bins not in self.nlls:
            self.nlls[n_bins] = self._create_zfit_nll(n_bins, *args, **kwargs)

        nll, param_values = self.nlls[n_bins]
        zfit.param.set_values(param_values)
        return nll

    def _create_zfit_nll(self, n_bins, bkgnp, datanp, obs, signp, uncnp):
        zdata = zfit.data.BinnedData.from_tensor(obs, datanp)
        zmcsig = zfit.data.BinnedData.from_tensor(obs, signp)
        zmcbkg = zfit.data.BinnedData.from_tensor(obs, bkgnp)
        shapesys = {
            f"shapesys_{i}": zfit.Parameter(f"shapesys_{i}", 1, 0.1, 10)
            for i in range(n_bins)
        }
        bkgmodel = BinnedTemplatePDFV1(zmcbkg, sysshape=shapesys)
        # sigyield = zfit.Parameter('sigyield', znp.sum(zmcsig.values()))
        mu = zfit.Parameter("mu", 1, 0.1, 10)
        # sigmodeltmp = BinnedTemplatePDFV1(zmcsig)
        sigyield = zfit.ComposedParameter(
            "sigyield",
            lambda params: params["mu"] * znp.sum(zmcsig.values()),
            params={"mu": mu},
        )
        sigmodel = BinnedTemplatePDFV1(zmcsig, extended=sigyield)
        zmodel = BinnedSumPDF([sigmodel, bkgmodel])
        unc = np.array(uncnp) / np.array(bkgnp)
        constraint = zfit.constraint.GaussianConstraint(
            list(shapesys.values()), np.ones_like(unc).tolist(), sigma=unc
        )
        nll = zfit.loss.ExtendedBinnedNLL(
            zmodel,
            zdata,
            constraints=constraint,
            # options={"numhess": False}
        )
        param_values = {p: float(p.value()) for p in nll.get_params()}
        return nll, param_values


create_zfit_nll = CachedNLLConstructor()


@pytest.mark.benchmark(
    # group="group-name",
    # min_time=0.1,
    max_time=6,
    min_rounds=1,

    # timer=time.time,
    disable_gc=False,
    warmup=False,
    warmup_iterations=1,
)
@pytest.mark.parametrize("n_bins", bins, ids=bin_ids)
@pytest.mark.parametrize(
    "hypotest",
    [
        "pyhf:numpy:", "pyhf:jax:",
        "zfit::minuit",
        "zfit::minuit1",
        "zfit::minuit2",
        "zfit::minuitzgrad",
        "zfit::minuitzgrad1",
        "zfit::minuitzgrad2",
        # "zfit::lm",  # TODO: reactivate once added
        "zfit::nloptmma", "zfit::nloptlbfgs",
        "zfit::scipylbfgs", "zfit::scipytrustconstr",
        "zfit::ipopt",
        "zfit::newtoncg",
        "zfit::truncnc",
        "zfit::nlopttruncnc",
        "zfit::shiftvar",

    ],
)
@pytest.mark.parametrize(
    "eager", [
        False,
        # True
    ], ids=lambda x: "eager" if x else "graph"
)
@pytest.mark.flaky(reruns=1)
def test_hypotest(benchmark, n_bins, hypotest, eager):
    """Benchmark the performance of pyhf.utils.hypotest() for various numbers of bins and different backends.

    Args:
        benchmark: pytest benchmark
        backend: `pyhf` tensorlib given by pytest parameterization
        n_bins: `list` of number of bins given by pytest parameterization

    Returns:
        None
    """
    source = generate_source_static(n_bins)

    signp = source["bindata"]["sig"]
    bkgnp = source["bindata"]["bkg"]
    uncnp = source["bindata"]["bkgerr"]
    datanp = source["bindata"]["data"]
    hypotest_orig = hypotest
    if "pyhf" in hypotest:
        backend = hypotest.split(":")[1]
        hypotest = hypotest_pyhf
        if backend == "jax":

            try:
                import jax
            except ImportError:
                return

        pyhf.set_backend(backend)

        pdf = uncorrelated_background(signp, bkgnp, uncnp)
        data = datanp + pdf.config.auxdata

        # warmup
        pyhf.infer.mle.fit(
            data, pdf, pdf.config.suggested_init(), pdf.config.suggested_bounds()
        )
        benchmark(hypotest, pdf, data)
    elif "zfit" in hypotest:
        with zfit.run.set_graph_mode(not eager):
            minimizer = hypotest.split(":")[2]
            hypotest = hypotest_zfit
            obs = zfit.Space(
                "signal",
                binning=zfit.binned.RegularBinning(
                    n_bins, -0.5, n_bins + 0.5, name="signal"
                ),
            )
            nll = create_zfit_nll(bkgnp=bkgnp, datanp=datanp, n_bins=n_bins, obs=obs, signp=signp, uncnp=uncnp)
            if minimizer == "minuit":
                minimizer = zfit.minimize.Minuit(tol=1e-3, gradient=True, mode=0, verbosity=7)
            elif minimizer == "minuit1":
                minimizer = zfit.minimize.Minuit(tol=1e-3, gradient=True, mode=1, verbosity=7)
            elif minimizer == "minuit2":
                minimizer = zfit.minimize.Minuit(tol=1e-3, gradient=True, mode=2, verbosity=7)
            elif minimizer == "minuitzgrad":
                minimizer = zfit.minimize.Minuit(tol=1e-3, gradient=False, mode=0, verbosity=7)
            elif minimizer == "minuitzgrad1":
                minimizer = zfit.minimize.Minuit(tol=1e-3, gradient=False, mode=1, verbosity=7)
            elif minimizer == "minuitzgrad2":
                minimizer = zfit.minimize.Minuit(tol=1e-3, gradient=False, mode=2, verbosity=7)
            elif minimizer == "nloptmma":
                nlopt = pytest.importorskip("nlopt")
                minimizer = zfit.minimize.NLoptMMA(tol=1e-3, verbosity=7)
            elif minimizer == "nloptlbfgs":
                nlopt = pytest.importorskip("nlopt")
                minimizer = zfit.minimize.NLoptLBFGS(verbosity=7)
            elif minimizer == "scipylbfgs":
                minimizer = zfit.minimize.ScipyLBFGSB(verbosity=7)
            elif minimizer == "scipytrustconstr":
                minimizer = zfit.minimize.ScipyTrustConstr(verbosity=7)
            elif minimizer == "ipopt":
                ipyopt = pytest.importorskip("ipyopt")
                minimizer = zfit.minimize.Ipyopt(verbosity=7)
            elif minimizer == "lm":
                minimizer = zfit.minimize.LevenbergMarquardt(verbosity=7)
            elif minimizer == "newtoncg":
                minimizer = zfit.minimize.ScipyNewtonCG(verbosity=7)
            elif minimizer == "truncnc":
                minimizer = zfit.minimize.ScipyTruncNC(verbosity=7)
            elif minimizer == "nlopttruncnc":
                _ = pytest.importorskip("nlopt")
                minimizer = zfit.minimize.NLoptTruncNewton(verbosity=7)
            elif minimizer == "shiftvar":
                _ = pytest.importorskip("nlopt")
                minimizer = zfit.minimize.NLoptShiftVar(verbosity=7)

            start = time.time()
            # print(f"Running {hypotest_orig} with {minimizer}, {n_bins} bins, evaluating nll")
            nll.value()
            nll.value()

            # print(f"NLL evaluated, {(start1 := time.time()) - start:.1f}s, running gradient...")
            nll.gradient()
            nll.gradient()
            # print(f"Gradient evaluated, {(start2 := time.time()) - start1:.1f}s, running value_gradient...")
            # nll.value_gradient_hessian()
            # nll.value_gradient_hessian()
            # print(
            #     f"Value_gradient_hessian evaluated, {(start3 := time.time()) - start2:.1f}s, running value_gradient...")
            nll.value_gradient()
            nll.value_gradient()
            # print(f"Value_gradient evaluated, {(start4 := time.time()) - start2:.1f}s, running hessian...")
            nll.hessian()
            nll.hessian()
            # print(f"Hessian evaluated, {(start5 := time.time()) - start4:.1f}s, running minimizer...")
            benchmark(hypotest, minimizer, nll)
            # print(f"Minimizer ran, {time.time() - start5:.1f}s")
    assert True
