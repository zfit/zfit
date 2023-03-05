#  Copyright (c) 2023 zfit
import json

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
            list(shapesys.values()), [1, 1], unc
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
    binning = [n_bins, -0.5, n_bins + 0.5]
    data = [120.0] * n_bins
    bkg = [100.0] * n_bins
    bkgerr = [10.0] * n_bins
    sig = [30.0] * n_bins

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
    with zfit.param.set_values(nll.get_params(), np.array(nll.get_params())):
        _ = minimizer.minimize(nll)


bins = [
    # 1,
    3,
    # 10,
    # 50,
    # 100,
    # 200,
    # 400,
    # 1000,
]
bin_ids = [f"{n_bins}_bins" for n_bins in bins]


@pytest.mark.benchmark(
    # group="group-name",
    # min_time=0.1,
    max_time=20,
    min_rounds=1,
    # timer=time.time,
    disable_gc=True,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("n_bins", bins, ids=bin_ids)
@pytest.mark.parametrize(
    "hypotest",
    ["pyhf", "zfit"],
)
@pytest.mark.parametrize(
    "eager", [False, True], ids=lambda x: "eager" if x else "graph"
)
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

    if "pyhf" in hypotest:
        hypotest = hypotest_pyhf
        if eager:
            pyhf.set_backend("numpy")
        else:
            try:
                import jax
            except ImportError:
                return
            else:
                pyhf.set_backend("jax")

        pdf = uncorrelated_background(signp, bkgnp, uncnp)
        data = datanp + pdf.config.auxdata

        # warmup
        pyhf.infer.mle.fit(
            data, pdf, pdf.config.suggested_init(), pdf.config.suggested_bounds()
        )
        benchmark(hypotest, pdf, data)
    elif hypotest == "zfit":
        with zfit.run.set_graph_mode(not eager):
            hypotest = hypotest_zfit
            obs = zfit.Space(
                "signal",
                binning=zfit.binned.RegularBinning(
                    n_bins, -0.5, n_bins + 0.5, name="signal"
                ),
            )
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
                list(shapesys.values()), np.ones_like(unc).tolist(), unc
            )
            nll = zfit.loss.ExtendedBinnedNLL(
                zmodel,
                zdata,
                constraints=constraint,
                # options={"numhess": False}
            )

            # minimizer = zfit.minimize.Minuit(tol=1e-3, gradient=True, mode=0, verbosity=8)
            # minimizer = zfit.minimize.NLoptMMAV1(tol=1e-3, verbosity=8)
            # minimizer = zfit.minimize.NLoptLBFGSV1()
            # minimizer = zfit.minimize.ScipyLBFGSBV1()
            minimizer = zfit.minimize.ScipyTrustConstrV1()
            # minimizer = zfit.minimize.IpyoptV1()

            nll.value()
            nll.value()
            nll.gradient()
            nll.gradient()
            nll.value_gradient_hessian(params=nll.get_params())
            benchmark(hypotest, minimizer, nll)
    assert True
