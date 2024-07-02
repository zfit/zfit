#  Copyright (c) 2024 zfit
import numpy as np


def test_dumpload(tmp_path):
    import zfit
    tmpfile = tmp_path / "test_dill_dumpload1.dill"

    def create_and_fit_data(_mean, _sigma, label="", extended=True):
        """This is a dummy function to create some dataset, fit it and return the fitterd pdf and the result of the fit.

        Takes as argument the mean and sigma of the dummy dataset
        """

        obs = zfit.Space("x", -2, 3)
        mu1 = zfit.Parameter("mu1" + label, 1.2, -4, 6)
        sigma1 = zfit.Parameter("sigma1" + label, 1.3, 0.5, 10)
        yield1 = zfit.Parameter("yield1" + label, 1000, 0, 10000)

        gauss1 = zfit.pdf.Gauss(mu=mu1, sigma=sigma1, obs=obs, extended=yield1)

        mu2 = zfit.Parameter("mu2" + label, 1.2, -4, 6, floating=False)
        sigma2 = zfit.Parameter("sigma2" + label, 1.3, 0.5, 10, floating=False)
        yield2 = zfit.Parameter("yield2" + label, 1000, 0, 10000, floating=False)
        gauss2 = zfit.pdf.Gauss(mu=mu2, sigma=sigma2, obs=obs, extended=yield2)

        mu3 = zfit.Parameter("mu3" + label, 1.2, -4, 6, floating=False)
        sigma3 = zfit.Parameter("sigma3" + label, 1.3, 0.5, 10, floating=False)
        yield3 = zfit.Parameter("yield3" + label, 1000, 0, 10000, floating=False)
        gauss3 = zfit.pdf.Gauss(mu=mu3, sigma=sigma3, obs=obs, extended=yield3)

        mu4 = zfit.Parameter("mu4" + label, 1.2, -4, 6, floating=False)
        sigma4 = zfit.Parameter("sigma4" + label, 1.3, 0.5, 10, floating=False)
        yield4 = zfit.Parameter("yield4" + label, 1000, 0, 10000, floating=False)
        gauss4 = zfit.pdf.Gauss(mu=mu4, sigma=sigma4, obs=obs, extended=yield4)

        mu5 = zfit.Parameter("mu5" + label, 1.2, -4, 6, floating=False)
        sigma5 = zfit.Parameter("sigma5" + label, 1.3, 0.5, 10, floating=False)
        yield5 = zfit.Parameter("yield5" + label, 1000, 0, 10000, floating=False)
        gauss5 = zfit.pdf.Gauss(mu=mu5, sigma=sigma5, obs=obs, extended=yield5)

        frac1 = zfit.Parameter("frac1" + label, 0.5, 0, 1)
        frac2 = zfit.Parameter("frac2" + label, 0.5, 0, 1)
        frac3 = zfit.Parameter("frac3" + label, 0.5, 0, 1)

        if extended:
            model1 = zfit.pdf.SumPDF([gauss1, gauss2])
            model2 = zfit.pdf.SumPDF([gauss3, gauss4, gauss5])
            gauss = zfit.pdf.SumPDF([model1, model2])
        else:
            model1 = zfit.pdf.SumPDF([gauss1, gauss2], fracs=frac1)
            model2 = zfit.pdf.SumPDF([gauss3, gauss4, gauss5], fracs=[frac2, frac3])
            gauss = zfit.pdf.SumPDF([model1, model2], fracs=0.9)

        data = np.random.normal(loc=_mean, scale=_sigma, size=1000)

        data = obs.filter(data)

        # create NLL
        if extended:
            nll = zfit.loss.ExtendedUnbinnedNLL(model=gauss, data=data)
        else:
            nll = zfit.loss.UnbinnedNLL(model=gauss1, data=data)

        # create a minimizer
        minimizer = zfit.minimize.Minuit(gradient=False)
        result = minimizer.minimize(nll).update_params()
        for p in nll.get_params(floating=None):
            p.floating = True

        return result, gauss

    for ntry in range(3):
        results_dict = {}
        result1, gauss1 = create_and_fit_data(2.0, 3.0, label="1")

        results_dict["fit_result1"] = result1
        results_dict["model1"] = gauss1

        with open(tmpfile, "wb") as f:
            zfit.dill.dump(results_dict, f, verify=False)

        with open(tmpfile, "rb") as f:
            results_dict_again = zfit.dill.load(f)
        result2, gauss2 = create_and_fit_data(3.0, 2.0, label="2")
        results_dict_again["fit_result2"] = result2
        results_dict_again["model2"] = gauss2

        # this is the test, this should not fail
        out = zfit.dill.dumps(results_dict_again, verify=True)
        loaded = zfit.dill.loads(out)
        assert loaded is not None
        with open(tmpfile, "w+b") as f:
            zfit.dill.dump(results_dict_again, f, verify=True)

        with open(tmpfile, "rb") as f:
            loaded = zfit.dill.load(f)

        assert loaded is not None
