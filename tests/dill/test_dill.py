#  Copyright (c) 2024 zfit
import numpy as np


def test_dumpload(tmp_path):
    import zfit
    tmpfile = tmp_path / "test_dill_dumpload1.dill"




    def create_and_fit_data(_mean, _sigma, label=""):
        """This is a dummy function to create some dataset, fit it and return the fitterd pdf and the result of the fit.

        Takes as argument the mean and sigma of the dummy dataset
        """

        obs = zfit.Space("x", -2, 3)
        mu = zfit.Parameter("mu" + label, 1.2, -4, 6)
        sigma = zfit.Parameter("sigma" + label, 1.3, 0.5, 10)

        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

        data = np.random.normal(loc=_mean, scale=_sigma, size=1000)

        data = obs.filter(data)

        # create NLL
        nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

        # create a minimizer
        minimizer = zfit.minimize.Minuit(gradient=False)
        result = minimizer.minimize(nll).update_params()

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
