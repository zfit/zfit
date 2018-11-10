import pytest
import tensorflow as tf
import numpy as np

from zfit import ztf
import zfit.core.basepdf
from zfit.core.limits import Range
from zfit.minimizers.minimizer_minuit import MinuitMinimizer
import zfit.pdfs.dist_tfp
from zfit.pdfs.dist_tfp import Normal
from zfit.pdfs.basic import Gauss
from zfit.core.parameter import FitParameter
import zfit.settings
from zfit.core.loss import unbinned_nll

mu_true = 1.2
sigma_true = 4.1

test_values_np = np.random.normal(loc=mu_true, scale=sigma_true, size=1000)

low, high = -24.3, 28.6
mu1 = FitParameter("mu1", ztf.to_float(mu_true)-0.2, mu_true - 1., mu_true + 1.)
sigma1 = FitParameter("sigma1", ztf.to_float(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)
mu2 = FitParameter("mu2", ztf.to_float(mu_true)-0.2, mu_true - 1., mu_true + 1.)
sigma2 = FitParameter("sigma2", ztf.to_float(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)

# HACK
# Gauss = Normal
# HACK END
mu_constr = Gauss(ztf.to_float(1.6), ztf.to_float(0.2), name="mu_constr")
sigma_constr = Gauss(ztf.to_float(3.8), ztf.to_float(0.2), name="sigma_constr")

gaussian1 = Gauss(mu1, sigma1, name="gaussian1")
gaussian2 = Gauss(mu2, sigma2, name="gaussian2")

init = tf.global_variables_initializer()


def test_unbinned_nll():
    with tf.Session() as sess:
        sess.run(init)
        with mu_constr.temp_norm_range((-np.infty, np.infty)):
            with sigma_constr.temp_norm_range((-np.infty, np.infty)):
                test_values = tf.constant(test_values_np)
                nll = unbinned_nll(probs=gaussian1.prob(x=test_values, norm_range=(-np.infty, np.infty)))
                nll_eval = sess.run(nll)
                minimizer = MinuitMinimizer(loss=nll)
                status = minimizer.minimize(params=[mu1, sigma1], sess=sess)
                params = status.get_parameters()
                # print(params)
                assert params[mu1.name]['value'] == pytest.approx(np.mean(test_values_np), rel=0.0005)
                assert params[sigma1.name]['value'] == pytest.approx(np.std(test_values_np), rel=0.0005)

                # with constraints
                sess.run(init)

                nll = unbinned_nll(probs=gaussian2.prob(x=test_values, norm_range=(-np.infty, np.infty)),
                                   constraints={mu2: mu_constr,
                                                sigma2: sigma_constr}
                                   )

                minimizer = MinuitMinimizer(loss=nll)
                status = minimizer.minimize(params=[mu2, sigma2], sess=sess)
                params = status.get_parameters()

                assert params[mu2.name]['value'] > np.mean(test_values_np)
                assert params[sigma2.name]['value'] < np.std(test_values_np)

                print(status)


def true_gaussian_func(x):
    return np.exp(- (x - mu_true) ** 2 / (2 * sigma_true ** 2))
