import pytest
import tensorflow as tf
import numpy as np

import zfit.core.basepdf
from zfit.core.limits import Range
from zfit.minimizers.minimizer_minuit import MinuitMinimizer
import zfit.pdfs.dist_tfp
from zfit.pdfs.basic import Gauss
from zfit.core.parameter import FitParameter
import zfit.settings
from zfit.core.loss import unbinned_nll

mu_true = 1.2
sigma_true = 4.5

test_values = np.random.normal(loc=mu_true, scale=sigma_true, size=10000)

low, high = -14.3, 18.6
mu = FitParameter("mu", mu_true, mu_true - 2., mu_true + 7.)
sigma = FitParameter("sigma", sigma_true, sigma_true - 10., sigma_true + 5.)

mu_constr = Gauss(mu=11., sigma=0.3, name="mu_constr")
sigma_constr = Gauss(mu=4.2, sigma=0.5, name="sigma_constr")

gaussian = Gauss(mu=4.2, sigma=0.5, name="gaussian")

init = tf.global_variables_initializer()


def test_unbinned_nll():
    with tf.Session() as sess:
        sess.run(init)
        with mu_constr.temp_norm_range((-np.infty, np.infty)):
            with sigma_constr.temp_norm_range((-np.infty, np.infty)):
                nll = unbinned_nll(probs=gaussian.prob(x=test_values, norm_range=(low, high)),
                                   constraints={mu: mu_constr, sigma: sigma_constr})
                nll_eval = sess.run(nll)
                minimizer = MinuitMinimizer(loss=nll_eval)


def true_gaussian_func(x):
    return np.exp(- (x - mu_true) ** 2 / (2 * sigma_true ** 2))
