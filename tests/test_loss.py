import pytest
import tensorflow as tf
import numpy as np

from zfit import ztf
import zfit.core.basepdf
from zfit.core.limits import Space
from zfit.minimizers.minimizer_minuit import MinuitMinimizer
import zfit.models.dist_tfp
from zfit.models.dist_tfp import Gauss
from zfit.core.parameter import Parameter
import zfit.settings
from zfit.core.loss import _unbinned_nll_tf, UnbinnedNLL

mu_true = 1.2
sigma_true = 4.1
mu_true2 = 1.01
sigma_true2 = 3.5

yield_true = 1000
test_values_np = np.random.normal(loc=mu_true, scale=sigma_true, size=(yield_true, 1))
test_values_np2 = np.random.normal(loc=mu_true2, scale=sigma_true2, size=yield_true)

low, high = -24.3, 28.6
mu1 = Parameter("mu1", ztf.to_real(mu_true) - 0.2, mu_true - 1., mu_true + 1.)
sigma1 = Parameter("sigma1", ztf.to_real(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)
mu2 = Parameter("mu25", ztf.to_real(mu_true) - 0.2, mu_true - 1., mu_true + 1.)
sigma2 = Parameter("sigma25", ztf.to_real(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)
mu3 = Parameter("mu35", ztf.to_real(mu_true) - 0.2, mu_true - 1., mu_true + 1.)
sigma3 = Parameter("sigma35", ztf.to_real(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)
yield3 = Parameter("yield35", yield_true + 300, 0, yield_true + 20000)

obs1 = 'obs1'

mu_constr = [1.6, 0.2]  # mu, sigma
sigma_constr = [3.8, 0.2]

gaussian1 = Gauss(mu1, sigma1, obs=obs1, name="gaussian1")
gaussian2 = Gauss(mu2, sigma2, obs=obs1, name="gaussian2")
gaussian3 = Gauss(mu3, sigma3, obs=obs1, name="gaussian3")
gaussian3 = gaussian3.create_extended(yield3)


def test_extended_unbinned_nll():
    test_values = ztf.constant(test_values_np)
    test_values = zfit.data.Data.from_tensors(obs=obs1, tensor=test_values)
    nll_object = zfit.loss.ExtendedUnbinnedNLL(model=gaussian3,
                                               data=test_values,
                                               fit_range=(-20, 20))
    minimizer = MinuitMinimizer()
    status = minimizer.minimize(loss=nll_object, params=[mu3, sigma3, yield3])
    params = status.params
    assert params[mu3]['value'] == pytest.approx(np.mean(test_values_np), rel=0.005)
    assert params[sigma3]['value'] == pytest.approx(np.std(test_values_np), rel=0.005)
    assert params[yield3]['value'] == pytest.approx(yield_true, rel=0.005)


def test_unbinned_simultaneous_nll():
    test_values = tf.constant(test_values_np)
    test_values = zfit.data.Data.from_tensors(obs=obs1, tensor=test_values)
    test_values2 = tf.constant(test_values_np2)
    test_values2 = zfit.data.Data.from_tensors(obs=obs1, tensor=test_values2)

    nll_object = zfit.loss.UnbinnedNLL(model=[gaussian1, gaussian2],
                                       data=[test_values, test_values2],
                                       fit_range=[(-np.infty, np.infty), (-np.infty, np.infty)]
                                       )
    # nll_eval = zfit.run(nll)
    minimizer = MinuitMinimizer()
    status = minimizer.minimize(loss=nll_object, params=[mu1, sigma1, mu2, sigma2])
    params = status.params
    assert params[mu1]['value'] == pytest.approx(np.mean(test_values_np), rel=0.005)
    assert params[mu2]['value'] == pytest.approx(np.mean(test_values_np2), rel=0.005)
    assert params[sigma1]['value'] == pytest.approx(np.std(test_values_np), rel=0.005)
    assert params[sigma2]['value'] == pytest.approx(np.std(test_values_np2), rel=0.005)


def test_unbinned_nll():
    test_values = tf.constant(test_values_np)
    test_values = zfit.data.Data.from_tensors(obs=obs1, tensor=test_values)
    nll_object = zfit.loss.UnbinnedNLL(model=gaussian1, data=test_values, fit_range=(-np.infty, np.infty))
    minimizer = MinuitMinimizer()
    status = minimizer.minimize(loss=nll_object, params=[mu1, sigma1])
    params = status.params
    assert params[mu1]['value'] == pytest.approx(np.mean(test_values_np), rel=0.005)
    assert params[sigma1]['value'] == pytest.approx(np.std(test_values_np), rel=0.005)

    constraints = zfit.constraint.nll_gaussian(params=[mu2, sigma2],
                                               mu=[mu_constr[0], sigma_constr[0]],
                                               sigma=[mu_constr[1], sigma_constr[1]])
    nll_object = UnbinnedNLL(model=gaussian2, data=test_values, fit_range=(-np.infty, np.infty),
                             constraints=constraints)

    minimizer = MinuitMinimizer()
    status = minimizer.minimize(loss=nll_object, params=[mu2, sigma2])
    params = status.params

    assert params[mu2]['value'] > np.mean(test_values_np)
    assert params[sigma2]['value'] < np.std(test_values_np)


def test_add():
    param1 = Parameter("param1", 1.)
    param2 = Parameter("param2", 2.)

    pdfs = [0] * 4
    pdfs[0] = Gauss(param1, 4, obs=obs1)
    pdfs[1] = Gauss(param2, 5, obs=obs1)
    pdfs[2] = Gauss(3, 6, obs=obs1)
    pdfs[3] = Gauss(4, 7, obs=obs1)

    datas = [0] * 4
    datas[0] = ztf.constant(1.)
    datas[1] = ztf.constant(2.)
    datas[2] = ztf.constant(3.)
    datas[3] = ztf.constant(4.)

    ranges = [0] * 4
    ranges[0] = (1, 4)
    ranges[1] = Space(limits=(2, 5), obs=obs1)
    ranges[2] = Space(limits=(3, 6), obs=obs1)
    ranges[3] = Space(limits=(4, 7), obs=obs1)

    constraint1 = zfit.constraint.nll_gaussian(params=param1, mu=1, sigma=0.5)
    constraint2 = zfit.constraint.nll_gaussian(params=param1, mu=2, sigma=0.25)
    merged_contraints = [constraint1, constraint2]

    nll1 = UnbinnedNLL(model=pdfs[0], data=datas[0], fit_range=ranges[0], constraints=constraint1)
    nll2 = UnbinnedNLL(model=pdfs[1], data=datas[1], fit_range=ranges[1], constraints=constraint2)
    nll3 = UnbinnedNLL(model=[pdfs[2], pdfs[3]], data=[datas[2], datas[3]], fit_range=[ranges[2], ranges[3]])

    simult_nll = nll1 + nll2 + nll3

    assert simult_nll.model == pdfs
    assert simult_nll.data == datas

    ranges[0] = Space._from_any(limits=ranges[0], obs=obs1,
                                axes=(0,))  # for comparison, Space can only compare with Space
    ranges[1]._axes = (0,)
    ranges[2]._axes = (0,)
    ranges[3]._axes = (0,)

    assert simult_nll.fit_range == ranges

    assert simult_nll.constraints == merged_contraints


def test_gradients():
    param1 = Parameter("param111", 1.)
    param2 = Parameter("param222", 2.)

    gauss1 = Gauss(param1, 4, obs=obs1)
    gauss1.set_norm_range((-5, 5))
    gauss2 = Gauss(param2, 5, obs=obs1)
    gauss2.set_norm_range((-5, 5))

    data1 = zfit.data.Data.from_tensors(obs=obs1, tensor=ztf.constant(1., shape=(100,)))
    data1.set_data_range((-5, 5))
    data2 = zfit.data.Data.from_tensors(obs=obs1, tensor=ztf.constant(1., shape=(100,)))
    data2.set_data_range((-5, 5))

    nll = UnbinnedNLL(model=[gauss1, gauss2], data=[data1, data2])

    gradient1 = nll.gradients(params=param1)
    assert zfit.run(gradient1) == zfit.run(tf.gradients(nll.value(), param1))
    gradient2 = nll.gradients(params=[param2, param1])
    both_gradients_true = zfit.run(tf.gradients(nll.value(), [param2, param1]))
    assert zfit.run(gradient2) == both_gradients_true
    gradient3 = nll.gradients()
    assert frozenset(zfit.run(gradient3)) == frozenset(both_gradients_true)
