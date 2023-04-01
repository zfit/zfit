#  Copyright (c) 2023 zfit
import numpy as np

import zfit


# Set up the simple model for the test
def create_loss(mu, sigma):
    obs = zfit.Space("x", limits=(-10, 10))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    data_np = np.random.normal(mu.value(), sigma.value(), size=5)
    data_np = zfit.Data.from_numpy(obs=obs, array=data_np)
    data = gauss.sample(n=5)
    return zfit.loss.UnbinnedNLL(model=[gauss, gauss], data=[data, data_np])


# Test the reproducibility of the fit
def test_reproducibility():
    # Create model and data

    mu = zfit.Parameter("mu", 0.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)

    # Set up minimizer
    minimizer = zfit.minimize.Minuit()
    params = [mu, sigma]
    vals = [0.2, 0.7]
    # Perform fit with fixed seed
    seed = 4294
    zfit.settings.set_seed(seed)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed1 = minimizer.minimize(loss)

    # Perform fit again with the same fixed seed
    zfit.settings.set_seed(seed)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed2 = minimizer.minimize(loss)

    # Perform fit without a fixed seed (set to None)
    zfit.settings.set_seed(None)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_no_seed = minimizer.minimize(loss)

    zfit.settings.set_seed(seed)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed3 = minimizer.minimize(loss)

    zfit.settings.set_seed(None)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_no_seed2 = minimizer.minimize(loss)

    # Check if the results with fixed seeds are the same
    assert np.allclose(
        result_fixed_seed1.values, result_fixed_seed2.values
    ), "Results with the same fixed seed should be the same"

    # Check if the results without a fixed seed are different
    assert not np.allclose(
        result_fixed_seed1.values, result_no_seed.values
    ), "Results without a fixed seed should be different"

    assert np.allclose(
        result_fixed_seed3.values, result_fixed_seed2.values
    ), "Results with the same fixed seed should be the same"
    assert not np.allclose(
        result_fixed_seed3.values, result_no_seed2.values
    ), "Results without a fixed seed should be different"
    assert not np.allclose(
        result_no_seed.values, result_no_seed2.values
    ), "Results without a fixed seed should be different"

    # check if numpy and tensorflow seeds can be set separately
    zfit.settings.set_seed(seed, numpy=2)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed_npvary = minimizer.minimize(loss)

    assert not np.allclose(
        result_fixed_seed_npvary.values, result_fixed_seed2.values
    ), "Results with the same fixed seed should be the same"
    assert not np.allclose(
        result_fixed_seed_npvary.values, result_no_seed.values
    ), "Results without a fixed seed should be different"

    zfit.settings.set_seed(seed, backend=3)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed_tvary = minimizer.minimize(loss)

    # TODO: that means that the rnd does not depend on the global? or what?
    # assert not np.allclose(result_fixed_seed_tvary.values, result_fixed_seed2.values), "Results with the same fixed seed should be the same"
    assert not np.allclose(
        result_fixed_seed_tvary.values, result_no_seed.values
    ), "Results without a fixed seed should be different"

    zfit.settings.set_seed(seed, numpy=2)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed_npvary2 = minimizer.minimize(loss)

    assert np.allclose(
        result_fixed_seed_npvary2.values, result_fixed_seed_npvary.values
    ), "Results with the same fixed seed should be the same"
    assert not np.allclose(
        result_fixed_seed_npvary2.values, result_no_seed.values
    ), "Results without a fixed seed should be different"

    zfit.settings.set_seed(seed, backend=3)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed_tvary2 = minimizer.minimize(loss)

    assert np.allclose(
        result_fixed_seed_tvary2.values, result_fixed_seed_tvary.values
    ), "Results with the same fixed seed should be the same"
    assert not np.allclose(
        result_fixed_seed_tvary2.values, result_no_seed.values
    ), "Results without a fixed seed should be different"

    zfit.settings.set_seed(seed, numpy=4, backend=5)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed_tvary_np4 = minimizer.minimize(loss)

    assert not np.allclose(
        result_fixed_seed_tvary_np4.values, result_fixed_seed_tvary.values
    ), "Results with the same fixed seed should be the same"
    assert not np.allclose(
        result_fixed_seed_tvary_np4.values, result_no_seed.values
    ), "Results without a fixed seed should be different"

    zfit.settings.set_seed(seed, numpy=4, backend=5)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed_tvary_np4_2 = minimizer.minimize(loss)

    assert np.allclose(
        result_fixed_seed_tvary_np4_2.values, result_fixed_seed_tvary_np4.values
    ), "Results with the same fixed seed should be the same"
