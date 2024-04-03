#  Copyright (c) 2024 zfit
import numpy as np
import pytest


@pytest.mark.parametrize("weights", [None, np.random.uniform(size=1000)], ids=['no_weights', 'weights'])
def test_update_data_sampler(weights):
    import zfit
    from zfit.data import SamplerData
    sample = np.random.normal(0, 1, size=(1000, 3))
    sample2 = np.random.normal(0, 1, size=(1000, 3))
    sample3 = np.random.normal(0, 1, size=(1000, 3))

    obs = zfit.Space('obs1', limits=(-100, 100)) * zfit.Space('obs2', limits=(-200, 200)) * zfit.Space('obs3', limits=(-150, 150))
    data = SamplerData.from_sampler(obs=obs, sample_and_weights_func=lambda n, params: (sample, weights), n=1000)
    if weights is not None:
        assert np.allclose(data.weights, weights)
    assert np.allclose(data.value(), sample)
    if weights is not None:
        with pytest.raises(ValueError):
            data.update_data(sample2, weights=None)
    data.update_data(sample2, weights=weights)
    assert np.allclose(data.value(), sample2)
    if weights is not None:
        with pytest.raises(ValueError):
            data.update_data(sample3, weights=None)
    else:
        data.update_data(sample3, weights=weights)
        assert np.allclose(data.value(), sample3)
