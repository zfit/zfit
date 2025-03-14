#  Copyright (c) 2025 zfit
import numpy as np
import pytest


@pytest.mark.parametrize("weights", [None, np.random.uniform(size=1000)], ids=['no_weights', 'weights'])
def test_update_data_sampler(weights):
    import zfit
    from zfit.data import SamplerData
    sample1 = np.random.uniform(-10, 30, size=(1000, 3))
    sample2 = np.random.uniform(-10, 30, size=(1000, 3))
    sample3 = np.random.uniform(-10, 30, size=(1000, 3))
    obs = zfit.Space('obs1', limits=(-100, 100)) * zfit.Space('obs2', limits=(-200, 200)) * zfit.Space('obs3', limits=(-150, 150))
    data1 = zfit.Data(sample1, obs=obs, weights=weights)

    data = SamplerData.from_sampler(obs=obs, sample_and_weights_func=lambda n, params: (sample1, weights), n=1000)
    if weights is not None:
        assert np.allclose(data.weights, weights)
    assert np.allclose(data.value(), sample1)
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

    data.update_data(data1)
    assert np.allclose(data.value(), sample1)
    if weights is None:
        assert data.weights is None
        samplesize = float(data.num_entries)
    else:
        assert np.allclose(data.weights, weights)
        samplesize = np.sum(weights)

    assert pytest.approx(data.samplesize) == samplesize
    assert data.num_entries == data.value().shape[0]
