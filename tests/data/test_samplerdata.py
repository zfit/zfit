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

    obs = zfit.Space('obs1', limits=(-1, 1)) * zfit.Space('obs2', limits=(-2, -1)) * zfit.Space('obs3', limits=(1.5, 2))
    data = SamplerData(obs=obs, data=lambda n: sample, weights=weights)
    if weights is not None:
        assert np.allclose(data.weights, weights)
    assert np.allclose(data.value(), sample)
    data.update_data(sample2)
    assert np.allclose(data.value(), sample2)
    if weights is not None:
        with pytest.raises(ValueError):
            data.update_data(sample3, weights=None)
    else:
        data.update_data(sample3, weights=weights)
        assert np.allclose(data.value(), sample3)
