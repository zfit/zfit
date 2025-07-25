#  Copyright (c) 2025 zfit
import numpy as np
import pytest

import zfit


@pytest.mark.parametrize("variances", [None, lambda size: np.random.uniform(size=size) * 10000], ids=['no_variances', 'variances'])
def test_binnedsampler_update_data(variances):

    bins1 = 51
    bins2 = 55
    bins3 = 64
    size = (bins1, bins2, bins3)
    if variances is not None:
        variances = variances(size)
    space1 = zfit.Space('obs1', limits=(-100, 100), binning=bins1)
    space2 = zfit.Space('obs2', limits=(-200, 100), binning=bins2)
    space3 = zfit.Space('obs3', limits=(-150, 350), binning=bins3)
    obs = space1 * space2 * space3
    sample1 = np.random.uniform(100, 10000, size=size)
    sample2 = np.random.uniform(100, 10000, size=size)
    sample3 = np.random.uniform(100, 10000, size=size)
    hist1 = zfit.data.BinnedData.from_tensor(space=obs, values=sample1, variances=variances)

    sampler = zfit.data.BinnedSamplerData.from_sampler(obs=obs, sample_and_variances_func=lambda n, params: (sample1, variances), n=10000)
    if variances is not None:
        assert np.allclose(sampler.variances(), variances)
    assert np.allclose(sampler.values(), sample1)
    if variances is not None:
        with pytest.raises(ValueError):
            sampler.update_data(sample2, variances=None)
    sampler.update_data(sample2, variances=variances)
    assert np.allclose(sampler.values(), sample2)
    if variances is not None:
        with pytest.raises(ValueError):
            sampler.update_data(sample3, variances=None)
    else:
        sampler.update_data(sample3, variances=variances)
        assert np.allclose(sampler.values(), sample3)

    # Check that the binning is preserved
    assert sampler.space == obs

    binneddata = sampler.with_obs(obs=space2 * space1 * space3)
    assert binneddata.space == space2 * space1 * space3

    sample_new = np.moveaxis(sample3, [0, 1, 2], [1, 0, 2])
    np.testing.assert_allclose(binneddata.values(), sample_new)
    if variances is not None:
        variances_new = np.moveaxis(variances, [0, 1, 2], [1, 0, 2])
        np.testing.assert_allclose(binneddata.variances(), variances_new)
    else:
        assert binneddata.variances() is None

    sampler.update_data(hist1)
    assert np.allclose(sampler.values(), sample1)
    if variances is not None:
        assert np.allclose(sampler.variances(), variances)
    else:
        assert sampler.variances() is None

@pytest.mark.parametrize("variances", [None, lambda size: np.random.uniform(size=size) * 10000], ids=['no_variances', 'variances'])
def test_binnedsamplerdata_from_hist(variances):

    bins1 = 51
    bins2 = 55
    bins3 = 64
    size = (bins1, bins2, bins3)
    if variances is not None:
        variances = variances(size)
    space1 = zfit.Space('obs1', limits=(-100, 100), binning=bins1)
    space2 = zfit.Space('obs2', limits=(-200, 100), binning=bins2)
    space3 = zfit.Space('obs3', limits=(-150, 350), binning=bins3)
    obs = space1 * space2 * space3
    sample1 = np.random.uniform(100, 10000, size=size)
    sample2 = np.random.uniform(100, 10000, size=size)
    sample3 = np.random.uniform(100, 10000, size=size)
    hist1 = zfit.data.BinnedData.from_tensor(space=obs, values=sample1, variances=variances)
    hist2 = zfit.data.BinnedData.from_tensor(space=obs, values=sample2, variances=variances)

    data1 = zfit.data.BinnedSamplerData(hist1)
    assert np.allclose(data1.values(), hist1.values())
    if variances is not None:
        assert np.allclose(data1.variances(), hist1.variances())
    else:
        assert data1.variances() is None

    data1.update_data(hist2)
    np.testing.assert_allclose(data1.values(), hist2.values())

    data1 = zfit.data.BinnedSamplerData(hist1.to_hist())
    assert np.allclose(data1.values(), hist1.values())
    if variances is not None:
        np.testing.assert_allclose(data1.variances(), hist1.variances())
    else:
        assert data1.variances() is None

    data1.update_data(hist2)
    np.testing.assert_allclose(data1.values(), hist2.values())
    data1.update_data(hist2.to_hist())
    np.testing.assert_allclose(data1.values(), hist2.values())
