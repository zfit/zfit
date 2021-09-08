import numpy as np
import pytest


@pytest.mark.parametrize('weights', [None, np.array([1., 2., 3., 4., 5., 6., 7., 8.])])
def test_padreflect_data_weights_1dim(weights):
    from zfit.models.kde import padreflect_data_weights_1dim
    testarray = np.array([0.1, 0.5, -2, 5, 6.7, -1.2, 1.2, 7])

    data, w = padreflect_data_weights_1dim(testarray, mode=0.1, weights=weights)
    true_data = np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3])
    np.testing.assert_allclose(true_data, np.sort(data))
    if weights is not None:
        true_w = np.sort(np.array([6, 3, 1., 2., 3., 4., 5., 6., 7., 8., 8, 5]))
        np.testing.assert_allclose(true_w, np.sort(w))

    data, w = padreflect_data_weights_1dim(testarray, mode={'lowermirror': 0.1, 'uppermirror': 0.1}, weights=weights)
    true_data = np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3])
    np.testing.assert_allclose(true_data, np.sort(data))
    if weights is not None:
        true_w = np.sort(np.array([6, 3, 1., 2., 3., 4., 5., 6., 7., 8., 8, 5]))
        np.testing.assert_allclose(true_w, np.sort(w))

    data, w = padreflect_data_weights_1dim(testarray, mode={'lowermirror': 0.1, 'uppermirror': 0.3}, weights=weights)
    true_data = np.sort(np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3, 9]))
    np.testing.assert_allclose(true_data, np.sort(data))
    if weights is not None:
        true_w = np.sort(np.array([6, 3, 1., 2., 3., 4., 5., 6., 7., 8., 8, 5, 4]))
        np.testing.assert_allclose(true_w, np.sort(w))

    data, w = padreflect_data_weights_1dim(testarray, mode={'lowermirror': 0.1}, weights=weights)
    true_data = np.sort(np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7]))
    np.testing.assert_allclose(true_data, np.sort(data))
    if weights is not None:
        true_w = np.sort(np.array([6, 3, 1., 2., 3., 4., 5., 6., 7., 8., ]))
        np.testing.assert_allclose(true_w, np.sort(w))

    data, w = padreflect_data_weights_1dim(testarray, mode={'lowermirror': 0.3}, weights=weights)
    true_data = np.sort(np.array([-4.1, -4.5, -2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7]))
    np.testing.assert_allclose(true_data, np.sort(data))
    if weights is not None:
        true_w = np.sort(np.array([1, 2, 6, 3, 1., 2., 3., 4., 5., 6., 7., 8., ]))
        np.testing.assert_allclose(true_w, np.sort(w))

    data, w = padreflect_data_weights_1dim(testarray, mode={'uppermirror': 0.3}, weights=weights)
    true_data = np.sort(np.array([-2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3, 9]))
    np.testing.assert_allclose(true_data, np.sort(data))
    if weights is not None:
        true_w = np.sort(np.array([1., 2., 3., 4., 5., 6., 7., 8., 8, 5, 4]))
        np.testing.assert_allclose(true_w, np.sort(w))

    with pytest.raises(ValueError):
        data, w = padreflect_data_weights_1dim(testarray, mode={'uper': 0.3}, weights=weights)
    with pytest.raises(ValueError):
        data, w = padreflect_data_weights_1dim(testarray, mode={'uppermirror': 0.3, 'lowe': 0.321}, weights=weights)
