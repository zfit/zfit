#  Copyright (c) 2023 zfit

import numpy as np
import pytest


@pytest.mark.parametrize(
    "bandwidth", [None, 0.1, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])]
)
@pytest.mark.parametrize(
    "weights", [None, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])]
)
def test_padreflect_data_weights_1dim(weights, bandwidth):
    from zfit import Space
    from zfit.models.kde import padreflect_data_weights_1dim

    testarray = np.array([0.1, 0.5, -2, 5, 6.7, -1.2, 1.2, 7])
    orig_bw = bandwidth

    data, w, bw = padreflect_data_weights_1dim(
        testarray, mode=0.1, weights=weights, bandwidth=bandwidth
    )
    true_data = np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3])

    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([6, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8, 5]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))
    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray,
        mode={"lowermirror": 0.1, "uppermirror": 0.1},
        weights=weights,
        bandwidth=bandwidth,
    )
    true_data = np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3])
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([6, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8, 5]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))
    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray,
        mode={"lowermirror": 0.1, "uppermirror": 0.3},
        weights=weights,
        bandwidth=bandwidth,
    )
    true_data = np.sort(
        np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3, 9])
    )
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([6, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8, 5, 4]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))

    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray,
        mode={"lowermirror": 0.1},
        weights=weights,
        bandwidth=bandwidth,
    )
    true_data = np.sort(np.array([-2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7]))
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(
        np.array(
            [
                6,
                3,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
    )
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))

    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray,
        mode={"lowermirror": 0.3},
        weights=weights,
        bandwidth=bandwidth,
    )
    true_data = np.sort(
        np.array([-4.1, -4.5, -2.8, -2, -2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7])
    )
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(
        np.array(
            [
                1,
                2,
                6,
                3,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
    )
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))
    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray,
        mode={"uppermirror": 0.3},
        weights=weights,
        bandwidth=bandwidth,
    )
    true_data = np.sort(np.array([-2, -1.2, 0.1, 0.5, 1.2, 5, 6.7, 7, 7, 7.3, 9]))
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8, 5, 4]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))
    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    testobs = Space("testSpace", limits=(-2, 7.5))
    data, w, bw = padreflect_data_weights_1dim(
        testarray, mode=0.1, limits=testobs.limits, weights=weights, bandwidth=bandwidth
    )
    true_data = np.sort(
        np.array([-2.8, -2, 0.1, 0.5, -2, 5, 6.7, -1.2, 1.2, 7, 8.0, 8.3])
    )
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([3, 6, 1, 2, 3, 4, 5, 6, 7, 8, 5, 8]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))
    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray, mode=0.3, limits=testobs.limits, weights=weights, bandwidth=bandwidth
    )
    true_data = np.sort(
        np.array(
            [-4.5, -4.1, -2.8, -2, 0.1, 0.5, -2, 5, 6.7, -1.2, 1.2, 7, 8.0, 8.3, 10]
        )
    )
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([1, 2, 3, 6, 1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 8]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))

    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray,
        mode={"lowermirror": 0.3},
        limits=testobs.limits,
        weights=weights,
        bandwidth=bandwidth,
    )
    true_data = np.sort(
        np.array([-4.5, -4.1, -2.8, -2, 0.1, 0.5, -2, 5, 6.7, -1.2, 1.2, 7])
    )
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([1, 2, 3, 6, 1, 2, 3, 4, 5, 6, 7, 8]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))
    if bandwidth is not None:
        if isinstance(orig_bw, np.ndarray):
            true_bw = true_w.copy()
            np.testing.assert_allclose(true_bw, np.sort(bw))
        else:
            assert bw == orig_bw

    data, w, bw = padreflect_data_weights_1dim(
        testarray,
        mode={"uppermirror": 0.3},
        limits=testobs.limits,
        weights=weights,
        bandwidth=bandwidth,
    )
    true_data = np.sort(np.array([0.1, 0.5, -2, 5, 6.7, -1.2, 1.2, 7, 8.0, 8.3, 10]))
    np.testing.assert_allclose(true_data, np.sort(data))
    true_w = np.sort(np.array([1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 8]))
    if weights is not None:
        np.testing.assert_allclose(true_w, np.sort(w))

    with pytest.raises(ValueError):
        data, w, bw = padreflect_data_weights_1dim(
            testarray,
            mode={"uper": 0.3},
            weights=weights,
            bandwidth=bandwidth,
        )
    with pytest.raises(ValueError):
        data, w, bw = padreflect_data_weights_1dim(
            testarray,
            mode={"uppermirror": 0.3, "lowe": 0.321},
            weights=weights,
            bandwidth=bandwidth,
        )
