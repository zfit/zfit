#  Copyright (c) 2026 zfit
from __future__ import annotations

import numpy as np
import pytest

import zfit


def test_regular_binning_limits_of_size_one():
    # arrays and tensors with a single element like `space.v1.lower` are fine as limits, also with NumPy >= 2.4
    obs = zfit.Space("obs_regular_binning", limits=(-15, 25))
    expected_edges = np.linspace(-15, 25, 93)
    limits = [
        (-15, 25),
        (-15.0, 25.0),
        (np.float64(-15), np.float64(25)),
        (np.array(-15.0), np.array(25.0)),
        (np.array([-15.0]), np.array([25.0])),
        (obs.v1.lower, obs.v1.upper),
    ]
    for start, stop in limits:
        binning = zfit.binned.RegularBinning(92, start, stop, name="obs_regular_binning")
        assert binning.size == 92
        np.testing.assert_allclose(binning.edges, expected_edges)
        assert isinstance(obs.with_binning(binning), zfit.Space)


def test_regular_binning_limits_with_more_than_one_element_fail():
    with pytest.raises(ValueError, match="`start` of a regular binning has to be a single number"):
        zfit.binned.RegularBinning(10, np.array([-15.0, -10.0]), 25, name="obs_regular_binning")
    with pytest.raises(ValueError, match="`stop` of a regular binning has to be a single number"):
        zfit.binned.RegularBinning(10, -15, np.array([25.0, 30.0]), name="obs_regular_binning")
