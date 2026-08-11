#  Copyright (c) 2025 zfit
"""Tests for the zfit-native binning classes in ``zfit._variables.axis``.

``RegularBinning``/``VariableBinning``/``Binnings`` no longer inherit from
``hist.axis.Regular``/``Variable``/``NamedAxesTuple``; these tests cover the
duck-typed behavior they still have to provide: the UHI ``PlottableAxis``
protocol (used e.g. by ``mplhep``), ``hist``/``boost_histogram`` interop, and
the container-level broadcasting semantics of ``Binnings``.
"""

import boost_histogram as bh
import hist
import numpy as np
import pytest

from zfit._variables.axis import (
    Binnings,
    RegularBinning,
    VariableBinning,
    binning_to_histaxes,
    new_from_axis,
)


def test_regular_binning_edges_and_size():
    axis = RegularBinning(5, 0, 10, name="x")
    np.testing.assert_allclose(axis.edges, [0, 2, 4, 6, 8, 10])
    assert axis.size == 5
    np.testing.assert_allclose(axis.widths, [2, 2, 2, 2, 2])
    np.testing.assert_allclose(axis.centers, [1, 3, 5, 7, 9])


def test_variable_binning_edges_and_size():
    axis = VariableBinning([0, 1, 3, 4, 10], name="y")
    assert axis.size == 4
    np.testing.assert_allclose(axis.widths, [1, 2, 1, 6])
    np.testing.assert_allclose(axis.centers, [0.5, 2, 3.5, 7])


def test_name_required():
    with pytest.raises(TypeError):
        RegularBinning(5, 0, 10)
    with pytest.raises(ValueError):
        RegularBinning(5, 0, 10, name="")


@pytest.mark.parametrize("binning_cls,args", [(RegularBinning, (5, 0, 10)), (VariableBinning, ([0, 1, 5],))])
def test_equality_and_hash(binning_cls, args):
    a = binning_cls(*args, name="x")
    b = binning_cls(*args, name="x")
    other_name = binning_cls(*args, name="y")
    assert a == b
    assert hash(a) == hash(b)
    assert a != other_name
    assert a != 42  # comparison against an unrelated type must not raise


def test_uhi_plottable_axis_protocol():
    """`len(axis)`/`axis[i]` are required by external plotting libs (e.g. mplhep) that duck-type
    against hist.axis.Regular's sequence behavior; this is what actually broke first when the
    hist.axis inheritance was dropped."""
    axis = RegularBinning(5, 0, 10, name="x")
    assert len(axis) == axis.size == 5
    assert axis[0] == (0, 2)
    assert axis[4] == (8, 10)
    with pytest.raises(IndexError):
        _ = axis[5]


def test_new_from_axis_passthrough_and_conversion():
    native = RegularBinning(5, 0, 10, name="x")
    assert new_from_axis(native) is native

    raw = hist.axis.Regular(5, 0, 10, name="x")
    converted = new_from_axis(raw)
    assert isinstance(converted, RegularBinning)
    np.testing.assert_allclose(converted.edges, raw.edges)

    raw_var = hist.axis.Variable([0, 1, 3, 10], name="y")
    converted_var = new_from_axis(raw_var)
    assert isinstance(converted_var, VariableBinning)
    np.testing.assert_allclose(converted_var.edges, raw_var.edges)

    with pytest.raises(ValueError, match="Transformed axes are not supported"):
        new_from_axis(hist.axis.Regular(5, 1, 10, name="x", transform=hist.axis.transform.log))


def test_binning_to_histaxes_builds_real_hist_axes():
    x = RegularBinning(5, 0, 10, name="x")
    y = VariableBinning([0, 1, 3, 10], name="y")
    histaxes = binning_to_histaxes(Binnings([x, y]))
    assert isinstance(histaxes[0], hist.axis.Regular)
    assert isinstance(histaxes[1], hist.axis.Variable)

    # this is the exact usage in BinnedData.to_hist/_to_boost_histogram_: the converted axes must
    # be genuine hist/boost_histogram axis objects, not just duck-typed lookalikes.
    h = hist.Hist(*histaxes, storage=hist.storage.Weight())
    assert h.axes[0].size == 5
    bhist = bh.Histogram(*histaxes, storage=bh.storage.Weight())
    assert bhist.axes[1].size == 3


def test_binnings_name_indexing():
    x = RegularBinning(5, 0, 10, name="x")
    y = VariableBinning([0, 1, 3, 10], name="y")
    binnings = Binnings([x, y])
    assert binnings["x"] is x
    assert binnings["y"] is y
    assert binnings.name == ("x", "y")
    assert binnings.size == (5, 3)
    with pytest.raises(KeyError):
        _ = binnings["z"]


def test_binnings_container_broadcasting():
    """Binnings.edges/centers/widths must replicate hist's sparse-meshgrid broadcasting so that
    downstream code like `np.prod(binning.widths, axis=0)` (used to compute bin areas) keeps
    working across multiple axes of different sizes."""
    x = RegularBinning(5, 0, 10, name="x")
    y = RegularBinning(2, 0, 4, name="y")
    binnings = Binnings([x, y])

    widths = binnings.widths
    assert widths[0].shape == (5, 1)
    assert widths[1].shape == (1, 2)

    areas = np.prod(widths, axis=0)
    assert areas.shape == (5, 2)
    np.testing.assert_allclose(areas, 2 * 2)  # bin width 2 along x, bin width 2 along y

    edges = binnings.edges
    assert edges[0].shape == (6, 1)
    assert edges[1].shape == (1, 3)
