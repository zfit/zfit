#  Copyright (c) 2025 zfit

"""Test serialization of KDE with different padding types."""

import numpy as np
import pytest

import zfit
import zfit.serialization as zserial

# Create test data
np.random.seed(42)
test_data = np.random.normal(0, 1, 100)
test_obs = zfit.Space("obs", (-3, 3))


@pytest.mark.parametrize(
    "kde_class",
    [
        zfit.pdf.KDE1DimFFT,
        zfit.pdf.KDE1DimGrid,
        zfit.pdf.KDE1DimISJ,
        zfit.pdf.KDE1DimExact,
    ],
)
@pytest.mark.parametrize(
    "padding",
    [
        0.1,  # float
        0.2,  # another float
        True,  # bool True
        False,  # bool False
        {"lowermirror": 0.1},  # dict with one key
        {"uppermirror": 0.2},  # dict with other key
        {"lowermirror": 0.1, "uppermirror": 0.2},  # dict with both keys
        None,  # None
    ],
)
def test_kde_padding_serialization(kde_class, padding):
    """Test that KDE padding parameter can be serialized and deserialized correctly."""
    # Create KDE with specified padding
    kde = kde_class(data=test_data, obs=test_obs, padding=padding)

    # Serialize to HS3
    hs3_data = zserial.Serializer.to_hs3(kde)

    # Deserialize from HS3
    loaded = zserial.Serializer.from_hs3(hs3_data, reuse_params=kde.get_params())
    kde_restored = list(loaded["distributions"].values())[0]

    # Check that padding was preserved
    original_padding = kde.hs3.original_init.get("padding")
    restored_padding = kde_restored.hs3.original_init.get("padding")

    if isinstance(original_padding, dict) and isinstance(restored_padding, dict):
        # For dict, check keys and values separately
        assert set(original_padding.keys()) == set(restored_padding.keys())
        for key in original_padding.keys():
            assert original_padding[key] == restored_padding[key]
    else:
        assert original_padding == restored_padding


@pytest.mark.parametrize(
    "kde_class",
    [
        zfit.pdf.KDE1DimFFT,
        zfit.pdf.KDE1DimGrid,
        zfit.pdf.KDE1DimISJ,
    ],
)
def test_kde_padding_asdf_serialization(kde_class):
    """Test that KDE padding parameter works with ASDF serialization."""
    try:
        import asdf  # noqa: F401
    except ImportError:
        pytest.skip("ASDF not installed")

    # Test with dict padding (the most complex case)
    padding = {"lowermirror": 0.1, "uppermirror": 0.2}
    kde = kde_class(data=test_data, obs=test_obs, padding=padding)

    # Serialize to ASDF
    asdf_obj = kde.to_asdf()

    # Get the tree
    tree = asdf_obj.tree

    # Deserialize from ASDF
    kde_restored = kde_class.from_asdf(asdf_obj)

    # Check that padding was preserved
    original_padding = kde.hs3.original_init.get("padding")
    restored_padding = kde_restored.hs3.original_init.get("padding")

    if isinstance(original_padding, dict) and isinstance(restored_padding, dict):
        # For dict, check keys and values separately
        assert set(original_padding.keys()) == set(restored_padding.keys())
        for key in original_padding.keys():
            assert original_padding[key] == restored_padding[key]
    else:
        assert original_padding == restored_padding
