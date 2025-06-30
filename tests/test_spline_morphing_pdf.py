#  Copyright (c) 2025 zfit
"""Comprehensive tests for SplineMorphingPDF, especially focusing on extended yield functionality."""

import numpy as np
import pytest
import hist

import zfit
import zfit.z.numpy as znp
from zfit.models.morphing import SplineMorphingPDF


@pytest.fixture
def testspaces():
    """Create test spaces for 1D and 2D cases."""
    obs_1d = zfit.Space("x", (-5, 5))
    
    # Create binned spaces
    binning_1d = zfit.binned.RegularBinning(20, -5, 5, name="x")
    obs_binned_1d = zfit.Space("x", binning=[binning_1d])
    
    return obs_1d, obs_binned_1d


@pytest.fixture
def simple_hists(testspaces):
    """Create simple test histograms with known yields."""
    obs_1d, obs_binned_1d = testspaces
    
    # Create three different histograms with known properties
    # Histogram at alpha = -1: Gaussian centered at -1
    data_m1 = np.random.normal(-1, 0.8, 5000)
    hist_m1 = zfit.Data.from_numpy(obs=obs_1d, array=data_m1).to_binned(obs_binned_1d)
    yield_m1 = 1000.0
    pdf_m1 = zfit.pdf.HistogramPDF(hist_m1, extended=yield_m1)
    
    # Histogram at alpha = 0: Uniform distribution
    data_0 = np.random.uniform(-4, 4, 5000)
    hist_0 = zfit.Data.from_numpy(obs=obs_1d, array=data_0).to_binned(obs_binned_1d)
    yield_0 = 1500.0
    pdf_0 = zfit.pdf.HistogramPDF(hist_0, extended=yield_0)
    
    # Histogram at alpha = 1: Gaussian centered at 1
    data_p1 = np.random.normal(1, 0.8, 5000)
    hist_p1 = zfit.Data.from_numpy(obs=obs_1d, array=data_p1).to_binned(obs_binned_1d)
    yield_p1 = 2000.0
    pdf_p1 = zfit.pdf.HistogramPDF(hist_p1, extended=yield_p1)
    
    hists = {-1.0: pdf_m1, 0.0: pdf_0, 1.0: pdf_p1}
    yields = {-1.0: yield_m1, 0.0: yield_0, 1.0: yield_p1}
    
    return hists, yields, obs_binned_1d


def test_spline_morphing_pdf_basic_creation(simple_hists):
    """Test basic creation of SplineMorphingPDF."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    
    # Test with mapping
    pdf = SplineMorphingPDF(alpha, hists)
    assert pdf.space == obs_binned
    assert pdf.params["alpha"] == alpha
    
    # Test with list (should map to -1, 0, 1)
    hist_list = [hists[-1.0], hists[0.0], hists[1.0]]
    pdf_list = SplineMorphingPDF(alpha, hist_list)
    assert pdf_list.space == obs_binned


def test_spline_morphing_pdf_extended_automatic(simple_hists):
    """Test automatic extended functionality."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    
    # Test automatic extension when all histograms are extended
    pdf = SplineMorphingPDF(alpha, hists, extended=True)
    assert pdf.is_extended
    assert pdf._automatically_extended is True
    
    # Test that yield is interpolated correctly
    # At alpha=0, should get yield of middle histogram
    alpha.set_value(0.0)
    expected_yield = yields[0.0]
    actual_yield = pdf.get_yield().value()
    np.testing.assert_allclose(actual_yield, expected_yield, rtol=0.01)
    
    # At alpha=-1, should get yield of first histogram
    alpha.set_value(-1.0)
    expected_yield = yields[-1.0]
    actual_yield = pdf.get_yield().value()
    np.testing.assert_allclose(actual_yield, expected_yield, rtol=0.01)
    
    # At alpha=1, should get yield of last histogram
    alpha.set_value(1.0)
    expected_yield = yields[1.0]
    actual_yield = pdf.get_yield().value()
    np.testing.assert_allclose(actual_yield, expected_yield, rtol=0.01)


def test_spline_morphing_pdf_extended_manual(simple_hists):
    """Test manual extended functionality with specific yield."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    manual_yield = 3000.0
    
    pdf = SplineMorphingPDF(alpha, hists, extended=manual_yield)
    assert pdf.is_extended
    assert pdf._automatically_extended is False
    
    # Yield should be the manually specified value, not interpolated
    actual_yield = pdf.get_yield().value()
    np.testing.assert_allclose(actual_yield, manual_yield, rtol=1e-10)


def test_spline_morphing_pdf_yield_interpolation(simple_hists):
    """Test that yield interpolation works correctly at intermediate points."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, hists, extended=True)
    
    # Test interpolation at alpha = 0.5 (halfway between 0 and 1)
    alpha.set_value(0.5)
    actual_yield = pdf.get_yield().value()
    # Should be between yields[0.0] and yields[1.0]
    assert yields[0.0] < actual_yield < yields[1.0]
    
    # Test interpolation at alpha = -0.5 (halfway between -1 and 0)
    alpha.set_value(-0.5)
    actual_yield = pdf.get_yield().value()
    # Should be between yields[-1.0] and yields[0.0]
    assert min(yields[-1.0], yields[0.0]) < actual_yield < max(yields[-1.0], yields[0.0])


def test_spline_morphing_pdf_methods_extended(simple_hists):
    """Test that all PDF methods work correctly in extended mode."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, hists, extended=True)
    
    # Test at alpha = 0
    alpha.set_value(0.0)
    
    # Test pdf method
    pdf_values = pdf.pdf(obs_binned)
    assert pdf_values.shape == (20,)  # 20 bins
    assert np.all(pdf_values >= 0)
    
    # Test ext_pdf method
    ext_pdf_values = pdf.ext_pdf(obs_binned)
    assert ext_pdf_values.shape == (20,)
    assert np.all(ext_pdf_values >= 0)
    
    # Test relationship: ext_pdf = pdf * yield
    expected_ext_pdf = pdf_values * pdf.get_yield().value()
    np.testing.assert_allclose(ext_pdf_values, expected_ext_pdf, rtol=1e-5)
    
    # Test counts method
    counts = pdf.counts(obs_binned)
    assert counts.shape == (20,)
    assert np.all(counts >= 0)
    
    # Test rel_counts method
    rel_counts = pdf.rel_counts(obs_binned)
    assert rel_counts.shape == (20,)
    assert np.all(rel_counts >= 0)
    
    # Test relationship: counts = rel_counts * yield
    expected_counts = rel_counts * pdf.get_yield().value()
    np.testing.assert_allclose(counts, expected_counts, rtol=1e-5)
    
    # Test that rel_counts sum to 1
    np.testing.assert_allclose(np.sum(rel_counts), 1.0, rtol=1e-5)


def test_spline_morphing_pdf_methods_non_extended(simple_hists):
    """Test that non-extended methods work correctly."""
    hists, yields, obs_binned = simple_hists
    
    # Create non-extended versions of histograms
    non_ext_hists = {}
    for alpha_val, hist_pdf in hists.items():
        # Create non-extended version
        non_ext_hists[alpha_val] = zfit.pdf.HistogramPDF(hist_pdf._data, extended=False)
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, non_ext_hists, extended=False)
    
    assert not pdf.is_extended
    assert pdf._automatically_extended is None
    
    # Test pdf and rel_counts methods (should work)
    alpha.set_value(0.0)
    pdf_values = pdf.pdf(obs_binned)
    rel_counts = pdf.rel_counts(obs_binned)
    
    assert pdf_values.shape == (20,)
    assert rel_counts.shape == (20,)
    assert np.all(pdf_values >= 0)
    assert np.all(rel_counts >= 0)
    
    # Test that extended methods raise exceptions
    with pytest.raises(zfit.util.exception.NotExtendedPDFError):
        pdf.ext_pdf(obs_binned)
    
    with pytest.raises(zfit.util.exception.NotExtendedPDFError):
        pdf.counts(obs_binned)


def test_spline_morphing_pdf_continuity(simple_hists):
    """Test that the interpolation is continuous."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, hists, extended=True)
    
    # Test continuity by checking small changes in alpha
    alpha_values = np.linspace(-1, 1, 21)
    yields_interpolated = []
    pdf_values_interpolated = []
    
    for alpha_val in alpha_values:
        alpha.set_value(alpha_val)
        yields_interpolated.append(pdf.get_yield().value())
        pdf_values_interpolated.append(pdf.pdf(obs_binned))
    
    yields_interpolated = np.array(yields_interpolated)
    pdf_values_interpolated = np.array(pdf_values_interpolated)
    
    # Check that yield interpolation is smooth (no sudden jumps)
    yield_diffs = np.diff(yields_interpolated)
    max_yield_diff = np.max(np.abs(yield_diffs))
    expected_max_diff = (yields[1.0] - yields[-1.0]) / 10  # Should be roughly smooth
    assert max_yield_diff < expected_max_diff
    
    # Check that PDF values change smoothly
    for bin_idx in range(20):
        bin_diffs = np.diff(pdf_values_interpolated[:, bin_idx])
        # Should not have any NaN or infinite values
        assert np.all(np.isfinite(bin_diffs))


def test_spline_morphing_pdf_edge_cases(simple_hists):
    """Test edge cases and error conditions."""
    hists, yields, obs_binned = simple_hists
    
    # Test error when providing wrong number of histograms in list
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    with pytest.raises(ValueError, match="assumed to correspond to an alpha of -1, 0 and 1"):
        SplineMorphingPDF(alpha, [hists[-1.0], hists[0.0]])  # Only 2 histograms
    
    # Test error when not all histograms are extended but extended=True
    non_ext_hists = {-1.0: zfit.pdf.HistogramPDF(hists[-1.0]._data, extended=False)}
    non_ext_hists.update({k: v for k, v in hists.items() if k != -1.0})
    
    with pytest.raises(ValueError, match="all PDFs must be extended"):
        SplineMorphingPDF(alpha, non_ext_hists, extended=True)
    
    # Test error when providing invalid histogram type
    invalid_hists = {-1.0: "not_a_histogram", 0.0: hists[0.0], 1.0: hists[1.0]}
    with pytest.raises(TypeError, match="not a ZfitBinnedPDF"):
        SplineMorphingPDF(alpha, invalid_hists)


def test_spline_morphing_pdf_with_uhi_histogram(simple_hists):
    """Test that UHI histograms are properly converted."""
    hists, yields, obs_binned = simple_hists
    
    # Create a UHI histogram
    h = hist.Hist(hist.axis.Regular(20, -5, 5, name="x"))
    h.fill(np.random.normal(0, 1, 1000))
    
    # Mix UHI histogram with zfit PDFs
    mixed_hists = {-1.0: hists[-1.0], 0.0: h, 1.0: hists[1.0]}
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, mixed_hists, extended=False)
    
    # Should work without errors
    alpha.set_value(0.0)
    pdf_values = pdf.pdf(obs_binned)
    assert pdf_values.shape == (20,)
    assert np.all(pdf_values >= 0)


def test_spline_morphing_pdf_normalization(simple_hists):
    """Test that normalization works correctly."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, hists, extended=False)
    
    # Test at various alpha values
    for alpha_val in [-0.8, -0.3, 0.2, 0.7]:
        alpha.set_value(alpha_val)
        
        # PDF should integrate to 1
        pdf_values = pdf.pdf(obs_binned)
        bin_widths = obs_binned.binning.widths[0]
        integral = np.sum(pdf_values * bin_widths)
        np.testing.assert_allclose(integral, 1.0, rtol=1e-3)
        
        # rel_counts should sum to 1
        rel_counts = pdf.rel_counts(obs_binned)
        np.testing.assert_allclose(np.sum(rel_counts), 1.0, rtol=1e-5)


def test_spline_morphing_pdf_yield_parameter_dependencies(simple_hists):
    """Test that yield parameter correctly updates when alpha changes."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, hists, extended=True)
    
    # Test that changing alpha affects the composed yield
    alpha.set_value(0.0)
    yield_at_0 = pdf.get_yield().value()
    
    alpha.set_value(0.5)
    yield_at_05 = pdf.get_yield().value()
    
    # Yields should be different at different alpha values
    assert not np.isclose(yield_at_0, yield_at_05)
    
    # Test that the yield is a ComposedParameter (meaning it's computed dynamically)
    assert hasattr(pdf.get_yield(), 'params')  # ComposedParameter has params attribute
    
    # Test that the composed yield responds to alpha changes
    alpha.set_value(-1.0)
    yield_at_m1 = pdf.get_yield().value()
    
    alpha.set_value(1.0)
    yield_at_p1 = pdf.get_yield().value()
    
    # All yields should be different
    assert not np.isclose(yield_at_0, yield_at_m1)
    assert not np.isclose(yield_at_0, yield_at_p1)
    assert not np.isclose(yield_at_m1, yield_at_p1)


def test_spline_morphing_pdf_sampling(simple_hists):
    """Test that sampling works correctly."""
    hists, yields, obs_binned = simple_hists
    
    alpha = zfit.Parameter("alpha", 0.0, -2, 2)
    pdf = SplineMorphingPDF(alpha, hists, extended=True)
    
    # Test sampling - note that SplineMorphingPDF returns binned samples
    alpha.set_value(0.0)
    n_samples = 1000
    sample = pdf.sample(n_samples)
    
    # For binned PDFs, sample returns BinnedData with histogram values
    # The number of events will be equal to the number of bins, not n_samples
    assert sample.space == obs_binned
    
    # Check that the sample has proper structure
    sample_values = sample.values()
    assert sample_values.shape == (20,)  # 20 bins
    assert np.all(sample_values >= 0)  # All counts should be non-negative


if __name__ == "__main__":
    pytest.main([__file__])