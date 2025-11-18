#  Copyright (c) 2025 zfit
"""Test binned PDFs with unbinned data inputs for pdf, ext_pdf, counts, and rel_counts methods."""

import numpy as np
import pytest
import hist

import zfit
import zfit.z.numpy as znp
from zfit.models.tobinned import BinnedFromUnbinnedPDF


@pytest.fixture
def gauss_1d_pdfs():
    """Create 1D test PDFs for testing."""
    # Create unbinned Gaussian
    mu = zfit.Parameter("mu_1d", 2.0)
    sigma = zfit.Parameter("sigma_1d", 1.5)
    obs = zfit.Space("x", (-5, 10))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create binned version
    n_yield = 10000
    gauss = gauss.create_extended(n_yield)
    nbins = 50
    axis = zfit.binned.RegularBinning(nbins, -5, 10, name="x")
    obs_binned = zfit.Space("x", binning=[axis])
    gauss_binned = BinnedFromUnbinnedPDF(pdf=gauss, space=obs_binned)

    # Create histogram PDF for comparison
    data_binned = gauss_binned.to_binneddata()
    hist_pdf = zfit.pdf.HistogramPDF(data=data_binned, extended=True)

    return gauss, gauss_binned, hist_pdf, obs, obs_binned


@pytest.fixture
def gauss_2d_pdfs():
    """Create 2D test PDFs for testing."""
    # Create unbinned 2D Gaussian
    mux = zfit.Parameter("mux_2d", 1.0)
    sigmax = zfit.Parameter("sigmax_2d", 1.2)
    muy = zfit.Parameter("muy_2d", -2.0)
    sigmay = zfit.Parameter("sigmay_2d", 2.0)

    obsx = zfit.Space("x", (-5, 10))
    obsy = zfit.Space("y", (-10, 10))
    obs2d = obsx * obsy

    gaussx = zfit.pdf.Gauss(mu=mux, sigma=sigmax, obs=obsx)
    gaussy = zfit.pdf.Gauss(mu=muy, sigma=sigmay, obs=obsy)
    gauss2d = zfit.pdf.ProductPDF([gaussx, gaussy])

    # Create binned version
    n_yield = 10000
    gauss2d = gauss2d.create_extended(n_yield)

    axisx = zfit.binned.RegularBinning(30, -5, 10, name="x")
    axisy = zfit.binned.RegularBinning(25, -10, 10, name="y")
    obs_binned = zfit.Space(["x", "y"], binning=[axisx, axisy])
    gauss2d_binned = BinnedFromUnbinnedPDF(pdf=gauss2d, space=obs_binned)

    # Create histogram PDF for comparison
    data_binned = gauss2d_binned.to_binneddata()
    hist_pdf = zfit.pdf.HistogramPDF(data=data_binned, extended=True)

    return gauss2d, gauss2d_binned, hist_pdf, obs2d, obs_binned


def test_pdf_unbinned_1d(gauss_1d_pdfs):
    """Test pdf() method with unbinned 1D data."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Create unbinned data points
    x = znp.array([0.0, 2.0, 4.0, -3.0])
    data = zfit.Data.from_tensor(obs=obs, tensor=x)

    # Get bin information
    bin_edges = obs_binned.binning.edges[0]
    bin_indices = np.digitize(x, bin_edges) - 1

    # Evaluate PDFs
    pdf_binned = gauss_binned.pdf(data)
    pdf_hist = hist_pdf.pdf(data)

    # Get binned values for comparison
    pdf_binned_space = gauss_binned.pdf(obs_binned)

    # Check shapes
    assert pdf_binned.shape == (4,)
    assert pdf_hist.shape == (4,)

    # Check that unbinned evaluation returns the bin values
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(pdf_binned_space):
            expected_value = pdf_binned_space[bin_idx]
            np.testing.assert_allclose(pdf_binned[i], expected_value, rtol=1e-5)
            np.testing.assert_allclose(pdf_hist[i], expected_value, rtol=1e-5)

    # Test with raw numpy array
    pdf_binned_raw = gauss_binned.pdf(x)
    np.testing.assert_allclose(pdf_binned, pdf_binned_raw)


def test_pdf_unbinned_2d(gauss_2d_pdfs):
    """Test pdf() method with unbinned 2D data."""
    gauss2d, gauss2d_binned, hist_pdf, obs2d, obs_binned = gauss_2d_pdfs

    # Create specific test points
    x = znp.array([[0.0, 0.0], [1.0, -2.0], [3.0, 5.0], [-2.0, -8.0]])
    data = zfit.Data.from_tensor(obs=obs2d, tensor=x)

    # Evaluate PDFs
    pdf_binned = gauss2d_binned.pdf(data)
    pdf_hist = hist_pdf.pdf(data)

    # Check shapes
    assert pdf_binned.shape == (4,)
    assert pdf_hist.shape == (4,)

    # Check that both binned PDFs give same results
    np.testing.assert_allclose(pdf_binned, pdf_hist, rtol=1e-5)


def test_ext_pdf_unbinned_1d(gauss_1d_pdfs):
    """Test ext_pdf() method with unbinned 1D data."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Create unbinned data points
    x = znp.array([-1.0, 0.0, 2.0, 3.5])
    data = zfit.Data.from_tensor(obs=obs, tensor=x)

    # Evaluate extended PDFs
    ext_pdf_binned = gauss_binned.ext_pdf(data)
    ext_pdf_hist = hist_pdf.ext_pdf(data)

    # Check shapes
    assert ext_pdf_binned.shape == (4,)
    assert ext_pdf_hist.shape == (4,)

    # Check relationship between pdf and ext_pdf
    pdf_binned = gauss_binned.pdf(data)
    yield_value = gauss_binned.get_yield()
    np.testing.assert_allclose(ext_pdf_binned, pdf_binned * yield_value, rtol=1e-5)

    # Check that both binned PDFs give same results
    np.testing.assert_allclose(ext_pdf_binned, ext_pdf_hist, rtol=1e-5)


def test_ext_pdf_unbinned_2d(gauss_2d_pdfs):
    """Test ext_pdf() method with unbinned 2D data."""
    gauss2d, gauss2d_binned, hist_pdf, obs2d, obs_binned = gauss_2d_pdfs

    # Create test points
    x = znp.array([[1.0, -2.0], [0.0, 0.0], [-3.0, 5.0], [4.0, -7.0]])
    data = zfit.Data.from_tensor(obs=obs2d, tensor=x)

    # Evaluate extended PDFs
    ext_pdf_binned = gauss2d_binned.ext_pdf(data)
    ext_pdf_hist = hist_pdf.ext_pdf(data)

    # Check shapes
    assert ext_pdf_binned.shape == (4,)
    assert ext_pdf_hist.shape == (4,)

    # Check relationship with pdf
    pdf_binned = gauss2d_binned.pdf(data)
    yield_value = gauss2d_binned.get_yield()
    np.testing.assert_allclose(ext_pdf_binned, pdf_binned * yield_value, rtol=1e-5)

    # Check that both binned PDFs give same results
    np.testing.assert_allclose(ext_pdf_binned, ext_pdf_hist, rtol=1e-5)


def test_counts_unbinned_1d(gauss_1d_pdfs):
    """Test counts() method with unbinned 1D data."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Create unbinned data points
    x = znp.array([0.0, 2.0, 4.0, -2.5])
    data = zfit.Data.from_tensor(obs=obs, tensor=x)

    # Get bin information
    bin_edges = obs_binned.binning.edges[0]
    bin_indices = np.digitize(x, bin_edges) - 1

    # Evaluate counts
    counts_binned = gauss_binned.counts(data)
    counts_hist = hist_pdf.counts(data)

    # Get binned counts for comparison
    counts_binned_space = gauss_binned.counts(obs_binned)

    # Check shapes
    assert counts_binned.shape == (4,)
    assert counts_hist.shape == (4,)

    # Check that unbinned evaluation returns the bin counts
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(counts_binned_space):
            expected_count = counts_binned_space[bin_idx]
            np.testing.assert_allclose(counts_binned[i], expected_count, rtol=1e-5)
            np.testing.assert_allclose(counts_hist[i], expected_count, rtol=1e-5)

    # Check relationship with ext_pdf
    ext_pdf_binned = gauss_binned.ext_pdf(data)
    bin_widths = obs_binned.binning.widths[0]
    # For unbinned data, counts = ext_pdf * bin_width of the bin containing the point
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(bin_widths):
            expected_count = ext_pdf_binned[i] * bin_widths[bin_idx]
            np.testing.assert_allclose(counts_binned[i], expected_count, rtol=1e-5)


def test_counts_unbinned_2d(gauss_2d_pdfs):
    """Test counts() method with unbinned 2D data."""
    gauss2d, gauss2d_binned, hist_pdf, obs2d, obs_binned = gauss_2d_pdfs

    # Create test points
    x = znp.array([[0.0, 0.0], [2.0, -3.0], [-1.0, 2.0], [4.0, -5.0]])
    data = zfit.Data.from_tensor(obs=obs2d, tensor=x)

    # Evaluate counts
    counts_binned = gauss2d_binned.counts(data)
    counts_hist = hist_pdf.counts(data)

    # Check shapes
    assert counts_binned.shape == (4,)
    assert counts_hist.shape == (4,)

    # Check that both binned PDFs give same results
    np.testing.assert_allclose(counts_binned, counts_hist, rtol=1e-5)

    # Check that counts are positive where PDF is positive
    ext_pdf_binned = gauss2d_binned.ext_pdf(data)
    positive_mask = ext_pdf_binned > 1e-10
    assert np.all(counts_binned[positive_mask] > 0)


def test_rel_counts_unbinned_1d(gauss_1d_pdfs):
    """Test rel_counts() method with unbinned 1D data."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Create unbinned data points
    x = znp.array([-3.0, -1.0, 1.0, 3.0, 5.0])
    data = zfit.Data.from_tensor(obs=obs, tensor=x)

    # Get bin information
    bin_edges = obs_binned.binning.edges[0]
    bin_indices = np.digitize(x, bin_edges) - 1

    # Evaluate relative counts
    rel_counts_binned = gauss_binned.rel_counts(data)
    rel_counts_hist = hist_pdf.rel_counts(data)

    # Get binned rel_counts for comparison
    rel_counts_binned_space = gauss_binned.rel_counts(obs_binned)

    # Check shapes
    assert rel_counts_binned.shape == (5,)
    assert rel_counts_hist.shape == (5,)

    # Check that unbinned evaluation returns the bin rel_counts
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(rel_counts_binned_space):
            expected_rel_count = rel_counts_binned_space[bin_idx]
            np.testing.assert_allclose(rel_counts_binned[i], expected_rel_count, rtol=1e-5)
            np.testing.assert_allclose(rel_counts_hist[i], expected_rel_count, rtol=1e-5)

    # For extended PDF, check relationship with counts
    counts_binned = gauss_binned.counts(data)
    yield_value = gauss_binned.get_yield()
    np.testing.assert_allclose(counts_binned, rel_counts_binned * yield_value, rtol=1e-5)

    # Check relationship with pdf
    pdf_binned = gauss_binned.pdf(data)
    bin_widths = obs_binned.binning.widths[0]
    # For unbinned data, rel_counts = pdf * bin_width of the bin containing the point
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(bin_widths):
            expected_rel_count = pdf_binned[i] * bin_widths[bin_idx]
            np.testing.assert_allclose(rel_counts_binned[i], expected_rel_count, rtol=1e-5)


def test_rel_counts_unbinned_2d(gauss_2d_pdfs):
    """Test rel_counts() method with unbinned 2D data."""
    gauss2d, gauss2d_binned, hist_pdf, obs2d, obs_binned = gauss_2d_pdfs

    # Create test points
    x = znp.array([[0.5, -1.5], [2.5, 3.0], [-2.0, -4.0], [3.5, 7.0]])
    data = zfit.Data.from_tensor(obs=obs2d, tensor=x)

    # Evaluate relative counts
    rel_counts_binned = gauss2d_binned.rel_counts(data)
    rel_counts_hist = hist_pdf.rel_counts(data)

    # Check shapes
    assert rel_counts_binned.shape == (4,)
    assert rel_counts_hist.shape == (4,)

    # Check that both binned PDFs give same results
    np.testing.assert_allclose(rel_counts_binned, rel_counts_hist, rtol=1e-5)

    # Check relationship with counts
    counts_binned = gauss2d_binned.counts(data)
    yield_value = gauss2d_binned.get_yield()
    np.testing.assert_allclose(counts_binned, rel_counts_binned * yield_value, rtol=1e-5)


def test_binned_data_still_works(gauss_1d_pdfs):
    """Test that methods still work correctly with binned data."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Test with binned data (the space itself)
    pdf_binned = gauss_binned.pdf(obs_binned)
    ext_pdf_binned = gauss_binned.ext_pdf(obs_binned)
    counts_binned = gauss_binned.counts(obs_binned)
    rel_counts_binned = gauss_binned.rel_counts(obs_binned)

    # Check shapes match binning
    expected_shape = (50,)  # 50 bins
    assert pdf_binned.shape == expected_shape
    assert ext_pdf_binned.shape == expected_shape
    assert counts_binned.shape == expected_shape
    assert rel_counts_binned.shape == expected_shape

    # Check relationships
    yield_value = gauss_binned.get_yield()
    np.testing.assert_allclose(ext_pdf_binned, pdf_binned * yield_value, rtol=1e-5)
    np.testing.assert_allclose(counts_binned, rel_counts_binned * yield_value, rtol=1e-5)

    # Check normalization
    bin_widths = obs_binned.binning.widths[0]
    integral = np.sum(pdf_binned * bin_widths)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-3)

    # Check sum of counts
    np.testing.assert_allclose(np.sum(counts_binned), 10000, rtol=1e-3)  # We know the yield is 10000
    np.testing.assert_allclose(np.sum(rel_counts_binned), 1.0, rtol=1e-3)


def test_data_outside_space(gauss_1d_pdfs):
    """Test behavior when unbinned data is outside the PDF space."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Create data points outside the space
    x_outside = znp.array([-10.0, -6.0, 11.0, 15.0])
    data_outside = zfit.Data.from_tensor(obs=obs, tensor=x_outside)

    # Evaluate all methods
    pdf_outside = gauss_binned.pdf(x_outside)
    ext_pdf_outside = gauss_binned.ext_pdf(x_outside)
    counts_outside = gauss_binned.counts(x_outside)
    rel_counts_outside = gauss_binned.rel_counts(x_outside)

    # Check that values outside space are zero
    expected_zeros = znp.array([0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(pdf_outside, expected_zeros)
    np.testing.assert_allclose(ext_pdf_outside, expected_zeros)
    np.testing.assert_allclose(counts_outside, expected_zeros)
    np.testing.assert_allclose(rel_counts_outside, expected_zeros)

    # Create mixed data (some inside, some outside)
    x_mixed = znp.array([-10.0, 0.0, 5.0, 15.0])
    pdf_mixed = gauss_binned.pdf(x_mixed)

    # Check that only outside points are zero
    assert pdf_mixed[0] == 0.0  # outside
    assert pdf_mixed[1] > 0.0   # inside
    assert pdf_mixed[2] > 0.0   # inside
    assert pdf_mixed[3] == 0.0  # outside


def test_consistency_between_methods(gauss_1d_pdfs):
    """Test consistency between different methods."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Create test data
    x = znp.linspace(-4, 9, 50)
    data = zfit.Data.from_tensor(obs=obs, tensor=x)

    # Get all values
    pdf = gauss_binned.pdf(data)
    ext_pdf = gauss_binned.ext_pdf(data)
    counts = gauss_binned.counts(data)
    rel_counts = gauss_binned.rel_counts(data)
    yield_value = gauss_binned.get_yield()

    # Test relationships
    np.testing.assert_allclose(ext_pdf, pdf * yield_value, rtol=1e-5)
    np.testing.assert_allclose(counts, rel_counts * yield_value, rtol=1e-5)

    # Test relationship between pdf and rel_counts
    # For each point, find its bin and check pdf = rel_counts / bin_width
    bin_edges = obs_binned.binning.edges[0]
    bin_widths = obs_binned.binning.widths[0]
    bin_indices = np.digitize(x, bin_edges) - 1

    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(bin_widths):
            expected_pdf = rel_counts[i] / bin_widths[bin_idx]
            np.testing.assert_allclose(pdf[i], expected_pdf, rtol=1e-5)


def test_variable_binning_unbinned_data():
    """Test with variable binning and unbinned data."""
    # Create unbinned Gaussian
    mu = zfit.Parameter("mu_var", 2.0)
    sigma = zfit.Parameter("sigma_var", 1.5)
    obs = zfit.Space("x", (-5, 10))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Create variable binning with more bins near the peak
    edges = np.concatenate([
        np.linspace(-5, 0, 10),
        np.linspace(0, 4, 20)[1:],  # More bins near peak
        np.linspace(4, 10, 10)[1:]
    ])
    axis = zfit.binned.VariableBinning(edges, name="x")
    obs_binned = zfit.Space("x", binning=[axis])

    n_yield = 10000
    gauss = gauss.create_extended(n_yield)
    gauss_binned = BinnedFromUnbinnedPDF(pdf=gauss, space=obs_binned)

    # Test with unbinned data
    x = znp.array([-3.0, 0.0, 2.0, 6.0])
    data = zfit.Data.from_tensor(obs=obs, tensor=x)

    # Get bin information
    bin_indices = np.digitize(x, edges) - 1

    # Evaluate all methods
    pdf_binned = gauss_binned.pdf(data)
    ext_pdf_binned = gauss_binned.ext_pdf(data)
    counts_binned = gauss_binned.counts(data)
    rel_counts_binned = gauss_binned.rel_counts(data)

    # Check shapes
    assert pdf_binned.shape == (4,)
    assert ext_pdf_binned.shape == (4,)
    assert counts_binned.shape == (4,)
    assert rel_counts_binned.shape == (4,)

    # Get binned values for comparison
    pdf_binned_space = gauss_binned.pdf(obs_binned)
    counts_binned_space = gauss_binned.counts(obs_binned)

    # Check that unbinned evaluation returns the bin values
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < len(pdf_binned_space):
            np.testing.assert_allclose(pdf_binned[i], pdf_binned_space[bin_idx], rtol=1e-5)
            np.testing.assert_allclose(counts_binned[i], counts_binned_space[bin_idx], rtol=1e-5)

    # Check consistency
    np.testing.assert_allclose(ext_pdf_binned, pdf_binned * n_yield, rtol=1e-5)
    np.testing.assert_allclose(counts_binned, rel_counts_binned * n_yield, rtol=1e-5)


def test_hist_comparison_1d(gauss_1d_pdfs):
    """Test that binned PDF results are mathematically consistent with hist operations."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Get zfit binned PDF values
    zfit_counts = gauss_binned.counts(obs_binned)
    zfit_rel_counts = gauss_binned.rel_counts(obs_binned)
    zfit_pdf = gauss_binned.pdf(obs_binned)

    # Test that rel_counts sum to 1
    np.testing.assert_allclose(np.sum(zfit_rel_counts), 1.0, rtol=1e-3)

    # Test that counts = rel_counts * yield
    np.testing.assert_allclose(zfit_counts, zfit_rel_counts * 10000, rtol=1e-5)

    # Test density calculation: pdf = rel_counts / bin_width
    bin_widths = obs_binned.binning.widths[0]
    expected_density = zfit_rel_counts / bin_widths
    np.testing.assert_allclose(zfit_pdf, expected_density, rtol=1e-5)

    # Test counts sum to yield
    np.testing.assert_allclose(np.sum(zfit_counts), 10000, rtol=1e-3)


def test_hist_comparison_2d(gauss_2d_pdfs):
    """Test that 2D binned PDF results are mathematically consistent with hist operations."""
    gauss2d, gauss2d_binned, hist_pdf, obs2d, obs_binned = gauss_2d_pdfs

    # Get zfit binned PDF values
    zfit_counts = gauss2d_binned.counts(obs_binned)
    zfit_rel_counts = gauss2d_binned.rel_counts(obs_binned)
    zfit_pdf = gauss2d_binned.pdf(obs_binned)

    # Test that rel_counts sum to 1
    np.testing.assert_allclose(np.sum(zfit_rel_counts), 1.0, rtol=1e-3)

    # Test that counts = rel_counts * yield
    np.testing.assert_allclose(zfit_counts, zfit_rel_counts * 10000, rtol=1e-5)

    # Test 2D density calculation
    bin_widths_x = obs_binned.binning.widths[0]
    bin_widths_y = obs_binned.binning.widths[1]
    # Create 2D bin width array
    bin_areas = np.outer(bin_widths_x, bin_widths_y)
    expected_density = zfit_rel_counts / bin_areas
    np.testing.assert_allclose(zfit_pdf, expected_density, rtol=1e-5)


def test_hist_from_binnedpdf_conversion(gauss_1d_pdfs):
    """Test conversion from zfit BinnedData to hist and back."""
    gauss, gauss_binned, hist_pdf, obs, obs_binned = gauss_1d_pdfs

    # Get binned data from zfit PDF
    binned_data = gauss_binned.to_binneddata()

    # Convert to hist
    h = binned_data.to_hist()

    # Create HistogramPDF from hist
    hist_pdf_from_hist = zfit.pdf.HistogramPDF(data=h, extended=True)

    # Test that all methods give the same results
    test_points = znp.array([-2.0, 0.0, 2.0, 4.0])
    data = zfit.Data.from_tensor(obs=obs, tensor=test_points)

    # Compare pdf values
    pdf_original = gauss_binned.pdf(data)
    pdf_from_hist = hist_pdf_from_hist.pdf(data)
    np.testing.assert_allclose(pdf_original, pdf_from_hist, rtol=1e-10)

    # Compare ext_pdf values
    ext_pdf_original = gauss_binned.ext_pdf(data)
    ext_pdf_from_hist = hist_pdf_from_hist.ext_pdf(data)
    np.testing.assert_allclose(ext_pdf_original, ext_pdf_from_hist, rtol=1e-10)

    # Compare counts
    counts_original = gauss_binned.counts(data)
    counts_from_hist = hist_pdf_from_hist.counts(data)
    np.testing.assert_allclose(counts_original, counts_from_hist, rtol=1e-10)

    # Compare rel_counts
    rel_counts_original = gauss_binned.rel_counts(data)
    rel_counts_from_hist = hist_pdf_from_hist.rel_counts(data)
    np.testing.assert_allclose(rel_counts_original, rel_counts_from_hist, rtol=1e-10)

    # Test that hist values match binned data values
    np.testing.assert_allclose(h.values(), binned_data.values(), rtol=1e-10)

    # Test that hist variances match binned data variances (if available)
    if binned_data.variances() is not None:
        np.testing.assert_allclose(h.variances(), binned_data.variances(), rtol=1e-10)


def test_hist_density_vs_pdf_consistency():
    """Test that hist density calculations are consistent with PDF values."""
    # Create a simple uniform distribution for easy verification
    obs = zfit.Space("x", (0, 10))
    uniform = zfit.pdf.Uniform(low=0, high=10, obs=obs)
    uniform = uniform.create_extended(1000)

    # Create binned version
    nbins = 20
    axis = zfit.binned.RegularBinning(nbins, 0, 10, name="x")
    obs_binned = zfit.Space("x", binning=[axis])
    uniform_binned = BinnedFromUnbinnedPDF(pdf=uniform, space=obs_binned)

    # Get zfit PDF values at bin centers
    bin_centers = obs_binned.binning.centers[0]
    pdf_values = uniform_binned.pdf(bin_centers)

    # For a uniform distribution, PDF should be constant = 1/(upper-lower) = 1/10 = 0.1
    expected_pdf = 0.1
    np.testing.assert_allclose(pdf_values, expected_pdf, rtol=1e-10)

    # Get rel_counts and verify relationship with PDF
    rel_counts = uniform_binned.rel_counts(obs_binned)
    bin_widths = obs_binned.binning.widths[0]

    # rel_counts should equal pdf * bin_width
    expected_rel_counts = expected_pdf * bin_widths
    np.testing.assert_allclose(rel_counts, expected_rel_counts, rtol=1e-10)

    # All rel_counts should be equal for uniform distribution
    np.testing.assert_allclose(rel_counts, rel_counts[0], rtol=1e-10)

    # Sum of rel_counts should be 1
    np.testing.assert_allclose(np.sum(rel_counts), 1.0, rtol=1e-10)

    # Convert to hist and check density
    binned_data = uniform_binned.to_binneddata()
    h = binned_data.to_hist()

    # hist density should match our PDF values
    hist_density = h.density()
    np.testing.assert_allclose(hist_density, expected_pdf, rtol=1e-10)


def test_auto_ext_pdf_normalization_bug_fix():
    """Test that verifies the fix for the ext_normalization bug in _auto_ext_pdf.

    This test ensures that the _auto_ext_pdf method correctly uses normalization()
    instead of ext_normalization() when NormNotImplemented is raised.
    """
    from zfit.util.exception import NormNotImplemented

    class TestPDFWithNormalizationFix(zfit.pdf.BaseBinnedPDF):
        """PDF that specifically triggers the normalization path in _auto_ext_pdf."""

        def __init__(self, values, obs, extended):
            self._values = znp.asarray(values)
            params = {}
            super().__init__(obs=obs, extended=extended, params=params)

        def _ext_pdf(self, x, norm):
            """Always raise NormNotImplemented to trigger the normalization path."""
            if norm is not False:
                raise NormNotImplemented("Force the except block in _auto_ext_pdf")
            # Return unnormalized ext_pdf (includes yield)
            return self._values * self.get_yield()

        def _rel_counts(self, x, norm):
            """Required for normalization calculation."""
            return self._values / znp.sum(self._values)

    # Create a simple binned space
    obs = zfit.Space("x", binning=zfit.binned.RegularBinning(5, 0, 5, name="x"))

    # Create normalized values that sum to 1
    values = znp.array([0.1, 0.2, 0.3, 0.2, 0.2])
    yield_val = 1000.0

    pdf = TestPDFWithNormalizationFix(values=values, obs=obs, extended=yield_val)

    # Test that ext_pdf gives correct results (pdf * yield)
    pdf_result = pdf.pdf(obs)  # Should be values
    ext_pdf_result = pdf.ext_pdf(obs)  # Should be values * yield

    # ext_pdf should equal pdf * yield
    expected_ext_pdf = pdf_result * yield_val
    np.testing.assert_allclose(ext_pdf_result, expected_ext_pdf, rtol=1e-5)

    # Verify the values are correct (not multiplied by yield^2 as with the bug)
    expected_values = znp.array([100., 200., 300., 200., 200.])  # values * yield
    np.testing.assert_allclose(ext_pdf_result, expected_values, rtol=1e-5)

    # Test with unbinned data as well
    x_unbinned = znp.array([0.5, 2.5, 4.5])
    data_unbinned = zfit.Data.from_tensor(obs=obs.with_binning(None), tensor=x_unbinned)

    pdf_unbinned = pdf.pdf(data_unbinned)
    ext_pdf_unbinned = pdf.ext_pdf(data_unbinned)

    # Should match the relationship
    np.testing.assert_allclose(ext_pdf_unbinned, pdf_unbinned * yield_val, rtol=1e-5)
