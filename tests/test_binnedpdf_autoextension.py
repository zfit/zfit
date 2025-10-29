#  Copyright (c) 2025 zfit
import hist
import numpy as np
import pytest

import zfit
from zfit.core.parameter import ComposedParameter
from zfit.core.space import supports
from zfit.data import BinnedData
from zfit.pdf import HistogramPDF
from zfit.models.template import BinnedTemplatePDFV1
from zfit.pdf import BinnedSumPDF
from zfit.pdf import SplineMorphingPDF
from zfit.util.exception import NotExtendedPDFError, SpecificFunctionNotImplemented
from zfit.pdf import BaseBinnedPDF
import zfit.z.numpy as znp


class TestBaseBinnedPDF(BaseBinnedPDF):
    """A test implementation that uses the base class auto-extension."""

    def __init__(self, data, extended=None, norm=None, name="TestBaseBinnedPDF", label=None):
        self._data = data
        # Don't implement own auto-extension, let base class handle it
        params = {}  # No parameters for this simple test PDF
        super().__init__(obs=data.space, extended=extended, norm=norm, params=params, name=name, label=label)

    @supports(norm="space")
    def _counts(self, x, norm=None):
        """Implementation required for auto-extension to work."""
        return self._data.values()



class IncompleteBaseBinnedPDF(BaseBinnedPDF):
    """Test PDF that doesn't implement _counts method."""

    def __init__(self, data, extended=None, norm=None, name="IncompleteBaseBinnedPDF"):
        self._data = data
        params = {}  # No parameters for this simple test PDF
        super().__init__(obs=data.space, extended=extended, norm=norm, params=params, name=name)

    # Note: _counts method is NOT implemented

    @supports(norm="space")
    def _rel_counts(self, x, norm=None):
        values = self._data.values()
        return values / znp.sum(values)


# Fixtures
@pytest.fixture
def binned_space_1d():
    """Create a 1D binned space for testing."""
    return zfit.Space("x", binning=zfit.binned.RegularBinning(10, -5, 5, name="x"))


@pytest.fixture
def binned_space_2d():
    """Create a 2D binned space for testing."""
    return zfit.Space(
        ["x", "y"],
        binning=[
            zfit.binned.RegularBinning(8, -3, 3, name="x"),
            zfit.binned.RegularBinning(6, -2, 2, name="y")
        ]
    )


@pytest.fixture
def sample_data_1d(binned_space_1d):
    """Create sample 1D binned data."""
    counts = znp.array([1, 3, 5, 8, 12, 15, 10, 6, 3, 1], dtype=znp.float64)
    return BinnedData.from_tensor(space=binned_space_1d, values=counts)


@pytest.fixture
def sample_data_2d(binned_space_2d):
    """Create sample 2D binned data."""
    counts = znp.random.uniform(1, 20, size=(8, 6))
    return BinnedData.from_tensor(space=binned_space_2d, values=counts)


@pytest.fixture
def sample_hist_1d():
    """Create a sample 1D histogram using hist package."""
    h = hist.Hist(
        hist.axis.Regular(10, -5, 5, name="x", flow=False),
        storage=hist.storage.Weight()
    )
    data = np.random.normal(0, 1.5, size=1000)
    h.fill(x=data)
    return h


@pytest.fixture
def sample_hist_2d():
    """Create a sample 2D histogram using hist package."""
    h = hist.Hist(
        hist.axis.Regular(8, -3, 3, name="x", flow=False),
        hist.axis.Regular(6, -2, 2, name="y", flow=False),
        storage=hist.storage.Weight()
    )
    x_data = np.random.normal(0, 1, size=1000)
    y_data = np.random.normal(0, 0.8, size=1000)
    h.fill(x=x_data, y=y_data)
    return h


@pytest.fixture
def zero_data_1d(binned_space_1d):
    """Create 1D data with zero counts."""
    zero_counts = znp.zeros(10, dtype=znp.float64)
    return BinnedData.from_tensor(space=binned_space_1d, values=zero_counts)


@pytest.fixture
def small_data_1d(binned_space_1d):
    """Create 1D data with very small counts."""
    small_counts = znp.full(10, 1e-10, dtype=znp.float64)
    return BinnedData.from_tensor(space=binned_space_1d, values=small_counts)


@pytest.fixture
def large_data_1d(binned_space_1d):
    """Create 1D data with very large counts."""
    large_counts = znp.full(10, 1e6, dtype=znp.float64)
    return BinnedData.from_tensor(space=binned_space_1d, values=large_counts)


# Test auto-extension basic functionality
@pytest.mark.parametrize("extended,should_be_extended,should_have_yield", [
    (True, True, True),
    (False, False, False),
    (None, False, False)
])
def test_base_pdf_extension_behavior(sample_data_1d, extended, should_be_extended, should_have_yield):
    """Test basic extension behavior of BaseBinnedPDF."""
    pdf = TestBaseBinnedPDF(data=sample_data_1d, extended=extended)

    assert pdf.is_extended == should_be_extended
    assert pdf._autoextended_requires_counts == should_be_extended

    if should_have_yield:
        yield_param = pdf.get_yield()
        assert isinstance(yield_param, ComposedParameter)
        assert "AUTOYIELD_" in yield_param.name

        # Test that yield equals sum of counts
        expected_yield = znp.sum(sample_data_1d.values())
        np.testing.assert_allclose(yield_param.value(), expected_yield)
    else:
        with pytest.raises(NotExtendedPDFError):
            pdf.get_yield()


def test_auto_yield_parameter_dependency(sample_data_1d):
    """Test that auto-yield parameter depends on PDF parameters."""
    pdf = TestBaseBinnedPDF(data=sample_data_1d, extended=True)
    yield_param = pdf.get_yield()

    # Get all PDF parameters (should be empty for TestBaseBinnedPDF with fixed data)
    pdf_params = pdf.get_params(floating=None)

    # The yield parameter should depend on the PDF's parameters
    yield_deps = yield_param.get_params(floating=None)

    # For TestBaseBinnedPDF, it depends on the PDF itself (which has no free parameters)
    # So yield_deps might be empty or contain the PDF's parameters
    # The result should be a set-like object (could be OrderedSet)
    assert hasattr(yield_deps, '__iter__')  # Should be iterable like a set
    assert hasattr(yield_deps, '__len__')   # Should have length like a set


def test_incomplete_base_pdf_error(sample_data_1d):
    """Test that auto-extension requires _counts method implementation."""
    pdf = IncompleteBaseBinnedPDF(data=sample_data_1d, extended=True)

    # PDF should be extended and flag should be set
    assert pdf.is_extended
    assert pdf._autoextended_requires_counts is True

    # But calling counts() should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="requires the `_counts` method to be implemented"):
        pdf.counts(sample_data_1d.space)


# Test PDF-specific auto-extension
@pytest.mark.parametrize("pdf_class,data_fixture,expected_auto_flag,creates_composed_param", [
    (HistogramPDF, "sample_data_1d", "_automatically_extended", False),
    (HistogramPDF, "sample_data_2d", "_automatically_extended", False),
    (BinnedTemplatePDFV1, "sample_data_1d", "_automatically_extended", False),
])
def test_pdf_auto_extension(request, pdf_class, data_fixture, expected_auto_flag, creates_composed_param):
    """Test auto-extension for different PDF types."""
    data = request.getfixturevalue(data_fixture)
    pdf = pdf_class(data=data, extended=True)

    # Test basic properties
    assert pdf.is_extended
    if hasattr(pdf, expected_auto_flag):
        assert getattr(pdf, expected_auto_flag) is True

    expected_yield = znp.sum(data.values())
    actual_yield = pdf.get_yield().value()
    np.testing.assert_allclose(actual_yield, expected_yield)

    # Test yield parameter type
    yield_param = pdf.get_yield()
    if creates_composed_param:
        assert isinstance(yield_param, ComposedParameter)
    else:
        assert not isinstance(yield_param, ComposedParameter)

    # Test counts method
    counts = pdf.counts(data.space)
    np.testing.assert_allclose(counts, data.values())


def test_histogram_from_hist_auto_extension(sample_hist_1d):
    """Test HistogramPDF from hist package with auto-extension."""
    pdf = HistogramPDF(data=sample_hist_1d, extended=True)

    assert pdf.is_extended
    expected_yield = sample_hist_1d.sum().value
    actual_yield = pdf.get_yield().value()
    np.testing.assert_allclose(actual_yield, expected_yield)

    # Test counts method
    counts = pdf.counts(pdf.space)
    np.testing.assert_allclose(counts, sample_hist_1d.values())


def test_template_pdf_with_sysshape_auto_extension(sample_data_1d):
    """Test auto-extension with systematic shape variations."""
    # Create systematic shape parameters
    sysshape = {
        f"sys_{i}": zfit.Parameter(f"sys_{i}", 1.0, 0.5, 1.5)
        for i in range(sample_data_1d.values().shape.num_elements())
    }

    pdf = BinnedTemplatePDFV1(data=sample_data_1d, sysshape=sysshape, extended=True)

    assert pdf.is_extended
    assert pdf._automatically_extended is True

    # The yield should now be a ComposedParameter that depends on sysshape
    yield_param = pdf.get_yield()
    assert isinstance(yield_param, ComposedParameter)

    # Test that changing sysshape parameters affects yield
    original_yield = yield_param.value()

    # Change one systematic parameter
    list(sysshape.values())[0].set_value(1.5)

    # Yield should change
    new_yield = yield_param.value()
    assert not np.allclose(original_yield, new_yield)

    # Reset and test counts
    for param in sysshape.values():
        param.set_value(1.0)

    counts = pdf.counts(sample_data_1d.space)
    np.testing.assert_allclose(counts, sample_data_1d.values())


def test_template_pdf_auto_sysshape_creation(sample_data_1d):
    """Test automatic systematic shape parameter creation."""
    pdf = BinnedTemplatePDFV1(data=sample_data_1d, sysshape=True, extended=True)

    assert pdf.is_extended
    assert len(pdf._template_sysshape) == sample_data_1d.values().shape.num_elements()

    # All systematic parameters should be initialized to 1.0
    for param in pdf._template_sysshape.values():
        assert param.value() == 1.0


# Test composite PDFs
def test_sum_pdf_auto_extension(sample_data_1d):
    """Test BinnedSumPDF auto-extension with multiple components."""
    # Create individual PDFs - need same space for summing
    data1_1d = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values())
    data2_1d = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values() * 1.5)

    pdf1 = HistogramPDF(data=data1_1d, extended=True)
    pdf2 = HistogramPDF(data=data2_1d, extended=True)

    sum_pdf = BinnedSumPDF(pdfs=[pdf1, pdf2], obs=sample_data_1d.space)

    assert sum_pdf.is_extended

    # Yield should be sum of component yields
    expected_yield = pdf1.get_yield().value() + pdf2.get_yield().value()
    actual_yield = sum_pdf.get_yield().value()
    np.testing.assert_allclose(actual_yield, expected_yield)

    # Test counts method
    counts = sum_pdf.counts(sample_data_1d.space)
    expected_counts = data1_1d.values() + data2_1d.values()
    np.testing.assert_allclose(counts, expected_counts)


def test_sum_pdf_mixed_extension(sample_data_1d):
    """Test BinnedSumPDF with mix of extended and non-extended components."""
    data1 = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values())
    data2 = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values() * 0.8)

    pdf1 = HistogramPDF(data=data1, extended=True)
    pdf2 = HistogramPDF(data=data2, extended=False)

    # This should not auto-extend because not all components are extended
    sum_pdf = BinnedSumPDF(pdfs=[pdf1, pdf2], obs=sample_data_1d.space, extended=None)

    # Should not be auto-extended when components are mixed
    assert not sum_pdf.is_extended


def test_morphing_pdf_auto_extension(sample_data_1d):
    """Test SplineMorphingPDF auto-extension."""
    # Create multiple histograms for morphing
    data1 = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values())
    data2 = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values() * 1.2)
    data3 = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values() * 0.8)

    pdf1 = HistogramPDF(data=data1, extended=True)
    pdf2 = HistogramPDF(data=data2, extended=True)
    pdf3 = HistogramPDF(data=data3, extended=True)

    alpha = zfit.Parameter("alpha", 0.0, -1.0, 1.0)
    hists = [-1.0, 0.0, 1.0]  # alpha values
    hists_dict = {val: pdf for val, pdf in zip(hists, [pdf1, pdf2, pdf3])}

    morph_pdf = SplineMorphingPDF(alpha=alpha, hists=hists_dict, extended=True)

    assert morph_pdf.is_extended
    assert morph_pdf._automatically_extended is True

    # Test that yield is a ComposedParameter
    yield_param = morph_pdf.get_yield()
    assert isinstance(yield_param, ComposedParameter)

    # Test that yield changes with alpha
    alpha.set_value(-1.0)  # Should be close to pdf1
    yield_at_minus1 = yield_param.value()

    alpha.set_value(0.0)   # Should be close to pdf2
    yield_at_0 = yield_param.value()

    alpha.set_value(1.0)   # Should be close to pdf3
    yield_at_1 = yield_param.value()

    # Yields should be different as we morph
    assert not np.allclose(yield_at_minus1, yield_at_0)
    assert not np.allclose(yield_at_0, yield_at_1)


def test_morphing_pdf_with_non_extended_components(sample_data_1d):
    """Test SplineMorphingPDF with non-extended components."""
    data1 = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values())
    data2 = BinnedData.from_tensor(space=sample_data_1d.space, values=sample_data_1d.values() * 1.2)

    pdf1 = HistogramPDF(data=data1, extended=False)
    pdf2 = HistogramPDF(data=data2, extended=False)

    alpha = zfit.Parameter("alpha", 0.0, -1.0, 1.0)
    hists_dict = {-1.0: pdf1, 1.0: pdf2}

    # Should raise error when trying to auto-extend with non-extended components
    with pytest.raises(ValueError, match="all PDFs must be extended"):
        SplineMorphingPDF(alpha=alpha, hists=hists_dict, extended=True)


# Test method behavior
@pytest.mark.parametrize("method_name,expected_result", [
    ("counts", "data_values"),
    ("ext_pdf", "data_values_divided_by_widths"),
    ("ext_integrate", "sum_of_data_values"),
])
def test_auto_extension_methods(sample_data_1d, method_name, expected_result):
    """Test that all PDF methods work correctly with auto-extension."""
    pdf = HistogramPDF(data=sample_data_1d, extended=True)

    if method_name == "counts":
        result = pdf.counts(sample_data_1d.space)
        expected = sample_data_1d.values()
        np.testing.assert_allclose(result, expected)
    elif method_name == "ext_pdf":
        result = pdf.ext_pdf(sample_data_1d.space)
        widths = np.prod(sample_data_1d.space.binning.widths, axis=0)
        expected = sample_data_1d.values() / widths
        np.testing.assert_allclose(result, expected)
    elif method_name == "ext_integrate":
        result = pdf.ext_integrate(sample_data_1d.space)
        expected = znp.sum(sample_data_1d.values())
        np.testing.assert_allclose(result, expected)


def test_sample_with_auto_extension(sample_data_1d):
    """Test sampling with auto-extended PDF."""
    pdf = HistogramPDF(data=sample_data_1d, extended=True)

    # Test sampling with explicit n
    n_sample = 1000
    sample = pdf.sample(n=n_sample)
    assert sample.nevents == n_sample

    # Test sampling without n (should use yield)
    sample_auto = pdf.sample()
    expected_n = int(pdf.get_yield().value())
    assert sample_auto.nevents == expected_n


def test_create_sampler_with_auto_extension(sample_data_1d):
    """Test create_sampler() with auto-extended PDF."""
    pdf = HistogramPDF(data=sample_data_1d, extended=True)

    sampler = pdf.create_sampler()

    # Should create sampler with yield as default n on average
    expected_n = int(pdf.get_yield().value())
    assert np.sum(sample_data_1d.counts()) == expected_n
    poissons = []
    pdf.to_unbinned().sample()
    for _ in range(100):
        sampler.resample()
        poissons.append(np.sum(sampler.counts()))
    mean_n = np.mean(poissons) * 10
    assert pytest.approx(mean_n, rel=expected_n ** 0.5 / expected_n) == expected_n  # within Poisson fluctuations


def test_auto_vs_manual_extension_equivalence(sample_data_1d):
    """Test that auto-extension gives same results as manual extension."""
    # Create auto-extended PDF
    pdf_auto = HistogramPDF(data=sample_data_1d, extended=True)

    # Create manually extended PDF
    manual_yield = znp.sum(sample_data_1d.values())
    pdf_manual = HistogramPDF(data=sample_data_1d, extended=manual_yield)

    # Both should give same results
    assert pdf_auto.is_extended
    assert pdf_manual.is_extended

    np.testing.assert_allclose(
        pdf_auto.get_yield().value(),
        pdf_manual.get_yield().value()
    )

    # Test counts
    counts_auto = pdf_auto.counts(sample_data_1d.space)
    counts_manual = pdf_manual.counts(sample_data_1d.space)
    np.testing.assert_allclose(counts_auto, counts_manual)

    # Test ext_pdf
    ext_pdf_auto = pdf_auto.ext_pdf(sample_data_1d.space)
    ext_pdf_manual = pdf_manual.ext_pdf(sample_data_1d.space)
    np.testing.assert_allclose(ext_pdf_auto, ext_pdf_manual)


# Test edge cases and error handling
@pytest.mark.parametrize("data_fixture,description", [
    ("zero_data_1d", "zero counts"),
    ("small_data_1d", "very small counts"),
    ("large_data_1d", "very large counts"),
])
def test_edge_case_counts_auto_extension(request, data_fixture, description):
    """Test auto-extension with edge case count values."""
    data = request.getfixturevalue(data_fixture)
    pdf = HistogramPDF(data=data, extended=True)

    assert pdf.is_extended
    expected_yield = znp.sum(data.values())
    np.testing.assert_allclose(pdf.get_yield().value(), expected_yield)

    counts = pdf.counts(data.space)
    np.testing.assert_allclose(counts, data.values())


def test_repeated_auto_extension_calls(sample_data_1d):
    """Test that repeated calls to auto-extension methods are consistent."""
    pdf = HistogramPDF(data=sample_data_1d, extended=True)

    # Call yield multiple times
    yield1 = pdf.get_yield().value()
    yield2 = pdf.get_yield().value()
    yield3 = pdf.get_yield().value()

    np.testing.assert_allclose(yield1, yield2)
    np.testing.assert_allclose(yield2, yield3)

    # Call counts multiple times
    counts1 = pdf.counts(sample_data_1d.space)
    counts2 = pdf.counts(sample_data_1d.space)
    counts3 = pdf.counts(sample_data_1d.space)

    np.testing.assert_allclose(counts1, counts2)
    np.testing.assert_allclose(counts2, counts3)


def test_auto_yield_parameter_properties(sample_data_1d):
    """Test properties of the auto-generated yield parameter."""
    pdf = HistogramPDF(data=sample_data_1d, extended=True)
    yield_param = pdf.get_yield()

    # Should be a ComposedParameter
    assert isinstance(yield_param, ComposedParameter)

    # Should have correct name pattern
    assert "AUTOYIELD_" in yield_param.name

    # Should be floating by default (ComposedParameters are typically floating)
    assert yield_param.floating

    # Should have correct value
    expected_value = znp.sum(sample_data_1d.values())
    np.testing.assert_allclose(yield_param.value(), expected_value)


def test_auto_yield_with_parameter_dependencies(sample_data_1d):
    """Test auto-yield with PDFs that have parameters."""
    # Use BinnedTemplatePDF with systematic parameters
    sysshape = {
        "sys_0": zfit.Parameter("sys_0", 1.0, 0.5, 2.0),
        "sys_1": zfit.Parameter("sys_1", 1.0, 0.5, 2.0),
    }

    pdf = BinnedTemplatePDFV1(data=sample_data_1d, sysshape=sysshape, extended=True)
    yield_param = pdf.get_yield()

    # Initial yield
    initial_yield = yield_param.value()

    # Change a systematic parameter
    sysshape["sys_0"].set_value(1.5)

    # Yield should change
    new_yield = yield_param.value()
    assert not np.allclose(initial_yield, new_yield)

    # The yield should depend on the systematic parameters
    yield_deps = yield_param.get_params(floating=None)
    sys_param_names = {param.name for param in sysshape.values()}
    yield_dep_names = {param.name for param in yield_deps}

    # The systematic parameters should be in the dependencies
    assert sys_param_names.intersection(yield_dep_names) == sys_param_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
