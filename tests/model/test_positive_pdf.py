#  Copyright (c) 2025 zfit
import pytest
import numpy as np

import zfit
from zfit.util.exception import SpecificFunctionNotImplemented, NotExtendedPDFError

limits1 = (-4, 3)
limits2 = (-2, 5)
obs1 = "obs1"

space1 = zfit.Space(obs=obs1, limits=limits1)
space2 = zfit.Space(obs=obs1, limits=limits2)


def test_positive_pdf_basic():
    """Test basic PositivePDF functionality with a simple Gaussian."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    positive_gauss = gauss.to_positive(epsilon=1e-10)

    # Test that the positive PDF maintains basic properties
    assert positive_gauss.obs == gauss.obs
    assert positive_gauss.norm == gauss.norm
    assert positive_gauss.is_extended == gauss.is_extended

    # Test evaluation - for a Gaussian, values should be positive, so epsilon should not change much
    x_test = np.array([[-2.0], [0.0], [2.0]])
    original_vals = gauss.pdf(x_test)
    positive_vals = positive_gauss.pdf(x_test)

    # Values should be very close since Gaussian doesn't produce negative values
    np.testing.assert_allclose(original_vals.numpy(), positive_vals.numpy(), rtol=1e-8)


def test_positive_pdf_negative_weights_kde():
    """Test PositivePDF with KDE that has negative weights (the main use case)."""
    # Create data with negative weights - this was the original problem case
    data_vals = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([1.0, 1.0, -0.1])  # One negative weight

    data = zfit.data.Data.from_numpy(obs=space1, array=data_vals, weights=weights)
    kde = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')

    # Create a positive version
    positive_kde = kde.to_positive()

    # Test that properties are preserved
    assert positive_kde.obs == kde.obs
    assert positive_kde.norm == kde.norm

    # Test evaluation without normalization (since KDE with negative weights can have normalization issues)
    test_x = np.array([[0.0], [1.0], [2.0]])
    positive_vals = positive_kde.pdf(test_x).numpy()

    # Verify no NaN or overly negative values
    assert not np.any(np.isnan(positive_vals)), "Positive PDF should not have NaN values"
    assert np.all(positive_vals >= 1e-310), "Positive PDF should not have values below epsilon"


def test_positive_pdf_integration():
    """Test that PositivePDF integration works correctly."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    positive_gauss = gauss.to_positive()

    # Test integration - should give similar results for a well-behaved PDF
    integral_original = gauss.integrate(limits=space1)
    integral_positive = positive_gauss.integrate(limits=space1)

    # Should be very close (both should be ~1)
    np.testing.assert_allclose(integral_original.numpy(), integral_positive.numpy(), rtol=1e-6)


def test_positive_pdf_custom_epsilon():
    """Test PositivePDF with custom epsilon."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    custom_epsilon = 1e-5
    positive_gauss = gauss.to_positive(epsilon=custom_epsilon)

    # Test that the epsilon is respected (convert to numpy for comparison)
    assert positive_gauss.epsilon.numpy() == custom_epsilon

    # For this test, we create an artificial case where epsilon matters
    # Use a very small test region where Gaussian is essentially zero
    x_test = np.array([[-100.0]])  # Far from the mean
    positive_vals = positive_gauss.pdf(x_test).numpy()

    # The value should be at least epsilon
    assert np.all(positive_vals >= custom_epsilon)


def test_positive_pdf_extended():
    """Test PositivePDF with extended PDFs."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    extended_gauss = gauss.create_extended(100.0)
    positive_extended = extended_gauss.to_positive()

    # Test that extension is preserved
    assert positive_extended.is_extended
    assert positive_extended.get_yield().numpy() == 100.0


def test_positive_pdf_default_epsilon():
    """Test that epsilon has correct default."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)

    # Create positive PDF with default epsilon
    positive_gauss = gauss.to_positive()

    # Test that default epsilon is set correctly (convert to numpy for comparison)
    assert positive_gauss.epsilon.numpy() == 1e-100  # Default epsilon


# Custom polynomial PDF that can go below zero for testing
class CustomPolynomialPDF(zfit.pdf.BasePDF):
    """A custom second-degree polynomial PDF: f(x) = a*x^2 + b*x + c

    This PDF is designed to test positive behavior as it can produce negative values
    depending on the coefficients and x values.
    """
    _N_OBS = 1

    def __init__(self, a, b, c, obs, extended=None, norm=None, name="CustomPolynomialPDF", label=None):
        params = {"a": a, "b": b, "c": c}
        super().__init__(obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @zfit.core.space.supports()
    def _unnormalized_pdf(self, x, params):
        a = params["a"]
        b = params["b"]
        c = params["c"]
        x_val = x.unstack_x()
        return a * x_val**2 + b * x_val + c


def test_positive_pdf_custom_negative_polynomial():
    """Test PositivePDF with a custom polynomial that goes below zero."""
    # Create polynomial f(x) = -x^2 + 1, which is positive at x=0 but negative for |x| > 1
    a = zfit.Parameter("a", -1.0)  # coefficient for x^2 term (negative)
    b = zfit.Parameter("b", 0.0)   # coefficient for x term
    c = zfit.Parameter("c", 1.0)   # constant term

    poly_pdf = CustomPolynomialPDF(a=a, b=b, c=c, obs=space1)

    # Test that the polynomial actually goes negative outside |x| > 1
    x_negative = np.array([[2.0]])  # Should give f(2) = -4 + 1 = -3
    unclamped_vals = poly_pdf.pdf(x_negative, norm=False).numpy()
    assert np.any(unclamped_vals < 0), "Polynomial should produce negative values for x=2"

    # Create positive version with epsilon
    epsilon = 1e-10
    positive_poly = poly_pdf.to_positive(epsilon=epsilon)

    # Test that the negative values are made positive to at least epsilon
    positive_vals = positive_poly.pdf(x_negative, norm=False).numpy()
    assert np.all(positive_vals >= epsilon), "Positive PDF should not have values below epsilon"

    # Test at a point where the polynomial is positive (x=0, f(0)=1)
    x_positive = np.array([[0.0]])
    positive_vals = positive_poly.pdf(x_positive, norm=False).numpy()
    assert np.all(positive_vals > epsilon), "Positive values should remain above epsilon"

    # Verify that values at x=0 are approximately 1 (the original polynomial value)
    expected_at_zero = c.numpy()  # c = 1.0
    np.testing.assert_allclose(positive_vals, expected_at_zero, rtol=1e-6)


def test_positive_pdf_extended_basic():
    """Test PositivePDF with extended PDFs."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    yield_param = 100.0
    extended_gauss = gauss.create_extended(yield_param)

    # Test with custom epsilon
    epsilon = 1e-8
    positive_extended = extended_gauss.to_positive(epsilon=epsilon)

    # Test that extension properties are preserved
    assert positive_extended.is_extended
    assert positive_extended.get_yield().numpy() == yield_param

    # Test that epsilon is set correctly (convert to numpy for comparison)
    assert positive_extended.epsilon.numpy() == epsilon


def test_positive_pdf_extended_custom_polynomial():
    """Test PositivePDF with extended custom polynomial that can go negative."""
    # Create polynomial f(x) = -x^2 + 1
    a = zfit.Parameter("poly_a", -1.0)
    b = zfit.Parameter("poly_b", 0.0)
    c = zfit.Parameter("poly_c", 1.0)

    poly_pdf = CustomPolynomialPDF(a=a, b=b, c=c, obs=space1)
    yield_param = 25.0
    extended_poly = poly_pdf.create_extended(yield_param)

    # Test that extended polynomial goes negative
    x_negative = np.array([[2.0]])  # Should give negative values
    unclamped_ext_vals = extended_poly.ext_pdf(x_negative, norm=False).numpy()
    assert np.any(unclamped_ext_vals < 0), "Extended polynomial should produce negative values"

    # Create positive version
    epsilon = 1e-12
    positive_ext_poly = extended_poly.to_positive(epsilon=epsilon)

    # Test that negative values are made positive
    positive_ext_vals = positive_ext_poly.ext_pdf(x_negative, norm=False).numpy()
    assert np.all(positive_ext_vals >= epsilon), "Extended PDF should not have values below epsilon"

    # Test at positive region
    x_positive = np.array([[0.0]])
    positive_ext_vals = positive_ext_poly.ext_pdf(x_positive, norm=False).numpy()
    assert np.all(positive_ext_vals > epsilon), "Positive extended values should remain above epsilon"


def test_positive_pdf_extended_error_non_extended():
    """Test that ext_pdf raises error when called on non-extended PDF."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)  # Non-extended
    positive_gauss = gauss.to_positive()

    x_test = np.array([[0.0]])
    with pytest.raises(NotExtendedPDFError):
        positive_gauss.ext_pdf(x_test)


def test_positive_pdf_inheritance_from_wrapped():
    """Test that PositivePDF correctly inherits properties from wrapped PDF."""
    gauss = zfit.pdf.Gauss(2.0, 0.8, obs=space2)
    extended_gauss = gauss.create_extended(75.0)

    # Test with custom epsilon
    positive = extended_gauss.to_positive(epsilon=1e-5)

    # Should inherit obs, norm, and extended status from wrapped PDF
    assert positive.obs == extended_gauss.obs
    assert positive.norm == extended_gauss.norm
    assert positive.is_extended == extended_gauss.is_extended
    assert positive.get_yield().numpy() == 75.0
