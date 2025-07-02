#  Copyright (c) 2024 zfit
import pytest

import zfit
from zfit.util.exception import NormRangeUnderdefinedError

limits1 = (-4, 3)
limits2 = (-2, 5)
limits3 = (-1, 7)
obs1 = "obs1"
obs2 = "obs2"

space1 = zfit.Space(obs=obs1, limits=limits1)
space2 = zfit.Space(obs=obs1, limits=limits2)
space3 = zfit.Space(obs=obs1, limits=limits3)
space4 = zfit.Space(obs=obs2, limits=limits2)

space5 = space1.combine(space4)


def test_norm_range():
    gauss1 = zfit.pdf.Gauss(1.0, 4.0, obs=space1)
    gauss2 = zfit.pdf.Gauss(1.0, 4.0, obs=space1)
    gauss3 = zfit.pdf.Gauss(1.0, 4.0, obs=space2)

    sum1 = zfit.pdf.SumPDF(pdfs=[gauss1, gauss2], fracs=0.4, obs=space1)
    assert sum1.obs == (obs1,)
    assert sum1.norm == space1

    with pytest.raises(NormRangeUnderdefinedError):
        _ = zfit.pdf.SumPDF(pdfs=[gauss1, gauss3], fracs=0.34)




def test_combine_range():
    gauss1 = zfit.pdf.Gauss(1.0, 4.0, obs=space1)
    gauss4 = zfit.pdf.Gauss(1.0, 4.0, obs=space4)
    gauss5 = zfit.pdf.Gauss(1.0, 4.0, obs=space4)

    product = zfit.pdf.ProductPDF(pdfs=[gauss1, gauss4])
    assert product.obs == (obs1, obs2)
    assert product.norm == space5

    product = zfit.pdf.ProductPDF(pdfs=[gauss1, gauss4, gauss5])
    assert product.obs == (obs1, obs2)
    assert product.norm == space5


def test_clamp_pdf_basic():
    """Test basic ClampPDF functionality with a simple Gaussian."""
    import numpy as np
    
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    clamped_gauss = gauss.create_clamped(lower=1e-10)
    
    # Test that the clamped PDF maintains basic properties
    assert clamped_gauss.obs == gauss.obs
    assert clamped_gauss.norm == gauss.norm
    assert clamped_gauss.is_extended == gauss.is_extended
    
    # Test evaluation - for a Gaussian, values should be positive, so clamping should not change much
    x_test = np.array([[-2.0], [0.0], [2.0]])
    original_vals = gauss.pdf(x_test)
    clamped_vals = clamped_gauss.pdf(x_test)
    
    # Values should be very close since Gaussian doesn't produce negative values
    np.testing.assert_allclose(original_vals.numpy(), clamped_vals.numpy(), rtol=1e-10)


def test_clamp_pdf_negative_weights_kde():
    """Test ClampPDF with KDE that has negative weights (the main use case)."""
    import numpy as np
    
    # Create data with negative weights - this was the original problem case
    data_vals = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([1.0, 1.0, -0.1])  # One negative weight
    
    data = zfit.data.Data.from_numpy(obs=space1, array=data_vals, weights=weights)
    kde = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')
    
    # Create a clamped version
    clamped_kde = kde.create_clamped()
    
    # Test that properties are preserved
    assert clamped_kde.obs == kde.obs
    assert clamped_kde.norm == kde.norm
    
    # Test evaluation
    test_x = np.array([[0.0], [1.0], [2.0]])
    clamped_vals = clamped_kde.pdf(test_x).numpy()
    
    # Verify no NaN or overly negative values
    assert not np.any(np.isnan(clamped_vals)), "Clamped PDF should not have NaN values"
    assert np.all(clamped_vals >= 1e-310), "Clamped PDF should not have values below lower bound"


def test_clamp_pdf_integration():
    """Test that ClampPDF integration works correctly."""
    import numpy as np
    
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    clamped_gauss = gauss.create_clamped()
    
    # Test integration - should give similar results for a well-behaved PDF
    integral_original = gauss.integrate(limits=space1)
    integral_clamped = clamped_gauss.integrate(limits=space1)
    
    # Should be very close (both should be ~1)
    np.testing.assert_allclose(integral_original.numpy(), integral_clamped.numpy(), rtol=1e-6)


def test_clamp_pdf_custom_bound():
    """Test ClampPDF with custom lower bound."""
    import numpy as np
    
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    custom_bound = 1e-5
    clamped_gauss = gauss.create_clamped(lower=custom_bound)
    
    # Test that the bound is respected
    assert clamped_gauss.lower == custom_bound
    
    # For this test, we create an artificial case where clamping matters
    # Use a very small test region where Gaussian is essentially zero
    x_test = np.array([[-100.0]])  # Far from the mean
    clamped_vals = clamped_gauss.pdf(x_test).numpy()
    
    # The value should be at least the custom bound
    assert np.all(clamped_vals >= custom_bound)


def test_clamp_pdf_extended():
    """Test ClampPDF with extended PDFs."""
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    extended_gauss = gauss.create_extended(100.0)
    clamped_extended = extended_gauss.create_clamped()
    
    # Test that extension is preserved
    assert clamped_extended.is_extended
    assert clamped_extended.get_yield().numpy() == 100.0


def test_clamp_pdf_upper_bound():
    """Test ClampPDF with upper bound."""
    import numpy as np
    
    # Create a Gaussian centered at 0
    gauss = zfit.pdf.Gauss(0.0, 0.1, obs=space1)  # Narrow Gaussian for high peak
    upper_bound = 1.0
    clamped_gauss = gauss.create_clamped(upper=upper_bound)
    
    # Test that both bounds are set correctly
    assert clamped_gauss.upper == upper_bound
    assert clamped_gauss.lower is None  # No lower bound set
    
    # Evaluate at the peak where the value would be highest
    x_test = np.array([[0.0]])  # At the mean
    clamped_vals = clamped_gauss.pdf(x_test).numpy()
    
    # The value should not exceed the upper bound
    assert np.all(clamped_vals <= upper_bound)


def test_clamp_pdf_both_bounds():
    """Test ClampPDF with both lower and upper bounds."""
    import numpy as np
    
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    lower_bound = 1e-8
    upper_bound = 0.5
    clamped_gauss = gauss.create_clamped(lower=lower_bound, upper=upper_bound)
    
    # Test that both bounds are set correctly
    assert clamped_gauss.lower == lower_bound
    assert clamped_gauss.upper == upper_bound
    
    # Test at various points
    x_test = np.array([[-10.0], [1.0], [10.0]])  # Far left, center, far right
    clamped_vals = clamped_gauss.pdf(x_test).numpy()
    
    # All values should be within bounds
    assert np.all(clamped_vals >= lower_bound)
    assert np.all(clamped_vals <= upper_bound)


# Custom polynomial PDF that can go below zero for testing
class CustomPolynomialPDF(zfit.pdf.BasePDF, zfit.core.SerializableMixin):
    """A custom second-degree polynomial PDF: f(x) = a*x^2 + b*x + c
    
    This PDF is designed to test clamping behavior as it can produce negative values
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


def test_clamp_pdf_optional_lower():
    """Test that lower bound is truly optional."""
    import numpy as np
    
    gauss = zfit.pdf.Gauss(1.0, 0.5, obs=space1)
    
    # Create clamped PDF with no lower bound
    clamped_gauss = gauss.create_clamped()
    
    # Test that lower bound is None
    assert clamped_gauss.lower is None
    assert clamped_gauss.upper is None
    
    # Test with only upper bound
    clamped_gauss_upper = gauss.create_clamped(upper=0.5)
    assert clamped_gauss_upper.lower is None
    assert clamped_gauss_upper.upper == 0.5


def test_clamp_pdf_custom_negative_polynomial():
    """Test ClampPDF with a custom polynomial that goes below zero."""
    import numpy as np
    
    # Create polynomial f(x) = -x^2 + 1, which is positive at x=0 but negative for |x| > 1
    a = zfit.Parameter("a", -1.0)  # coefficient for x^2 term (negative)
    b = zfit.Parameter("b", 0.0)   # coefficient for x term
    c = zfit.Parameter("c", 1.0)   # constant term
    
    poly_pdf = CustomPolynomialPDF(a=a, b=b, c=c, obs=space1)
    
    # Test that the polynomial actually goes negative outside |x| > 1
    x_negative = np.array([[2.0]])  # Should give f(2) = -4 + 1 = -3
    unclamped_vals = poly_pdf.pdf(x_negative, norm=False).numpy()
    assert np.any(unclamped_vals < 0), "Polynomial should produce negative values for x=2"
    
    # Create clamped version with lower bound
    lower_bound = 1e-10
    clamped_poly = poly_pdf.create_clamped(lower=lower_bound)
    
    # Test that the negative values are clamped to the lower bound
    clamped_vals = clamped_poly.pdf(x_negative, norm=False).numpy()
    assert np.all(clamped_vals >= lower_bound), "Clamped PDF should not have values below lower bound"
    
    # Test at a point where the polynomial is positive (x=0, f(0)=1)
    x_positive = np.array([[0.0]])
    positive_vals = clamped_poly.pdf(x_positive, norm=False).numpy()
    assert np.all(positive_vals > lower_bound), "Positive values should remain above lower bound"
    
    # Verify that values at x=0 are approximately 1 (the original polynomial value)
    expected_at_zero = c.numpy()  # c = 1.0
    np.testing.assert_allclose(positive_vals, expected_at_zero, rtol=1e-6)
