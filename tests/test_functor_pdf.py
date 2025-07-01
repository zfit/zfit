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
    clamped_gauss = gauss.create_clamped(lower_bound=1e-10)
    
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
    clamped_gauss = gauss.create_clamped(lower_bound=custom_bound)
    
    # Test that the bound is respected
    assert clamped_gauss.lower_bound == custom_bound
    
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
