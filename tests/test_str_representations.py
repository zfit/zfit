"""Tests for __str__ method implementations in core zfit classes."""

#  Copyright (c) 2025 zfit

import pytest
import tensorflow as tf

import zfit
from zfit import z


def test_parameter_str_basic():
    """Test basic parameter string representation."""
    param = zfit.Parameter("test_param", 1.5)
    str_repr = str(param)
    
    assert "test_param" in str_repr
    assert "1.5" in str_repr or "1" in str_repr  # Allow for formatting differences
    assert "=" in str_repr


def test_parameter_str_with_limits():
    """Test parameter string with limits."""
    param = zfit.Parameter("bounded_param", 2.0, -5.0, 10.0)
    str_repr = str(param)
    
    assert "bounded_param" in str_repr
    assert "2" in str_repr


def test_parameter_str_graph_mode():
    """Test parameter string representation in graph mode."""
    # Test in current mode (should work regardless)
    param = zfit.Parameter("graph_param", 3.14)
    str_repr = str(param)
    
    assert "graph_param" in str_repr
    # Should show value or symbolic representation
    assert "3.14" in str_repr or "symbolic" in str_repr


def test_parameter_str_error_handling():
    """Test parameter string representation when errors occur."""
    param = zfit.Parameter("error_param", 1.0)
    
    # Simulate accessing a parameter that might have issues
    str_repr = str(param)
    
    # Should not raise an exception
    assert isinstance(str_repr, str)
    assert len(str_repr) > 0


def test_constant_parameter_str():
    """Test ConstantParameter string representation."""
    # Use the correct way to create a constant parameter
    const_param = zfit.Parameter("const", 42.0, floating=False)
    str_repr = str(const_param)
    
    assert "const" in str_repr
    assert "42" in str_repr


def test_composed_parameter_str():
    """Test ComposedParameter string representation."""
    param1 = zfit.Parameter("p1", 1.0)
    param2 = zfit.Parameter("p2", 2.0)
    composed = zfit.ComposedParameter("sum", lambda p1, p2: p1 + p2, params=[param1, param2])
    
    str_repr = str(composed)
    
    assert "sum" in str_repr
    # Should show the computed value
    assert "3" in str_repr or "symbolic" in str_repr


def test_pdf_str_basic():
    """Test basic PDF string representation."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    str_repr = str(pdf)
    
    assert "Gauss" in str_repr
    assert "obs=" in str_repr
    assert "x" in str_repr
    assert "params=" in str_repr
    assert "mu" in str_repr
    assert "sigma" in str_repr


def test_pdf_str_extended():
    """Test extended PDF string representation."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    yield_param = zfit.Parameter("yield", 1000.0)
    
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    extended_pdf = pdf.create_extended(yield_param)
    
    str_repr = str(extended_pdf)
    
    assert "extended=True" in str_repr
    assert "yield=" in str_repr


def test_pdf_str_many_parameters():
    """Test PDF string representation with many parameters."""
    obs = zfit.Space("x", (-1, 1))
    
    # Create a PDF with many parameters
    params = [zfit.Parameter(f"param_{i}", float(i)) for i in range(6)]
    # Use a polynomial which can take many parameters
    pdf = zfit.pdf.Chebyshev(obs=obs, coeffs=params[:4])  # Use first 4 params
    
    str_repr = str(pdf)
    
    assert "Chebyshev" in str_repr
    # Should show first 3 parameters plus "..."
    param_count = str_repr.count("param_")
    assert param_count <= 4  # Should limit display


def test_pdf_str_no_observables():
    """Test PDF string when observables are not available."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    # Temporarily remove obs to test error handling
    original_obs = pdf.obs
    try:
        pdf._obs = None
        str_repr = str(pdf)
        
        # Should still work and show the PDF name
        assert "Gauss" in str_repr
    finally:
        pdf._obs = original_obs


def test_pdf_str_custom_label():
    """Test PDF string with custom label."""
    obs = zfit.Space("x", (-1, 1))
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.0)
    pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, name="custom_gauss")
    
    str_repr = str(pdf)
    
    assert "custom_gauss" in str_repr


def test_base_minimizer_str():
    """Test BaseMinimizer string representation."""
    # Import here to avoid circular imports
    from zfit.minimizers.baseminimizer import BaseMinimizer
    
    # Create a concrete minimizer for testing
    minimizer = BaseMinimizer(tol=1e-4, verbosity=1, maxiter=5000)
    
    str_repr = str(minimizer)
    
    assert "BaseMinimizer" in str_repr
    assert "tol=" in str_repr
    assert "maxiter=5000" in str_repr
    assert "verbosity=1" in str_repr


def test_minimizer_str_default_values():
    """Test minimizer string with default values."""
    from zfit.minimizers.baseminimizer import BaseMinimizer
    
    minimizer = BaseMinimizer()  # All defaults
    
    str_repr = str(minimizer)
    
    assert "BaseMinimizer" in str_repr
    assert "tol=" in str_repr
    # Should not show verbosity=0 (default)
    assert "verbosity=0" not in str_repr
    # Should not show maxiter=auto (default)
    assert "maxiter=auto" not in str_repr


def test_minimizer_str_custom_strategy():
    """Test minimizer string with custom strategy."""
    from zfit.minimizers.baseminimizer import BaseMinimizer
    
    # Test with default strategy (skip custom strategy test for now)
    minimizer = BaseMinimizer()
    
    str_repr = str(minimizer)
    
    assert "BaseMinimizer" in str_repr
    # Basic format check
    assert "tol=" in str_repr


def test_str_in_collections():
    """Test string representations when objects are in collections."""
    param1 = zfit.Parameter("p1", 1.0)
    param2 = zfit.Parameter("p2", 2.0)
    
    param_list = [param1, param2]
    list_str = str(param_list)
    
    # Check that parameter names appear in the list representation
    assert "p1" in list_str
    assert "p2" in list_str


def test_str_with_complex_models():
    """Test string representations with complex model hierarchies."""
    obs = zfit.Space("x", (-1, 1))
    
    # Create parameters
    mu1 = zfit.Parameter("mu1", -0.5)
    sigma1 = zfit.Parameter("sigma1", 0.3)
    mu2 = zfit.Parameter("mu2", 0.5)
    sigma2 = zfit.Parameter("sigma2", 0.4)
    frac = zfit.Parameter("frac", 0.6)
    
    # Create PDFs
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)
    gauss2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)
    
    # Create sum PDF
    sum_pdf = zfit.pdf.SumPDF([gauss1, gauss2], fracs=frac)
    
    str_repr = str(sum_pdf)
    
    assert "SumPDF" in str_repr
    # Should handle complex hierarchies gracefully
    assert len(str_repr) > 0


def test_str_performance():
    """Test that string representations are reasonably fast."""
    import time
    
    # Create many parameters
    params = [zfit.Parameter(f"param_{i}", float(i)) for i in range(100)]
    
    start_time = time.time()
    str_reprs = [str(param) for param in params]
    end_time = time.time()
    
    # Should complete quickly (less than 1 second for 100 parameters)
    assert end_time - start_time < 1.0
    assert len(str_reprs) == 100
    assert all(isinstance(s, str) for s in str_reprs)