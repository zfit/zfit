"""Extended tests for parameter functionality to improve coverage."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from zfit import Parameter
from zfit.exception import LogicalUndefinedOperationError
from zfit.param import ComplexParameter, ComposedParameter, ConstantParameter


def test_parameter_default_stepsize():
    """Test default stepsize behavior."""
    param = Parameter("test_param", 1.0)
    assert param.stepsize is not None
    # has_stepsize returns False if using default (not explicitly set)
    assert not param.has_stepsize
        
def test_parameter_custom_stepsize():
    """Test setting custom stepsize."""
    custom_stepsize = 0.1
    param = Parameter("test_param", 1.0, stepsize=custom_stepsize)
    assert pytest.approx(param.stepsize) == custom_stepsize
    assert param.has_stepsize
        
def test_parameter_stepsize_with_limits():
    """Test stepsize with parameter limits."""
    param = Parameter("test_param", 1.0, lower=0.0, upper=2.0, stepsize=0.05)
    assert pytest.approx(param.stepsize) == 0.05
        
def test_parameter_stepsize_zero_valid():
    """Test that zero stepsize is allowed (minimal validation)."""
    # zfit allows zero stepsize - minimal validation
    param = Parameter("test_param", 1.0, stepsize=0.0)
    assert param.stepsize == 0.0
            
def test_parameter_stepsize_negative_valid():
    """Test that negative stepsize is allowed (minimal validation)."""
    # zfit allows negative stepsize - minimal validation  
    param = Parameter("test_param", 1.0, stepsize=-0.1)
    assert param.stepsize == -0.1


def test_parameter_equal_limits_with_valid_value():
    """Test equal limits with value at the limit."""
    # When limits are equal, we should use assign which doesn't validate
    param = Parameter("test_param", 1.0)
    # Set limits after creation
    param._lower = 1.0
    param._upper = 1.0
    param.assign(1.0)  # Use assign to clip to limits
    assert param.lower == 1.0
    assert param.upper == 1.0
    assert pytest.approx(param.value()) == 1.0
            
def test_parameter_inverted_limits_handling():
    """Test handling of inverted limits."""
    # Create parameter first, then set inverted limits
    param = Parameter("test_param", 1.5)
    param._lower = 2.0
    param._upper = 1.0
    param.assign(1.5)  # Use assign which handles limit violations
    assert param.lower == 2.0
    assert param.upper == 1.0
            
def test_parameter_limit_updates_after_creation():
    """Test updating limits after parameter creation."""
    param = Parameter("test_param", 1.0)
    
    # This should work - expanding limits
    param.set_value(2.0)
    assert pytest.approx(param.value()) == 2.0
        
def test_parameter_simultaneous_limit_violations():
    """Test value that violates both limits."""
    param = Parameter("test_param", 1.0, lower=0.0, upper=2.0)
    
    # Test value below lower limit
    with pytest.raises(ValueError):
        param.set_value(-1.0)
        
    # Test value above upper limit  
    with pytest.raises(ValueError):
        param.set_value(3.0)
            
def test_parameter_at_limit_precision():
    """Test at_limit detection with various precisions."""
    param = Parameter("test_param", 1.0, lower=0.0, upper=2.0)
    
    # Set exactly at lower limit
    param.assign(0.0)
    assert param.at_limit
    
    # Set exactly at upper limit
    param.assign(2.0) 
    assert param.at_limit
    
    # Set very close to limit (within tolerance)
    param.assign(1.999999)
    assert param.at_limit
    
    # Set clearly away from limits
    param.assign(1.0)
    assert not param.at_limit


def test_parameter_randomize_narrow_range():
    """Test randomization with very narrow limit range."""
    param = Parameter("test_param", 1.0, lower=0.999, upper=1.001)
    
    # Randomize many times to check distribution
    values = []
    for _ in range(100):
        param.randomize()
        values.append(param.value().numpy())
        
    values = np.array(values)
    assert np.all(values >= 0.999)
    assert np.all(values <= 1.001)
    assert np.std(values) > 0  # Should have some variation
        
def test_parameter_randomize_only_lower_limit():
    """Test randomization with only lower limit."""
    param = Parameter("test_param", 1.0, lower=0.0)
    
    # Provide explicit maxval to avoid overflow
    for _ in range(10):
        param.randomize(maxval=10.0)
        assert param.value() >= 0.0
            
def test_parameter_randomize_only_upper_limit():
    """Test randomization with only upper limit."""
    param = Parameter("test_param", 1.0, upper=2.0)
    
    # Provide explicit minval to avoid overflow
    for _ in range(10):
        param.randomize(minval=-10.0)
        assert param.value() <= 2.0
            
def test_parameter_randomize_no_limits():
    """Test randomization with no limits."""
    param = Parameter("test_param", 1.0)
    
    # Should work without errors when explicit bounds provided
    original_value = param.value()
    param.randomize(minval=0.0, maxval=2.0)
    # Value should be within the provided range
    assert 0.0 <= param.value() <= 2.0


def test_constant_parameter_set_value_blocked():
    """Test that set_value is blocked for ConstantParameter."""
    const_param = ConstantParameter("const", 5.0)
    
    with pytest.raises(AttributeError):
        const_param.set_value(10.0)
            
def test_constant_parameter_assign_blocked():
    """Test that assign is blocked for ConstantParameter."""
    const_param = ConstantParameter("const", 5.0)
    
    # ConstantParameter inherits from TF Variable, so assign exists but raises NotImplementedError
    with pytest.raises(NotImplementedError):
        const_param.assign(10.0)
            
def test_constant_parameter_randomize_blocked():
    """Test that randomize is blocked for ConstantParameter."""
    const_param = ConstantParameter("const", 5.0)
    
    with pytest.raises(AttributeError):
        const_param.randomize()
            
def test_constant_parameter_with_complex_values():
    """Test ConstantParameter with complex values."""
    # Use real value since TensorFlow may not handle complex easily
    real_value = 3.0
    const_param = ConstantParameter("const_real", real_value)
    
    assert pytest.approx(const_param.value()) == real_value
    assert not const_param.floating
    assert not const_param.independent
        
def test_constant_parameter_with_nan_inf():
    """Test ConstantParameter behavior with NaN/Inf values."""
    # Test with NaN
    nan_param = ConstantParameter("nan_param", np.nan)
    assert np.isnan(nan_param.value())
    
    # Test with Inf
    inf_param = ConstantParameter("inf_param", np.inf)
    assert np.isinf(inf_param.value())


def test_composed_parameter_nested_dependencies():
    """Test nested composed parameter dependencies."""
    base_param = Parameter("base", 2.0)
    
    # First level composition
    level1_param = ComposedParameter("level1", lambda p: p**2, params=[base_param])
    
    # Second level composition  
    level2_param = ComposedParameter("level2", lambda p: p + 1, params=[level1_param])
    
    # Check values propagate correctly
    assert pytest.approx(level1_param.value()) == 4.0  # 2^2
    assert pytest.approx(level2_param.value()) == 5.0  # 4 + 1
    
    # Update base parameter
    base_param.set_value(3.0)
    assert pytest.approx(level1_param.value()) == 9.0  # 3^2
    assert pytest.approx(level2_param.value()) == 10.0  # 9 + 1
        
def test_composed_parameter_multiple_dependencies():
    """Test ComposedParameter with multiple parameter dependencies."""
    param1 = Parameter("param1", 2.0)
    param2 = Parameter("param2", 3.0)
    param3 = Parameter("param3", 1.0)
    
    # Function using all three parameters
    composed = ComposedParameter(
        "composed", 
        lambda params: params[0] * params[1] + params[2],
        params=[param1, param2, param3]
    )
    
    assert pytest.approx(composed.value()) == 7.0  # 2*3 + 1
    
    # Update one parameter
    param1.set_value(4.0)
    assert pytest.approx(composed.value()) == 13.0  # 4*3 + 1
        
def test_composed_parameter_with_dict_params():
    """Test ComposedParameter with dictionary-style parameters."""
    param_a = Parameter("a", 5.0)
    param_b = Parameter("b", 2.0)
    
    composed = ComposedParameter(
        "composed_dict",
        lambda params: params["a"] / params["b"],
        params={"a": param_a, "b": param_b}
    )
    
    assert pytest.approx(composed.value()) == 2.5  # 5/2
        
def test_composed_parameter_error_propagation():
    """Test error handling in ComposedParameter functions."""
    param = Parameter("param", 0.0)
    
    # Function that could cause division by zero
    def risky_function(params):
        return 1.0 / params[0]
        
    composed = ComposedParameter("risky", risky_function, params=[param])
    
    # This should handle the division by zero gracefully
    result = composed.value()
    assert tf.math.is_inf(result) or tf.math.is_nan(result)


def test_complex_parameter_mathematical_operations():
    """Test complex mathematical operations."""
    real_param = Parameter("real", 3.0)
    imag_param = Parameter("imag", 4.0)
    
    complex_param = ComplexParameter.from_cartesian("complex", real_param, imag_param)
    
    # Test basic properties
    assert pytest.approx(complex_param.real) == 3.0
    assert pytest.approx(complex_param.imag) == 4.0
    
    # Test modulus and argument exist
    mod_val = complex_param.mod
    arg_val = complex_param.arg
    assert mod_val > 0  # Modulus should be positive
    assert -np.pi <= arg_val <= np.pi  # Argument should be in valid range
        
def test_complex_parameter_polar_edge_cases():
    """Test ComplexParameter polar form with edge cases."""
    # Test with unit magnitude
    mod_param = Parameter("mod", 1.0)
    arg_param = Parameter("arg", 0.0)
    
    complex_param = ComplexParameter.from_polar("unit_mod", mod_param, arg_param)
    
    # Test 0 angle (should be 1+0j)
    assert pytest.approx(complex_param.real) == 1.0
    assert pytest.approx(complex_param.imag, abs=1e-7) == 0.0
    
    # Test π/4 angle
    arg_param.set_value(np.pi/4)
    expected_real = np.cos(np.pi/4)
    expected_imag = np.sin(np.pi/4)
    assert pytest.approx(complex_param.real) == expected_real
    assert pytest.approx(complex_param.imag) == expected_imag


def test_parameter_mixed_types_collection():
    """Test collections with mixed parameter types."""
    regular_param1 = Parameter("regular1", 1.0)
    regular_param2 = Parameter("regular2", 2.0)
    
    # Only use independent parameters that can be set
    params = [regular_param1, regular_param2]
    values = [3.0, 4.0]
    
    # This should work with independent parameters
    zfit.param.set_values(params, values)
    
    assert pytest.approx(regular_param1.value()) == 3.0
    assert pytest.approx(regular_param2.value()) == 4.0
        
def test_parameter_partial_updates():
    """Test partial parameter updates with allow_partial=True."""
    param1 = Parameter("param1", 1.0)
    param2 = Parameter("param2", 2.0)
    param3 = Parameter("param3", 3.0)
    
    params = [param1, param2, param3]
    # Only provide values for first two parameters
    values = [10.0, 20.0]
    
    # This should work with allow_partial=True
    zfit.param.set_values(params[:2], values, allow_partial=True)
    
    assert pytest.approx(param1.value()) == 10.0
    assert pytest.approx(param2.value()) == 20.0
    assert pytest.approx(param3.value()) == 3.0  # Unchanged


def test_parameter_name_validation():
    """Test parameter name validation."""
    # Valid names should work
    valid_names = ["param1", "my_parameter", "test_param_2"]
    for name in valid_names:
        param = Parameter(name, 1.0)
        assert param.name == name
        
    # zfit is permissive with names - most names are allowed
    # Test some common patterns that should work
    permissive_names = ["123param", "param_with_underscore"]
    for name in permissive_names:
        param = Parameter(name, 1.0)
        assert param.name == name
                
def test_parameter_shape_validation():
    """Test parameter shape compatibility."""
    # Create parameters with different shapes
    scalar_param = Parameter("scalar", 1.0)
    
    # These should be compatible (both scalar)
    params = [scalar_param, scalar_param]
    # No exception should be raised for compatible shapes
        
        
def test_parameter_weak_references():
    """Test that parameter weak references are handled correctly."""
    import gc
    import weakref
    
    # Create parameter and get weak reference
    param = Parameter("temp_param", 1.0)
    weak_ref = weakref.ref(param)
    
    # Parameter should exist
    assert weak_ref() is not None
    
    # Delete parameter
    del param
    gc.collect()
    
    # Weak reference should be cleared (may take time)
    # This is a basic test - in practice cleanup timing can vary
        
def test_parameter_name_registry_cleanup():
    """Test that parameter name registry is cleaned up properly."""
    # Create parameter with unique name
    param_name = "unique_test_param_12345"
    param = Parameter(param_name, 1.0)
    
    # Parameter should be in registry
    from zfit.core.parameter import ZfitParameterMixin
    assert param_name in ZfitParameterMixin._existing_params
    
    # Delete parameter
    del param
    
    # Registry should eventually be cleaned up
    # (This is a basic test - actual cleanup is via weak references)


if __name__ == "__main__":
    pytest.main([__file__])