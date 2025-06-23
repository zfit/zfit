"""Tests for parameter validation utilities."""

#  Copyright (c) 2025 zfit

import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit import Parameter
from zfit.param import ComposedParameter, ConstantParameter
from zfit.util.exception import ShapeIncompatibleError
from zfit.util.param_validation import (
    calculate_limit_tolerance,
    check_at_limit,
    format_parameter_error,
    validate_limit_bounds,
    validate_parameter_names,
    validate_parameter_shapes,
    validate_parameter_type,
    validate_parameters,
    validate_stepsize,
    validate_value_in_limits,
)


def test_valid_parameter():
    """Test validation passes for valid parameter."""
    param = Parameter("test", 1.0)
    # Should not raise
    validate_parameter_type(param)
    validate_parameter_type(param, "custom_name")
    

def test_invalid_parameter_types():
    """Test validation fails for invalid types."""
    invalid_inputs = [1.0, "string", [1, 2, 3], None, tf.constant(1.0)]
    
    for invalid_input in invalid_inputs:
        with pytest.raises(TypeError, match="must be ZfitParameter"):
            validate_parameter_type(invalid_input)
            

def test_custom_parameter_name_in_error():
    """Test custom parameter name appears in error message."""
    with pytest.raises(TypeError, match="custom_param.*must be ZfitParameter"):
        validate_parameter_type(1.0, "custom_param")


def test_multiple_valid_parameters():
    """Test validation passes for multiple valid parameters."""
    param1 = Parameter("param1", 1.0)
    param2 = Parameter("param2", 2.0)
    param3 = ConstantParameter("param3", 3.0)
    
    # Should not raise
    validate_parameters(param1, param2, param3)
    

def test_multiple_parameters_with_names():
    """Test validation with custom names."""
    param1 = Parameter("param1", 1.0)
    param2 = Parameter("param2", 2.0)
    
    # Should not raise
    validate_parameters(param1, param2, names=["first", "second"])
    

def test_invalid_parameter_in_list():
    """Test validation fails when one parameter is invalid."""
    param1 = Parameter("param1", 1.0)
    invalid_param = 42
    param3 = Parameter("param3", 3.0)
    
    with pytest.raises(TypeError, match="param2.*must be ZfitParameter"):
        validate_parameters(param1, invalid_param, param3)
        

def test_custom_names_in_error():
    """Test custom names appear in error messages."""
    param1 = Parameter("param1", 1.0)
    invalid_param = "invalid"
    
    with pytest.raises(TypeError, match="custom_name.*must be ZfitParameter"):
        validate_parameters(param1, invalid_param, names=["first", "custom_name"])


def test_valid_bounds():
    """Test validation passes for valid bounds."""
    lower, upper = validate_limit_bounds(0.0, 1.0)
    assert lower == 0.0
    assert upper == 1.0
    

def test_none_bounds():
    """Test validation passes when bounds are None."""
    lower, upper = validate_limit_bounds(None, 1.0)
    assert lower is None
    assert upper == 1.0
    
    lower, upper = validate_limit_bounds(0.0, None)
    assert lower == 0.0
    assert upper is None
    

def test_invalid_bounds_order():
    """Test validation fails when lower >= upper."""
    with pytest.raises(ValueError, match="Lower limit.*must be less than upper limit"):
        validate_limit_bounds(2.0, 1.0)
        
    with pytest.raises(ValueError, match="Lower limit.*must be less than upper limit"):
        validate_limit_bounds(1.0, 1.0)


def test_value_within_limits():
    """Test validation passes for value within limits."""
    # Should not raise
    validate_value_in_limits(0.5, lower=0.0, upper=1.0)
    validate_value_in_limits(0.0, lower=0.0, upper=1.0)  # At boundary
    validate_value_in_limits(1.0, lower=0.0, upper=1.0)  # At boundary
    

def test_value_below_lower_limit():
    """Test validation fails for value below lower limit."""
    with pytest.raises(ValueError, match="below lower limit"):
        validate_value_in_limits(-0.5, lower=0.0, upper=1.0)
        

def test_value_above_upper_limit():
    """Test validation fails for value above upper limit."""
    with pytest.raises(ValueError, match="above upper limit"):
        validate_value_in_limits(1.5, lower=0.0, upper=1.0)
        

def test_only_lower_limit():
    """Test validation with only lower limit."""
    # Should not raise
    validate_value_in_limits(5.0, lower=0.0)
    
    with pytest.raises(ValueError, match="below lower limit"):
        validate_value_in_limits(-1.0, lower=0.0)
        

def test_only_upper_limit():
    """Test validation with only upper limit."""
    # Should not raise
    validate_value_in_limits(-5.0, upper=0.0)
    
    with pytest.raises(ValueError, match="above upper limit"):
        validate_value_in_limits(1.0, upper=0.0)


def test_tolerance_with_bounds():
    """Test tolerance calculation with both bounds."""
    tol = calculate_limit_tolerance(0.0, 1.0, exact=False)
    assert tol > 0
    
    tol_exact = calculate_limit_tolerance(0.0, 1.0, exact=True)
    assert tol_exact > 0
    assert tol_exact < tol  # Exact should be stricter
    

def test_tolerance_without_bounds():
    """Test tolerance calculation without bounds."""
    tol = calculate_limit_tolerance(None, None, exact=False)
    assert tol == 1e-5
    
    tol_exact = calculate_limit_tolerance(None, None, exact=True)
    assert tol_exact == 1e-7
    

def test_tolerance_one_bound_none():
    """Test tolerance with one bound None."""
    tol = calculate_limit_tolerance(None, 1.0, exact=False)
    assert tol == 1e-5
    
    tol = calculate_limit_tolerance(0.0, None, exact=False)
    assert tol == 1e-5


def test_at_lower_limit():
    """Test detection of value at lower limit."""
    assert check_at_limit(0.0, lower=0.0, upper=1.0)
    assert check_at_limit(1e-6, lower=0.0, upper=1.0)  # Within tolerance
    

def test_at_upper_limit():
    """Test detection of value at upper limit."""
    assert check_at_limit(1.0, lower=0.0, upper=1.0)
    assert check_at_limit(1.0 - 1e-6, lower=0.0, upper=1.0)  # Within tolerance
    

def test_not_at_limit():
    """Test detection when value is not at limit."""
    assert not check_at_limit(0.5, lower=0.0, upper=1.0)
    assert not check_at_limit(0.1, lower=0.0, upper=1.0)
    assert not check_at_limit(0.9, lower=0.0, upper=1.0)
    

def test_no_limits():
    """Test with no limits."""
    assert not check_at_limit(42.0)
    assert not check_at_limit(42.0, lower=None, upper=None)


def test_compatible_shapes():
    """Test validation passes for compatible shapes."""
    param1 = Parameter("param1", 1.0)  # Scalar
    param2 = Parameter("param2", 2.0)  # Scalar
    
    # Should not raise
    validate_parameter_shapes(param1, param2)
    

def test_single_parameter():
    """Test validation passes for single parameter."""
    param = Parameter("param", 1.0)
    
    # Should not raise
    validate_parameter_shapes(param)
    

def test_no_parameters():
    """Test validation passes for no parameters."""
    # Should not raise
    validate_parameter_shapes()


def test_valid_names():
    """Test validation passes for valid names."""
    valid_names = ["param1", "my_parameter", "test_param_2", "_private", "CamelCase"]
    
    # Should not raise
    validate_parameter_names(*valid_names)
    

def test_invalid_name_types():
    """Test validation fails for non-string names."""
    with pytest.raises(ValueError, match="must be string"):
        validate_parameter_names(123)
        
    with pytest.raises(ValueError, match="must be string"):
        validate_parameter_names("valid", None)
        

def test_empty_name():
    """Test validation fails for empty name."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_parameter_names("")
        

def test_python_keywords():
    """Test validation fails for Python keywords."""
    keywords = ["class", "def", "if", "else", "for", "while", "import"]
    
    for keyword in keywords:
        with pytest.raises(ValueError, match="Python keyword"):
            validate_parameter_names(keyword)
            

def test_invalid_identifiers():
    """Test validation fails for invalid identifiers."""
    invalid_names = ["123param", "param-with-dash", "param with space", "param.with.dot"]
    
    for name in invalid_names:
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            validate_parameter_names(name)
            

def test_reserved_patterns():
    """Test validation fails for reserved patterns."""
    reserved_names = ["__special__", "_zfit_internal", "zfit_reserved"]
    
    for name in reserved_names:
        with pytest.raises(ValueError, match="reserved pattern"):
            validate_parameter_names(name)


def test_error_message_format():
    """Test error message formatting."""
    msg = format_parameter_error("ZfitParameter", int, "test_param")
    assert "test_param" in msg
    assert "ZfitParameter" in msg
    assert "int" in msg
    

def test_default_parameter_name():
    """Test default parameter name in error message."""
    msg = format_parameter_error("ZfitParameter", float)
    assert "parameter" in msg
    assert "ZfitParameter" in msg
    assert "float" in msg


def test_valid_stepsize_values():
    """Test validation passes for valid stepsize values."""
    valid_values = [0.1, 1.0, 0.001, tf.constant(0.1), np.array(0.1)]
    
    for value in valid_values:
        # Should not raise
        validate_stepsize(value)
        

def test_none_stepsize():
    """Test validation passes for None stepsize."""
    # Should not raise
    validate_stepsize(None)
    

def test_invalid_stepsize_types():
    """Test validation fails for invalid types."""
    invalid_types = ["string", [1, 2, 3], {"key": "value"}]
    
    for invalid_type in invalid_types:
        with pytest.raises(ValueError, match="must be numeric"):
            validate_stepsize(invalid_type)
            

def test_non_positive_stepsize():
    """Test validation fails for non-positive values."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_stepsize(0.0)
        
    with pytest.raises(ValueError, match="must be positive"):
        validate_stepsize(-0.1)
        

def test_custom_parameter_name_in_error():
    """Test custom parameter name in error message."""
    with pytest.raises(ValueError, match="custom_param"):
        validate_stepsize(-0.1, "custom_param")


if __name__ == "__main__":
    pytest.main([__file__])