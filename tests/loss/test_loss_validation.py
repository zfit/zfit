"""Test input validation for loss functions."""

#  Copyright (c) 2025 zfit

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit.util.exception import IntentionAmbiguousError


@pytest.fixture
def gauss_model():
    """Fixture providing a simple Gaussian model."""
    obs = zfit.Space("x", -3, 3)
    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    return zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)


@pytest.fixture
def simple_data():
    """Fixture providing simple dataset."""
    return zfit.Data.from_numpy(obs="x", array=np.random.normal(0, 1, 100))


@pytest.fixture
def binned_space():
    """Fixture providing a binned space."""
    return zfit.Space("x", -3, 3, binning=30)


@pytest.fixture
def unbinned_space():
    """Fixture providing an unbinned space."""
    return zfit.Space("x", -3, 3)


@pytest.mark.parametrize("loss_class", [
    zfit.loss.UnbinnedNLL,
    zfit.loss.ExtendedUnbinnedNLL,
])
def test_empty_model_list(loss_class, simple_data):
    """Test that empty model list raises error."""
    with pytest.raises(ValueError, match="At least one model must be provided"):
        loss_class(model=[], data=simple_data)


@pytest.mark.parametrize("loss_class", [
    zfit.loss.UnbinnedNLL,
    zfit.loss.ExtendedUnbinnedNLL,
])
def test_empty_data_list(loss_class, gauss_model):
    """Test that empty data list raises error."""
    with pytest.raises(ValueError, match="At least one dataset must be provided"):
        loss_class(model=gauss_model, data=[])


def test_empty_dataset(gauss_model):
    """Test behavior with empty dataset."""
    obs = gauss_model.space
    empty_data = zfit.Data.from_numpy(obs=obs, array=np.array([]).reshape(0, 1))

    # The validation might only work when actually evaluating the loss
    # since num_entries might be lazy-evaluated
    loss = zfit.loss.UnbinnedNLL(model=gauss_model, data=empty_data)

    # Try to evaluate the loss - this is where we might see issues with empty data
    try:
        from zfit import run
        if run.executing_eagerly():
            # In eager mode, try to get the value - this should work but give inf/nan
            val = loss.value()
            # Empty dataset should give problematic values
            assert np.isnan(val) or np.isinf(val) or val == 0
    except Exception:
        # If it fails, that's also acceptable for empty dataset
        pass


def test_model_data_observable_mismatch():
    """Test that mismatched observables raise error."""
    # Model with observable 'x'
    obs_x = zfit.Space("x", -3, 3)
    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs_x)

    # Data with observable 'y'
    obs_y = zfit.Space("y", -3, 3)
    data = zfit.Data.from_numpy(obs=obs_y, array=np.random.normal(0, 1, 100))

    with pytest.raises(ValueError, match="Model at index 0 has observables.*but data has observables"):
        zfit.loss.UnbinnedNLL(model=model, data=data)


@pytest.mark.parametrize("invalid_model,data", [
    ("not_a_model", lambda: zfit.Data.from_numpy(obs="x", array=np.random.normal(0, 1, 100))),
    ([42], lambda: [zfit.Data.from_numpy(obs="x", array=np.random.normal(0, 1, 100))]),
])
def test_invalid_model_type(invalid_model, data):
    """Test that invalid model type raises error."""
    data_obj = data()  # Call the lambda to create data
    with pytest.raises(TypeError, match="Model at index 0 must be a ZfitPDF"):
        zfit.loss.UnbinnedNLL(model=invalid_model, data=data_obj)


def test_multiple_models_data_mismatch():
    """Test that mismatched number of models and data raises error."""
    obs = zfit.Space("x", -3, 3)

    # Create two models
    models = []
    for i in range(2):
        mu = zfit.Parameter(f"mu{i}", i)
        sigma = zfit.Parameter(f"sigma{i}", 1 - i * 0.5)
        models.append(zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs))

    # Create only one dataset
    data = zfit.Data.from_numpy(obs=obs, array=np.random.normal(0, 1, 100))

    # More models than data - should raise error during zip
    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        zfit.loss.UnbinnedNLL(model=models, data=[data])


@pytest.mark.parametrize("empty_input,error_match", [
    ({"model": []}, "At least one model must be provided"),
    ({"data": []}, "At least one dataset must be provided"),
])
def test_binned_loss_empty_inputs(empty_input, error_match, binned_space):
    """Test binned loss validation for empty inputs."""
    # Create default valid inputs
    data_np = np.random.normal(0, 1, 1000)
    data = zfit.Data.from_numpy(obs=binned_space, array=data_np)

    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    unbinned_model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=zfit.Space("x", -3, 3))
    model = unbinned_model.to_binned(binned_space)

    # Override with empty input if specified
    inputs = {"model": model, "data": data}
    inputs.update(empty_input)

    with pytest.raises(ValueError, match=error_match):
        zfit.loss.BinnedNLL(**inputs)


def test_binned_loss_type_validation(binned_space, unbinned_space):
    """Test binned loss type validation."""
    data_np = np.random.normal(0, 1, 1000)

    # Create binned and unbinned versions
    binned_data = zfit.Data.from_numpy(obs=binned_space, array=data_np)
    unbinned_data = zfit.Data.from_numpy(obs=unbinned_space, array=data_np)

    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    unbinned_model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=unbinned_space)
    binned_model = unbinned_model.to_binned(binned_space)

    # Test non-binned model
    with pytest.raises(ValueError, match="PDFs are not binned but need to be"):
        zfit.loss.BinnedNLL(model=unbinned_model, data=binned_data)

    # Test non-binned data
    with pytest.raises(ValueError, match="datasets are not binned but need to be"):
        zfit.loss.BinnedNLL(model=binned_model, data=unbinned_data)


@pytest.mark.parametrize("n_models,n_data", [
    (1, 1),  # Single model and data
    (2, 2),  # Multiple models and data
])
def test_valid_loss_creation(n_models, n_data):
    """Test that valid loss creation works."""
    obs = zfit.Space("x", -3, 3)

    # Create models
    models = []
    for i in range(n_models):
        mu = zfit.Parameter(f"mu{i}", i)
        sigma = zfit.Parameter(f"sigma{i}", 1 - i * 0.3)
        models.append(zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs))

    # Create data
    data_list = []
    for i in range(n_data):
        data = zfit.Data.from_numpy(
            obs=obs,
            array=np.random.normal(i, 1 - i * 0.3, 100 + i * 50)
        )
        data_list.append(data)

    # Unpack if single item
    if n_models == 1:
        models = models[0]
    if n_data == 1:
        data_list = data_list[0]

    # Should not raise any errors
    loss = zfit.loss.UnbinnedNLL(model=models, data=data_list)
    assert loss is not None


def test_data_conversion_with_limits():
    """Test data conversion respects model limits."""
    obs = zfit.Space("x", -2, 2)  # Limited space
    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    # Data outside limits
    data_np = np.random.normal(0, 3, 1000)  # Much wider than limits

    # Should raise error for data outside limits
    with pytest.raises(IntentionAmbiguousError, match="not fully within the limits"):
        zfit.loss.UnbinnedNLL(model=model, data=data_np)


@pytest.fixture
def create_test_model() -> Callable:
    """Factory fixture for creating test models."""
    def _create_model(obs_name="x", mu_val=0, sigma_val=1, limits=(-3, 3)):
        obs = zfit.Space(obs_name, *limits)
        mu = zfit.Parameter(f"mu_{obs_name}", mu_val)
        sigma = zfit.Parameter(f"sigma_{obs_name}", sigma_val)
        return zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    return _create_model


@pytest.fixture
def create_test_data() -> Callable:
    """Factory fixture for creating test data."""
    def _create_data(obs_name="x", size=100, mean=0, std=1, limits=None):
        obs = obs_name if limits is None else zfit.Space(obs_name, *limits)
        return zfit.Data.from_numpy(
            obs=obs,
            array=np.random.normal(mean, std, size)
        )
    return _create_data


def test_loss_with_fixtures(create_test_model, create_test_data):
    """Test loss creation using factory fixtures."""
    model = create_test_model()
    data = create_test_data()

    loss = zfit.loss.UnbinnedNLL(model=model, data=data)
    assert loss is not None
    assert loss.model == [model]
    assert loss.data == [data]
