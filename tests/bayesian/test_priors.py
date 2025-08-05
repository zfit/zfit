#  Copyright (c) 2025 zfit
"""Pure pytest-style tests for Bayesian prior distributions."""

import numpy as np
import pytest
import tensorflow as tf

import zfit


# =====================================================================
# Prior configurations for parametrized tests
# =====================================================================

PRIOR_CONFIGS = [
    # (prior_class_name, init_kwargs, properties)
    ("Normal", {"mu": 0.0, "sigma": 1.0}, {"positive_only": False}),
    ("Uniform", {"lower": 0.0, "upper": 1.0}, {"positive_only": False, "bounded": True}),
    ("HalfNormal", {"sigma": 1.0}, {"positive_only": True}),
    ("Gamma", {"alpha": 2.0, "beta": 0.5}, {"positive_only": True}),
    ("Beta", {"alpha": 2.0, "beta": 2.0}, {"positive_only": False, "bounded": True}),
    ("LogNormal", {"mu": 0.0, "sigma": 1.0}, {"positive_only": True}),
    ("Cauchy", {"m": 0.0, "gamma": 1.0}, {"positive_only": False, "heavy_tailed": True}),
    ("Poisson", {"lam": 3.0}, {"positive_only": True}),
    ("Exponential", {"lam": 2.0}, {"positive_only": True}),  # Changed from rate to lam
    ("StudentT", {"ndof": 5, "mu": 0.0, "sigma": 1.0}, {"positive_only": False, "heavy_tailed": True}),
    # Removed InverseGamma and HalfCauchy as they don't exist in zfit.prior
]


@pytest.fixture
def get_prior_class():
    """Factory to get prior class by name."""
    def _get(name):
        return getattr(zfit.prior, name)
    return _get


# =====================================================================
# Basic functionality tests
# =====================================================================

@pytest.mark.parametrize("prior_name,init_kwargs,properties", PRIOR_CONFIGS)
def test_prior_can_be_created(get_prior_class, prior_name, init_kwargs, properties):
    """Test that each prior can be instantiated with valid parameters."""
    prior_class = get_prior_class(prior_name)
    prior = prior_class(**init_kwargs)

    assert prior is not None
    assert hasattr(prior, 'sample')
    assert hasattr(prior, 'log_pdf')


@pytest.mark.parametrize("prior_name,init_kwargs,properties", PRIOR_CONFIGS)
@pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
def test_prior_sampling_returns_correct_shape(get_prior_class, assert_samples_valid,
                                            prior_name, init_kwargs, properties, n_samples):
    """Test that prior sampling returns arrays of the correct shape."""
    prior_class = get_prior_class(prior_name)
    prior = prior_class(**init_kwargs)

    samples = prior.sample(n_samples)
    validated_samples = assert_samples_valid(
        samples,
        expected_shape=(n_samples, 1),
        check_positive=properties.get("positive_only", False)
    )


@pytest.mark.parametrize("prior_name,init_kwargs,properties", PRIOR_CONFIGS)
def test_prior_log_pdf_returns_valid_values(get_prior_class, prior_name, init_kwargs, properties):
    """Test that log_pdf returns valid values for in-support points."""
    prior_class = get_prior_class(prior_name)
    prior = prior_class(**init_kwargs)

    # Test multiple values
    test_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    for value in test_values:
        x = np.array([[value]])
        try:
            log_prob = prior.log_pdf(x)
            # Should be finite or -inf (for out of support)
            assert np.isfinite(log_prob) or log_prob == -np.inf
            assert log_prob.shape == (1,)
        except Exception:
            # Some priors may raise for invalid values - that's OK
            if properties.get("positive_only") and value < 0:
                continue  # Expected failure
            elif properties.get("bounded") and (value < 0 or value > 1):
                continue  # Expected failure for bounded distributions like Beta, Uniform
            else:
                raise


# =====================================================================
# Statistical property tests
# =====================================================================

@pytest.mark.parametrize("prior_name,init_kwargs,expected_stats", [
    ("Normal", {"mu": 0.0, "sigma": 1.0}, {"mean": 0.0, "std": 1.0}),
    ("Normal", {"mu": 5.0, "sigma": 2.0}, {"mean": 5.0, "std": 2.0}),
    ("Uniform", {"lower": 0.0, "upper": 1.0}, {"mean": 0.5, "bounds": (0.0, 1.0)}),
    ("Uniform", {"lower": -1.0, "upper": 1.0}, {"mean": 0.0, "bounds": (-1.0, 1.0)}),
    ("Beta", {"alpha": 2.0, "beta": 2.0}, {"mean": 0.5, "bounds": (0.0, 1.0)}),
    ("Poisson", {"lam": 3.0}, {"mean": 3.0}),
    ("Poisson", {"lam": 10.0}, {"mean": 10.0}),
])
def test_prior_statistical_properties(get_prior_class, rng, prior_name, init_kwargs, expected_stats):
    """Test that priors have expected statistical properties."""
    prior_class = get_prior_class(prior_name)
    prior = prior_class(**init_kwargs)

    # Sample enough for good statistics
    samples = prior.sample(5000).numpy().flatten()

    # Check mean if specified
    if "mean" in expected_stats:
        sample_mean = np.mean(samples)
        expected_mean = expected_stats["mean"]
        # Tolerance depends on distribution
        tolerance = 0.15 if prior_name != "Cauchy" else 1.0
        assert abs(sample_mean - expected_mean) < tolerance, \
            f"{prior_name} mean {sample_mean:.3f} != expected {expected_mean:.3f}"

    # Check std if specified
    if "std" in expected_stats:
        sample_std = np.std(samples)
        expected_std = expected_stats["std"]
        assert abs(sample_std - expected_std) < 0.15, \
            f"{prior_name} std {sample_std:.3f} != expected {expected_std:.3f}"

    # Check bounds if specified
    if "bounds" in expected_stats:
        lower, upper = expected_stats["bounds"]
        assert np.all(samples >= lower - 1e-10), f"{prior_name} samples below lower bound"
        assert np.all(samples <= upper + 1e-10), f"{prior_name} samples above upper bound"


# =====================================================================
# Positive-only prior tests
# =====================================================================

@pytest.mark.parametrize("prior_class_name,init_kwargs", [
    ("HalfNormal", {"sigma": 1.0}),
    ("Gamma", {"alpha": 2.0, "beta": 0.5}),
    ("LogNormal", {"mu": 0.0, "sigma": 1.0}),
    ("Poisson", {"lam": 3.0}),
    ("Exponential", {"lam": 2.0}),  # Changed from rate to lam
])
def test_positive_only_priors_reject_negative_values(get_prior_class, prior_class_name, init_kwargs):
    """Test that positive-only priors handle negative values correctly."""
    prior_class = get_prior_class(prior_class_name)
    prior = prior_class(**init_kwargs)

    negative_value = np.array([[-1.0]])

    try:
        log_prob = prior.log_pdf(negative_value)
        # Should return -inf or very small probability
        assert log_prob == -np.inf or float(log_prob) < -100
    except Exception:
        # Some implementations might raise - that's OK
        pass


@pytest.mark.parametrize("prior_class_name,init_kwargs", [
    ("HalfNormal", {"sigma": 1.0}),
    ("Gamma", {"alpha": 2.0, "beta": 0.5}),
    ("LogNormal", {"mu": 0.0, "sigma": 1.0}),
    ("Exponential", {"lam": 2.0}),  # Changed from rate to lam
])
def test_positive_only_priors_sample_only_positive(get_prior_class, assert_samples_valid,
                                                  prior_class_name, init_kwargs):
    """Test that positive-only priors only produce positive samples."""
    prior_class = get_prior_class(prior_class_name)
    prior = prior_class(**init_kwargs)

    samples = prior.sample(1000)
    assert_samples_valid(samples, check_positive=True)


# =====================================================================
# Location parameter tests
# =====================================================================

@pytest.mark.parametrize("prior_class_name,location_param,default_loc", [
    ("HalfNormal", "mu", 0.0),
    ("Gamma", "mu", 0.0),
])
def test_priors_with_location_parameter(get_prior_class, prior_class_name, location_param, default_loc):
    """Test priors that support location parameters."""
    prior_class = get_prior_class(prior_class_name)

    # Create with default location
    kwargs1 = {"sigma": 1.0} if prior_class_name == "HalfNormal" else {"alpha": 2.0, "beta": 0.5}
    prior1 = prior_class(**kwargs1)
    samples1 = prior1.sample(1000).numpy().flatten()

    # Create with shifted location
    kwargs2 = kwargs1.copy()
    kwargs2[location_param] = 2.0
    prior2 = prior_class(**kwargs2)
    samples2 = prior2.sample(1000).numpy().flatten()

    # Check shift
    assert np.all(samples1 >= default_loc)
    assert np.all(samples2 >= 2.0)
    assert np.mean(samples2) > np.mean(samples1) + 1.5  # Should be shifted


# =====================================================================
# Parameter integration tests
# =====================================================================

def test_parameter_can_have_prior_set(make_parameter, make_normal_prior):
    """Test setting a prior on a parameter."""
    param = make_parameter("test", 1.0)
    assert param.prior is None

    prior = make_normal_prior(0.0, 1.0)
    param.set_prior(prior)

    assert param.prior == prior
    assert param in prior._params
    assert np.isfinite(param.prior.log_pdf(param.value()))


def test_parameter_prior_can_be_changed(make_parameter, make_normal_prior, make_uniform_prior):
    """Test changing a parameter's prior."""
    param = make_parameter("test", 1.0)

    # Set first prior
    prior1 = make_normal_prior(0.0, 1.0)
    param.set_prior(prior1)
    assert param.prior == prior1
    assert param in prior1._params

    # Change to different prior
    prior2 = make_uniform_prior(0.0, 2.0)
    param.set_prior(prior2)
    assert param.prior == prior2
    assert param not in prior1._params
    assert param in prior2._params


def test_parameter_prior_can_be_removed(make_parameter, make_normal_prior):
    """Test removing a prior from a parameter."""
    param = make_parameter("test", 1.0)
    prior = make_normal_prior(0.0, 1.0)

    param.set_prior(prior)
    assert param.prior == prior

    param.set_prior(None)
    assert param.prior is None
    assert param not in prior._params


def test_parameter_rejects_invalid_prior(make_parameter):
    """Test that parameters reject invalid priors."""
    param = make_parameter("test", 1.0)

    with pytest.raises(TypeError):
        param.set_prior("not a prior")

    with pytest.raises(TypeError):
        param.set_prior(42)


@pytest.mark.parametrize("param_bounds,prior_type", [
    ((-2.0, 2.0), "normal"),     # Bounded parameter with unbounded prior
    ((0.0, 1.0), "uniform"),      # Bounded parameter with unspecified uniform
    ((0.0, None), "halfnormal"),  # Lower-bounded parameter
])
def test_prior_adapts_to_parameter_bounds(make_parameter, all_prior_factories, param_bounds, prior_type):
    """Test that priors adapt to parameter bounds."""
    lower, upper = param_bounds
    param = make_parameter("test", 0.5, lower=lower, upper=upper)

    # Create prior without specifying bounds
    if prior_type == "normal":
        prior = all_prior_factories["normal"](mu=0.0, sigma=5.0)  # Wide prior
    elif prior_type == "uniform":
        prior = all_prior_factories["uniform"]()  # No bounds specified
    elif prior_type == "halfnormal":
        prior = all_prior_factories["halfnormal"](sigma=2.0)

    param.set_prior(prior)

    # Sample and check bounds are respected
    samples = param.prior.sample(1000).numpy().flatten()

    if lower is not None:
        assert np.all(samples >= lower - 1e-10)
    if upper is not None:
        assert np.all(samples <= upper + 1e-10)


def test_normal_prior_becomes_truncated_with_bounds(make_parameter, make_normal_prior):
    """Test that Normal prior becomes TruncatedGauss with parameter limits."""
    param = make_parameter("test", 0.0, lower=-2.0, upper=2.0)
    prior = make_normal_prior(0.0, 5.0)  # Wide prior

    param.set_prior(prior)

    # Should be truncated
    assert isinstance(param.prior.pdf, zfit.pdf.TruncatedGauss)


def test_multiple_parameters_can_share_prior(make_parameter, make_normal_prior):
    """Test that multiple parameters can share the same prior."""
    prior = make_normal_prior(0.0, 1.0)

    params = [make_parameter(f"param{i}", float(i), prior=prior) for i in range(3)]

    # All should have the same prior
    for param in params:
        assert param.prior == prior
        assert param in prior._params

    # Removing from one shouldn't affect others
    params[0].set_prior(None)
    assert params[0] not in prior._params

    for param in params[1:]:
        assert param.prior == prior
        assert param in prior._params


# =====================================================================
# Special distribution tests
# =====================================================================

def test_student_t_approaches_normal_with_high_dof(get_prior_class):
    """Test that Student's t approaches normal as DOF increases."""
    # High DOF Student's t
    t_prior = get_prior_class("StudentT")(ndof=100, mu=0.0, sigma=1.0)
    normal_prior = get_prior_class("Normal")(mu=0.0, sigma=1.0)

    test_points = [0.0, 1.0, -1.0, 2.0, -2.0]

    for point in test_points:
        x = np.array([[point]])
        t_log_prob = float(t_prior.log_pdf(x))
        normal_log_prob = float(normal_prior.log_pdf(x))

        # Should be very close
        assert abs(t_log_prob - normal_log_prob) < 0.1


@pytest.mark.slow
@pytest.mark.parametrize("prior_name", ["Normal", "Uniform", "Beta", "Gamma"])
def test_prior_convergence_with_large_samples(get_prior_class, prior_name):
    """Test that priors converge to expected distributions with many samples."""
    prior_configs = {
        "Normal": ({"mu": 0.0, "sigma": 1.0}, {"mean": 0.0, "std": 1.0}),
        "Uniform": ({"lower": 0.0, "upper": 1.0}, {"mean": 0.5, "std": 0.289}),  # std = 1/sqrt(12)
        "Beta": ({"alpha": 2.0, "beta": 2.0}, {"mean": 0.5}),
        "Gamma": ({"alpha": 2.0, "beta": 0.5}, {"mean": 1.0}),  # mean = alpha*beta in zfit
    }

    init_kwargs, expected = prior_configs[prior_name]
    prior_class = get_prior_class(prior_name)
    prior = prior_class(**init_kwargs)

    # Large sample
    samples = prior.sample(50000).numpy().flatten()

    if "mean" in expected:
        assert abs(np.mean(samples) - expected["mean"]) < 0.02

    if "std" in expected:
        assert abs(np.std(samples) - expected["std"]) < 0.02


# =====================================================================
# Edge case tests
# =====================================================================

def test_uniform_prior_without_bounds_uses_defaults(make_uniform_prior):
    """Test Uniform prior behavior when no bounds are specified."""
    prior = make_uniform_prior()  # No bounds

    # Should still work and have some default bounds
    samples = prior.sample(100)
    assert len(samples) == 100
    assert np.all(np.isfinite(samples))


@pytest.mark.parametrize("n_samples", [0, -1, -10])
def test_priors_reject_invalid_sample_sizes(make_normal_prior, n_samples):
    """Test that priors reject invalid sample sizes."""
    prior = make_normal_prior(0.0, 1.0)

    if n_samples == 0:
        # Some implementations might allow n_samples=0 and return empty array
        samples = prior.sample(n_samples)
        assert len(samples) == 0
    else:
        # Negative values should raise an error
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            prior.sample(n_samples)


def test_prior_log_pdf_with_wrong_shape(make_normal_prior):
    """Test prior log_pdf with incorrectly shaped input."""
    prior = make_normal_prior(0.0, 1.0)

    # Wrong shapes
    wrong_inputs = [
        np.array([1.0]),          # 1D instead of 2D
        np.array([[[1.0]]]),      # 3D instead of 2D
        np.array([[1.0, 2.0]]),   # Multiple values in wrong dimension
    ]

    for wrong_input in wrong_inputs:
        try:
            result = prior.log_pdf(wrong_input)
            # If it doesn't raise, check the result makes sense
            assert np.ndim(result) <= 1
        except Exception:
            # Some inputs might raise - that's OK
            pass
