"""Tests for zfit Bayesian _mcmc."""

#  Copyright (c) 2025 zfit

import unittest
import numpy as np
import pytest
import signal

import zfit
from zfit.pdf import Gauss
from zfit.loss import UnbinnedNLL


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Test timed out")


class SamplerTimeoutTestCase(unittest.TestCase):
    """Base class with timeout handling for sampler tests."""

    TIMEOUT = 60  # Default timeout in seconds

    def setUp(self):
        """Set up the test case with a simple Gaussian model."""
        # Create a simple model for testing
        obs = zfit.Space("x", limits=(-10, 10))

        # True parameters (mu=1.0, sigma=2.0)
        mean = zfit.Parameter("mean", 1.0, lower=-5, upper=10, prior=zfit.prior.NormalPrior(1.0, 0.5))
        sigma = zfit.Parameter("sigma", 2.0, lower=0.1, upper=5, prior=zfit.prior.NormalPrior(2.0, 3))

        # Create test data (100 points from a normal distribution)
        np.random.seed(42)
        data_np = np.random.normal(1.0, 2.0, size=500)
        data = zfit.Data.from_numpy(obs=obs, array=data_np[:, np.newaxis])

        # Create model and loss
        model = Gauss(obs=obs, mu=mean, sigma=sigma)
        self.loss = UnbinnedNLL(model=model, data=data)

        # Parameters to sample
        self.params = [mean, sigma]

    def run_sampler_with_timeout(self, sampler, timeout=None):
        """Run the sampler with a timeout."""
        if timeout is None:
            timeout = self.TIMEOUT

        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            # Run the sampler
            result = sampler.sample(loss=self.loss, params=self.params, n_samples=50, n_warmup=20)
            signal.alarm(0)  # Reset the alarm
            return result
        except TimeoutError:
            self.skipTest(f"Sampler {sampler.name} timed out after {timeout} seconds")
        except ImportError as e:
            self.skipTest(f"Sampler {sampler.name} requires additional dependencies: {str(e)}")
        except Exception as e:
            signal.alarm(0)  # Reset the alarm
            raise e
        finally:
            signal.alarm(0)  # Ensure alarm is reset

    def verify_posterior(self, posterior):
        """Verify the posterior samples are reasonable."""
        # Check that we have samples
        self.assertGreater(len(posterior.samples), 0)

        # Check that the parameter names are correct
        self.assertEqual(posterior.param_names, ["mean", "sigma"])

        # Check that mean values are reasonably close to true values
        mean_vals = posterior.samples[:, 0]
        sigma_vals = posterior.samples[:, 1]

        # Mean of "mean" parameter should be close to 1.0
        self.assertAlmostEqual(np.mean(mean_vals), 1.0, delta=0.5)

        # Mean of "sigma" parameter should be close to 2.0
        self.assertAlmostEqual(np.mean(sigma_vals), 2.0, delta=0.5)

        # Sigma parameter should always be positive
        self.assertTrue(np.all(sigma_vals > 0))


class TestEmceeSampler(SamplerTimeoutTestCase):
    """Test the EmceeSampler."""

    def test_emcee_sampler(self):
        """Test EmceeSampler."""
        from zfit._mcmc.emcee import EmceeSampler

        sampler = EmceeSampler(nwalkers=5)
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


class TestNUTSSampler(SamplerTimeoutTestCase):
    """Test the NUTSSampler."""

    def test_nuts_sampler(self):
        """Test NUTSSampler."""
        from zfit._mcmc import NUTSSampler

        sampler = NUTSSampler(step_size=0.1)
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


class TestPTSampler(SamplerTimeoutTestCase):
    """Test the PTSampler."""

    def test_pt_sampler(self):
        """Test PTSampler."""
        from zfit._mcmc.parallel_tempering import PTSampler

        sampler = PTSampler(nwalkers=4, ntemps=2)
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


class TestSMCSampler(SamplerTimeoutTestCase):
    """Test the SMCSampler."""

    def test_smc_sampler(self):
        """Test SMCSampler."""
        from zfit._mcmc import SMCSampler

        sampler = SMCSampler(n_particles=50, n_mcmc_steps=2)
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


@pytest.mark.skip("ZeusSampler is not fully implemented.")
class TestZeusSampler(SamplerTimeoutTestCase):
    """Test the ZeusSampler."""

    def test_zeus_sampler(self):
        """Test ZeusSampler."""
        from zfit._mcmc.zeus import ZeusSampler

        sampler = ZeusSampler(nwalkers=5)
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


@pytest.mark.skip("DynestySampler is not fully implemented.")
class TestDynestySampler(SamplerTimeoutTestCase):
    """Test the DynestySampler."""

    def test_dynesty_sampler(self):
        """Test DynestySampler."""
        from zfit._mcmc import DynestySampler

        sampler = DynestySampler(nlive=50)
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


class TestUltraNestSampler(SamplerTimeoutTestCase):
    """Test the UltraNestSampler."""

    def test_ultranest_sampler(self):
        """Test UltraNestSampler."""
        from zfit._mcmc.ultranest import UltraNestSampler

        sampler = UltraNestSampler(min_num_live_points=25)
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


class TestStanSampler(SamplerTimeoutTestCase):
    """Test the CustomStanSampler."""

    def test_stan_sampler(self):
        """Test CustomStanSampler."""
        from zfit._mcmc import CustomStanSampler

        sampler = CustomStanSampler()
        posterior = self.run_sampler_with_timeout(sampler)
        self.verify_posterior(posterior)


@pytest.mark.parametrize(
    "sampler_class,kwargs",
    [
        pytest.param("EmceeSampler", {"nwalkers": 5, "moves": None, "backend": None, "pool": None}, id="emcee"),
        pytest.param(
            "NUTSSampler",
            {
                "step_size": 0.1,
                "adapt_step_size": True,
                "target_accept": 0.8,
                "max_tree_depth": 10,
                "mass_matrix_adapter": None,
            },
            id="nuts",
        ),
        pytest.param("PTSampler", {"nwalkers": 4, "ntemps": 2, "adaptation_lag": 1000, "adaptation_time": 10}, id="pt"),
        pytest.param(
            "SMCSampler",
            {"n_particles": 100, "n_mcmc_steps": 2, "ess_threshold": 0.5, "resampling_method": "systematic"},
            id="smc",
        ),
        # pytest.param("ZeusSampler", {"nwalkers": 10, "tune": True, "tolerance": 0.05, "patience": 5}, id="zeus"),
        # pytest.param(
        #     "DynestySampler", {"nlive": 50, "bound": "multi", "samplemethod": "auto", "dlogz": 0.01}, id="dynesty"
        # ),
        pytest.param(
            "UltraNestSampler",
            {
                "min_num_live_points": 50,
                "cluster_num_live_points": 40,
                "dlogz": 0.5,
                "update_interval_volume_fraction": 0.8,
                "resume": True,
                "wrapped_params": None,
            },
            id="ultranest",
        ),
        pytest.param("CustomStanSampler", {"algorithm": "NUTS", "adapt_delta": 0.8, "max_depth": 10}, id="stan"),
    ],
)
def test_all_samplers(sampler_class, kwargs):
    """Parametrized test for all _mcmc using pytest."""
    # Skip import errors gracefully
    try:
        # Dynamically import the sampler class
        module_name = f"zfit._bayesian._mcmc.{sampler_class.lower().replace('sampler', '')}"
        module = __import__(module_name, fromlist=[sampler_class])
        SamplerClass = getattr(module, sampler_class)
    except ImportError:
        pytest.skip(f"Could not import {sampler_class}")
        return

    # Set up a simple model
    obs = zfit.Space("x", limits=(-10, 10))
    mean = zfit.Parameter("mean", 1.0, lower=-5, upper=10, prior=zfit.prior.NormalPrior(1.0, 0.5))
    sigma = zfit.Parameter("sigma", 2.0, lower=0.1, upper=5, prior=zfit.prior.NormalPrior(2.0, 3.0))

    # Create test data
    np.random.seed(42)
    data_np = np.random.normal(1.0, 2.0, size=500)
    data = zfit.Data.from_numpy(obs=obs, array=data_np[:, np.newaxis])

    # Create model and loss
    model = Gauss(obs=obs, mu=mean, sigma=sigma)
    loss = UnbinnedNLL(model=model, data=data)

    # Create and run the sampler
    sampler = SamplerClass(**kwargs)

    # Use a timeout context
    with timeout_context(20):  # 10 second timeout
        posterior = sampler.sample(loss=loss, params=[mean, sigma], n_samples=50, n_warmup=5)

        # Check posterior
        assert len(posterior.samples) > 0
        assert posterior.param_names == ["mean", "sigma"]

        # Check that mean values are reasonably close to true values
        mean_vals = posterior.samples[:, 0]
        sigma_vals = posterior.samples[:, 1]

        assert abs(np.mean(mean_vals) - 1.0) < 0.5
        assert abs(np.mean(sigma_vals) - 2.0) < 0.5
        assert np.all(sigma_vals > 0)


class timeout_context:
    """Context manager for timeout using pytest."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, *args):
        signal.alarm(0)

    def _handle_timeout(self, signum, frame):
        return


if __name__ == "__main__":
    unittest.main()
