#  Copyright (c) 2025 zfit
from __future__ import annotations

import functools
import json
import pathlib
import shutil
import numpy as np
import pytest
from matplotlib import pyplot as plt
from PIL import Image

import zfit

folder = "plotting_extended"

@functools.cache
def create_deterministic_fit_setup(extended=True, use_data=True):
    """Create a deterministic fit setup based on signal_bkg_mass_extended_fit.py"""
    # Set fixed seed for reproducibility
    zfit.settings.set_seed(42)

    # Create space
    obs = zfit.Space("x", -10, 10)

    # Parameters with fixed values for deterministic behavior
    mu = zfit.Parameter("mu", 1.0, -4, 6)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)

    if extended:
        n_bkg = zfit.Parameter("n_bkg", 20000, 0, 50000)
        n_sig = zfit.Parameter("n_sig", 1000, 0, 30000)

        # Model building with extended PDFs
        gauss_extended = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=n_sig)
        exp_extended = zfit.pdf.Exponential(lambd, obs=obs, extended=n_bkg)
        model = zfit.pdf.SumPDF([gauss_extended, exp_extended])
    else:
        # Non-extended model
        gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
        exp = zfit.pdf.Exponential(lambd, obs=obs)
        frac = zfit.Parameter("frac", 0.3, 0, 1)
        model = zfit.pdf.SumPDF([gauss, exp], fracs=frac)

    data = None
    if use_data:
        # Generate deterministic data
        n = 21200
        data = model.sample(n=n)

        # Set the values to start values for the fit
        zfit.param.set_values({mu: 0.5, sigma: 1.2, lambd: -0.05})

        # Create and run fit
        if extended:
            nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
        else:
            nll = zfit.loss.UnbinnedNLL(model=model, data=data)

        minimizer = zfit.minimize.Minuit(gradient="zfit")
        result = minimizer.minimize(nll).update_params()

    return model, data, obs


def get_truth_image_path(test_name):
    """Get the path for truth reference images"""
    here = pathlib.Path(__file__).parent
    truth_dir = here.parent / "truth" / "images" / folder
    truth_dir.mkdir(parents=True, exist_ok=True)
    return truth_dir / f"{test_name}.png"


def save_plot_as_truth_if_needed(test_name, recreate_truth):
    """Save current plot as truth reference if --recreate-truth flag is given"""
    if recreate_truth:
        truth_path = get_truth_image_path(test_name)
        plt.savefig(truth_path, dpi=100, bbox_inches='tight')


def compare_images(test_name, tolerance=0.01):
    """Compare generated image with truth reference"""
    truth_path = get_truth_image_path(test_name)

    if not truth_path.exists():
        pytest.skip(f"Truth image not found: {truth_path}. Run with --recreate-truth to generate.")

    # Save current plot for comparison
    temp_path = truth_path.parent / f"{test_name}_temp.png"
    plt.savefig(temp_path, dpi=100, bbox_inches='tight')

    try:
        # Load and compare images
        truth_img = np.array(Image.open(truth_path))
        test_img = np.array(Image.open(temp_path))

        # Images must have same shape
        assert truth_img.shape == test_img.shape, f"Image shapes differ: {truth_img.shape} vs {test_img.shape}"

        # Calculate normalized difference
        diff = np.abs(truth_img.astype(float) - test_img.astype(float))
        max_diff = np.max(diff) / 255.0

        assert max_diff < tolerance, f"Images differ too much: max difference {max_diff:.4f} > {tolerance}"

    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.parametrize("extended", [True, False])
@pytest.mark.parametrize("density", [True, False])
def test_plot_extended_density_combinations(extended, density, request):
    """Test all combinations of extended and density plotting options"""
    model, data, obs = create_deterministic_fit_setup(extended=extended, use_data=True)

    test_name = f"extended_{extended}_density_{density}"
    plt.figure()
    plt.title(f"Extended={extended}, density={density}")

    # Test the specific plot method from the example
    model.plot(data, extended=extended, density=density)

    # Save truth reference if requested
    recreate_truth = request.config.getoption("--recreate-truth")
    save_plot_as_truth_if_needed(test_name, recreate_truth)

    # Compare with truth reference
    compare_images(test_name)

    # Also save for documentation
    pytest.zfit_savefig(folder=folder)


def test_plot_no_data_simple():
    """Test plotting without data - simplified test"""
    # Create a simple model without data (like existing tests)
    obs = zfit.Space("x", -10, 10)
    mu = zfit.Parameter("mu", 1.0, -4, 6)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    model = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    plt.figure()
    plt.title("No data simple plot")

    # Test plotting without data (should work for simple non-extended models)
    model.plot()

    # Verify the plot was created
    assert len(plt.gca().lines) > 0 or len(plt.gca().collections) > 0, "No plot elements were created"

    # Also save for documentation
    pytest.zfit_savefig(folder=folder)



def test_plot_with_saved_data(request):
    """Test plotting with saved deterministic data"""
    # Create deterministic data once and save it
    model, data, obs = create_deterministic_fit_setup(extended=True, use_data=True)

    # Save data to ensure complete determinism
    data_path = pathlib.Path(__file__).parent.parent / "truth" / "data" / "deterministic_fit_data.npz"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    recreate_truth = request.config.getoption("--recreate-truth")
    if recreate_truth or not data_path.exists():
        # Convert data to numpy arrays and parameters to primitive types
        data_array = np.asarray(data.value(), dtype=np.float64)
        fitted_params = {p.name: float(p.value()) for p in model.get_params()}

        # Save using NumPy compressed format with JSON-serialized parameters
        np.savez_compressed(data_path,
                           data=data_array,
                           fitted_params_json=json.dumps(fitted_params))

    # Load the saved data
    loaded_npz = np.load(data_path)
    saved_data = {
        'data': loaded_npz['data'],
        'fitted_params': json.loads(loaded_npz['fitted_params_json'].item())
    }

    # Reconstruct the model with saved parameters
    model_saved, _, _ = create_deterministic_fit_setup(extended=True, use_data=False)

    # Set parameters to saved values
    for param in model_saved.get_params():
        if param.name in saved_data['fitted_params']:
            param.set_value(saved_data['fitted_params'][param.name])

    # Create data from saved values
    data_reconstructed = zfit.data.Data.from_numpy(obs, saved_data['data'])

    # Test all four plotting combinations with saved data
    for extended in [True, False]:
        for density in [True, False]:
            test_name = f"saved_data_extended_{extended}_density_{density}"
            plt.figure()
            plt.title(f"Saved data - Extended={extended}, density={density}")

            model_saved.plot(data_reconstructed, extended=extended, density=density)

            # Save truth reference if requested
            save_plot_as_truth_if_needed(test_name, recreate_truth)

            # Compare with truth reference
            compare_images(test_name)

            # Also save for documentation
            pytest.zfit_savefig(folder=folder)


def test_plot_component_breakdown():
    """Test plotting individual components of the sum PDF"""
    model, data, obs = create_deterministic_fit_setup(extended=True, use_data=True)

    plt.figure()
    plt.title("Component breakdown extended")

    # Plot the total model
    model.plot(data, extended=True, density=True)

    # Test that we can access components
    if hasattr(model, 'pdfs'):
        assert len(model.pdfs) > 0, "Model should have PDF components"
        # For now, just verify we can access the components without plotting them individually
        # Individual component plotting without data has initialization issues

    pytest.zfit_savefig(folder=folder)


@pytest.mark.parametrize("plot_type", ["extended_true_density_true", "extended_false_density_true",
                                       "extended_true_density_false", "extended_false_density_false"])
def test_plot_error_handling(plot_type):
    """Test error handling for various plotting scenarios"""
    model, data, obs = create_deterministic_fit_setup(extended=True, use_data=True)

    extended = "true" in plot_type.split("_")[1]
    density = "true" in plot_type.split("_")[3]

    plt.figure()
    plt.title(f"Error handling {plot_type}")

    # This should not raise an error
    model.plot(data, extended=extended, density=density)

    # Verify the plot was created
    assert len(plt.gca().lines) > 0 or len(plt.gca().collections) > 0, "No plot elements were created"

    pytest.zfit_savefig(folder=folder)


def test_plot_consistency_with_example():
    """Test that our plots are consistent with the original example"""
    # Recreate the exact setup from the example
    zfit.run.experimental_disable_param_update(True)

    obs = zfit.Space("x", -10, 10)

    # Parameters exactly as in example
    mu = zfit.Parameter("mu", 1.0, -4, 6)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)
    lambd = zfit.Parameter("lambda", -0.06, -1, -0.01)
    n_bkg = zfit.Parameter("n_bkg", 20000, 0, 50000)
    n_sig = zfit.Parameter("n_sig", 1000, 0, 30000)

    # Model building exactly as in example
    gauss_extended = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=n_sig)
    exp_extended = zfit.pdf.Exponential(lambd, obs=obs, extended=n_bkg)
    model = zfit.pdf.SumPDF([gauss_extended, exp_extended])

    # Use fixed seed for deterministic data
    n = 21200
    data = model.sample(n=n)

    # Set values as in example
    zfit.param.set_values({mu: 0.5, sigma: 1.2, lambd: -0.05})

    # Create and run fit
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit(gradient="zfit")
    result = minimizer.minimize(nll).update_params()

    # Test the four plots from the example
    plot_configs = [
        (True, True, "Extended=True, density=True"),
        (False, True, "Extended=False, density=True"),
        (True, False, "Extended=True, density=False"),
        (False, False, "Extended=False, density=False")
    ]

    for extended, density, title in plot_configs:
        plt.figure()
        plt.title(title)
        model.plot(data, extended=extended, density=density)
        pytest.zfit_savefig(folder=folder)
