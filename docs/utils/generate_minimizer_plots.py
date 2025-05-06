"""This script generates visualization plots for different minimizers in zfit.

It creates plots showing how different minimizers perform on a complex version of the Rosenbrock function,
tracking their paths and metrics like number of function evaluations and gradient calculations.
"""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import zfit

# Get the directory of this script
here = Path(__file__).parent.absolute()

# Set the output directory for the plots
outpath = Path(here / "../images/_generated/minimizers").absolute().resolve()
outpath.mkdir(parents=True, exist_ok=True)
print(f"Saving plots to {outpath}")

# Set the figure size and style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


def save_plot(filename):
    """Save the current plot to the specified filename."""
    plt.tight_layout()
    plt.savefig(outpath / f"{filename}", dpi=100, bbox_inches="tight")
    plt.close()


def complex_rosenbrock(x, a=1.0, b=100.0, c=0.5, d=0.1):
    """A more complex version of the Rosenbrock function.

    This function adds additional terms to the standard Rosenbrock function
    to make it more challenging for minimizers.

    Args:
        x: Input array of shape (n,) where n >= 2
        a, b, c, d: Parameters controlling the shape of the function

    Returns:
        The function value at x
    """
    x = np.asarray(x)
    result = 0

    # Standard Rosenbrock terms
    for i in range(len(x) - 1):
        result += b * (x[i + 1] - x[i] ** 2) ** 2 + (a - x[i]) ** 2

    # Additional complexity
    for i in range(len(x)):
        result += c * np.sin(d * x[i]) ** 2

    return result


class FunctionWrapper:
    """Wrapper for the function to track evaluations and gradients."""

    def __init__(self, func, name="Function"):
        """Initialize the wrapper.

        Args:
            func: The function to wrap
            name: Name of the function for display
        """
        self.func = func
        self.name = name
        self.evaluations = 0
        self.gradient_calls = 0
        self.history = []

    def __call__(self, x):
        """Call the function and track the evaluation."""
        self.evaluations += 1
        result = self.func(x)
        self.history.append((np.copy(x), result))
        return result

    def gradient(self, x):
        """Calculate the gradient using finite differences."""
        self.gradient_calls += 1
        eps = 1e-8
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = np.copy(x)
            x_plus[i] += eps
            x_minus = np.copy(x)
            x_minus[i] -= eps

            grad[i] = (self.func(x_plus) - self.func(x_minus)) / (2 * eps)

        return grad

    def reset(self):
        """Reset the tracking counters and history."""
        self.evaluations = 0
        self.gradient_calls = 0
        self.history = []


def plot_minimizer_paths(minimizer_classes, starting_points, func_wrapper, title, filename, pbar):
    """Plot the paths taken by different minimizers.

    Args:
        minimizer_classes: List of minimizer classes to test
        starting_points: List of starting points for minimization
        func_wrapper: Function wrapper object
        title: Title for the plot
        filename: Filename to save the plot
    """
    # Find the true minimum of the complex Rosenbrock function
    # For the complex Rosenbrock function, the true minimum is close to (1, 1)
    # but might be slightly different due to the additional terms
    true_minimum = [1.0, 1.0]  # Approximate true minimum for the Rosenbrock function

    X, Y, Z, x, y = create_meshgrid_arrays(func_wrapper)

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the contour
    contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)
    fig.colorbar(contour, ax=ax, label="Function value")

    # Plot the true minimum
    ax.plot(
        true_minimum[0],
        true_minimum[1],
        "o",
        color="green",
        markersize=15,
        markeredgewidth=2,
        alpha=1.0,
        label="True Minimum",
    )

    # Plot the paths for each minimizer and starting point
    colors = plt.cm.tab10.colors
    markers = ["x", "s", "^", "d", "v"]

    for i, minimizer_class in enumerate(minimizer_classes):
        minimizer_name = minimizer_class.__name__
        color = colors[i % len(colors)]

        for j, start_point in enumerate(starting_points):
            marker = markers[j % len(markers)]

            # Create the minimizer
            minimizer = minimizer_class()

            # Reset the function wrapper
            func_wrapper.reset()

            # Set the errordef
            func_wrapper.errordef = 0.5

            # Create a copy of the starting point
            params = [zfit.Parameter(f"x{i}", start_point[i], stepsize=0.1) for i in range(len(start_point))]

            # Minimize the function
            try:
                start_time = time.time()
                result = minimizer.minimize(func_wrapper, params)
                end_time = time.time()

                # Extract the path from the history
                path = np.array([point[0] for point in func_wrapper.history])

                # Plot the path
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    "--",
                    color=color,
                    alpha=0.4,
                    linewidth=1.1,
                    label=f"{minimizer_name}",
                )
                label = None
                if j == 0:
                    label = "Func Eval"
                ax.plot(path[:, 0], path[:, 1], "o", color=color, alpha=0.6, markersize=2.7, label=label)

                # Plot the starting point
                ax.plot(start_point[0], start_point[1], marker, color=color, markersize=10, alpha=0.8)

                # Plot the final point (where minimizer stops) in red
                label = None
                if j == 0:
                    label = "Found Min"
                ax.plot(
                    path[-1, 0], path[-1, 1], "o", color="red", markersize=8, markeredgewidth=2, alpha=0.8, label=label
                )

                # Add text with metrics
                ax.annotate(
                    "Starting point\n"
                    f"Evals: {func_wrapper.evaluations}\n"
                    f"Grads: {func_wrapper.gradient_calls}\n"
                    f"Time: {end_time - start_time:.2f}s",
                    xy=(path[0, 0], path[0, 1]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=11,
                    color="black",
                    alpha=0.9,
                )

            except Exception as e:
                print(f"Error with {minimizer_name} from {start_point}: {e}")
            finally:
                pbar.update(1)

    # Set reasonable axis limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # Add labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    # Add legend
    ax.legend(loc="upper left")

    # Save the plot
    save_plot(filename)


def create_meshgrid_arrays(func_wrapper):
    # Create a grid of x and y values for the contour plot
    x = np.linspace(-1.5, 1.5, 100)  # Set reasonable bounds
    y = np.linspace(-1.0, 1.5, 100)  # Set reasonable bounds
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    # Calculate function values for the contour plot
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = func_wrapper.func([X[j, i], Y[j, i]])
    return X, Y, Z, x, y


def plot_minimizers():
    """Generate plots for different minimizers."""
    # Define the function to minimize
    func = lambda x: complex_rosenbrock(x)
    func_wrapper = FunctionWrapper(func, name="Complex Rosenbrock")

    # Define the minimizers to test
    # Group minimizers by category
    minuit_minimizers = [zfit.minimize.Minuit]

    levenberg_marquardt_minimizers = [zfit.minimize.LevenbergMarquardt]

    ipyopt_minimizers = [zfit.minimize.Ipyopt]

    scipy_minimizers = [
        zfit.minimize.ScipyBFGS,
        zfit.minimize.ScipyLBFGSB,
        zfit.minimize.ScipyTrustConstr,
        zfit.minimize.ScipyPowell,
        zfit.minimize.ScipySLSQP,
        zfit.minimize.ScipyTruncNC,
        zfit.minimize.ScipyCOBYLA,
        zfit.minimize.ScipyTrustNCG,
        zfit.minimize.ScipyDogleg,
        zfit.minimize.ScipyTrustKrylov,
        zfit.minimize.ScipyNewtonCG,
    ]

    nlopt_minimizers = [
        zfit.minimize.NLoptLBFGSV1,
        zfit.minimize.NLoptTruncNewtonV1,
        zfit.minimize.NLoptSLSQPV1,
        zfit.minimize.NLoptMMAV1,
        zfit.minimize.NLoptCCSAQV1,
        zfit.minimize.NLoptSubplexV1,
        zfit.minimize.NLoptCOBYLAV1,
        zfit.minimize.NLoptMLSLV1,
        zfit.minimize.NLoptStoGOV1,
        zfit.minimize.NLoptBOBYQAV1,
        zfit.minimize.NLoptISRESV1,
        zfit.minimize.NLoptESCHV1,
        zfit.minimize.NLoptShiftVarV1,
    ]

    # Combine all minimizers for the combined plot
    minimizer_classes = (
        minuit_minimizers + levenberg_marquardt_minimizers + ipyopt_minimizers + scipy_minimizers + nlopt_minimizers
    )

    # Define the starting points within reasonable bounds
    starting_points = [
        [-0.5, 1.2],  # Top left
        [1.2, 1.2],  # Top right
        [0.0, -0.0],  # Center
        [-0.5, -0.5],  # Bottom left
        [0.5, -0.5],  # Bottom right
    ]
    pbar = tqdm(
        enumerate(minimizer_classes),
        desc="Generating minimizer plots",
        total=len(minimizer_classes) * len(starting_points),
    )
    plot_minimizer_paths_pbar = partial(plot_minimizer_paths, pbar=pbar)
    # # Plot the combined minimizer paths
    # plot_minimizer_paths_pbar(
    #     minimizer_classes,
    #     starting_points,
    #     func_wrapper,
    #     "Minimizer Paths on Complex Rosenbrock Function",
    #     "minimizer_paths.png",
    # )

    # Plot individual minimizer paths for each category

    # Minuit minimizers
    print("Generating Minuit minimizer plots...")
    for minimizer_class in minuit_minimizers:
        func_wrapper.reset()
        plot_minimizer_paths_pbar(
            [minimizer_class],
            starting_points,
            func_wrapper,
            f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
            f"{minimizer_class.__name__.lower()}_paths.png",
        )

    # Levenberg-Marquardt minimizers
    print("Generating Levenberg-Marquardt minimizer plots...")
    for minimizer_class in levenberg_marquardt_minimizers:
        func_wrapper.reset()
        plot_minimizer_paths_pbar(
            [minimizer_class],
            starting_points,
            func_wrapper,
            f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
            f"{minimizer_class.__name__.lower()}_paths.png",
        )

    # Ipyopt minimizers
    print("Generating Ipyopt minimizer plots...")
    for minimizer_class in ipyopt_minimizers:
        func_wrapper.reset()
        plot_minimizer_paths_pbar(
            [minimizer_class],
            starting_points,
            func_wrapper,
            f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
            f"{minimizer_class.__name__.lower()}_paths.png",
        )

    # Scipy minimizers
    print("Generating Scipy minimizer plots...")
    for minimizer_class in scipy_minimizers:
        func_wrapper.reset()
        plot_minimizer_paths_pbar(
            [minimizer_class],
            starting_points,
            func_wrapper,
            f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
            f"{minimizer_class.__name__.lower()}_paths.png",
        )

    # NLopt minimizers
    print("Generating NLopt minimizer plots...")
    for minimizer_class in nlopt_minimizers:
        func_wrapper.reset()
        plot_minimizer_paths_pbar(
            [minimizer_class],
            starting_points,
            func_wrapper,
            f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
            f"{minimizer_class.__name__.lower()}_paths.png",
        )


def main():
    """Generate all minimizer plots."""
    print("Generating minimizer plots...")

    # Generate the plots
    plot_minimizers()

    print("Done generating minimizer plots.")


if __name__ == "__main__":
    main()
