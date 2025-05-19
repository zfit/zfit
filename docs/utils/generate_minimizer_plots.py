"""This script generates visualization plots for different minimizers in zfit.

It creates both static and animated plots showing how different minimizers perform
on a complex version of the Rosenbrock function, tracking their paths and metrics
like number of function evaluations and gradient calculations.
"""

#  Copyright (c) 2025 zfit

from __future__ import annotations

import time
from functools import lru_cache, partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import ray
import tensorflow as tf
from matplotlib import animation
from tqdm import tqdm

import zfit
import zfit.z.numpy as znp

# Get the directory of this script
here = Path(__file__).parent.absolute()

# Set the output directory for the plots
outpath = Path(here / "../images/_generated/minimizers").absolute().resolve()
outpath.mkdir(parents=True, exist_ok=True)
print(f"Saving plots to {outpath}")

# Set the figure size and style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)
plt.rcParams["font.size"] = 12
plt.rcParams["figure.autolayout"] = True


def save_plot(filename):
    """Save the current plot to the specified filename with '_static' appended."""
    # Append '_static' to the filename before the extension
    base_name, ext = filename.rsplit(".", 1)
    static_filename = f"{base_name}_static.{ext}"

    plt.tight_layout()
    plt.savefig(outpath / static_filename, dpi=160)
    plt.close()


# @z.function
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
    x = znp.asarray(x)

    # Standard Rosenbrock terms
    result = b * (x[1] - x[0] ** 2) ** 2 + (a - x[0]) ** 2

    # Additional complexity
    result += c * znp.sin(d * x[0]) ** 2
    result += c * znp.sin(d * x[1]) ** 2

    return result


MAX_EVALS = 5000  # Maximum number of function evaluations


class MaxNumberOfEvaluations(Exception):
    pass


class FunctionWrapper(zfit.loss.SimpleLoss):
    """Wrapper for the function to track evaluations and gradients."""

    def __init__(self, func, params):
        """Initialize the wrapper.

        Args:
            func: The function to wrap
            name: Name of the function for display
        """
        super().__init__(func, params=params, jit=False, gradient="zfit", hessian="zfit", errordef=0.5)
        self.func = func
        self.evaluations = 0
        self.gradient_calls = 0
        self.hessian_calls = 0
        self.history = []

    # def __call__(self, *args, **kwargs):
    #     """Call the function and track the evaluation."""
    #
    def value(self, *args, **kwargs):
        if self.evaluations > MAX_EVALS:
            msg = "Maximum number of evaluations exceeded."
            raise MaxNumberOfEvaluations(msg)
        self.evaluations += 1
        x = np.array(self.get_params())
        result = super().value(*args, **kwargs)
        self.history.append((x, result))
        return result

    def gradient(self, *args, **kwargs):
        """Calculate the gradient using finite differences."""
        self.gradient_calls += 1
        return super().gradient(*args, **kwargs)

    def value_gradient(self, *args, **kwargs) -> tuple[tf.Tensor, tf.Tensor]:
        self.gradient_calls += 1
        return super().value_gradient(*args, **kwargs)
        # eps = 1e-8
        # grad = np.zeros_like(x)
        #
        # for i in range(len(x)):
        #     x_plus = np.copy(x)
        #     x_plus[i] += eps
        #     x_minus = np.copy(x)
        #     x_minus[i] -= eps
        #
        #     grad[i] = (self.func(x_plus) - self.func(x_minus)) / (2 * eps)
        #
        # return grad

    def hessian(self, *args, **kwargs):
        self.hessian_calls += 1
        return super().hessian(*args, **kwargs)

    def reset(self):
        """Reset the tracking counters and history."""
        self.evaluations = 0
        self.gradient_calls = 0
        self.hessian_calls = 0
        self.history = []

    def _check_assert_autograd(self, params):
        pass


def plot_minimizer_paths(minimizer_classes, starting_points, meshgrid_arrays, func, title, filename, pbar=None):
    """Plot the paths taken by different minimizers.

    Args:
        minimizer_classes: List of minimizer classes to test
        starting_points: List of starting points for minimization
        func: Function wrapper object
        title: Title for the plot
        filename: Filename to save the plot
    """
    # Find the true minimum of the complex Rosenbrock function
    # For the complex Rosenbrock function, the true minimum is close to (1, 1)
    # but might be slightly different due to the additional terms
    true_minimum = [1.0, 1.0]  # Approximate true minimum for the Rosenbrock function
    params = [zfit.Parameter(f"x{i}", starting_points[0][i], stepsize=0.1) for i in range(len(starting_points[0]))]
    func_wrapper = FunctionWrapper(func, params=params)
    X, Y, Z, x, y = meshgrid_arrays

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

    # Store paths for animation
    all_paths = []
    all_colors = []
    all_labels = []
    all_markers = []
    all_start_points = []

    for i, minimizer_class in enumerate(minimizer_classes):
        minimizer_name = minimizer_class.__name__
        color = colors[i % len(colors)]

        for j, start_point in enumerate(starting_points):
            marker = markers[j % len(markers)]

            # Create the minimizer
            minimizer = minimizer_class()

            # Reset the function wrapper
            func_wrapper.reset()

            # # Set the errordef
            # func_wrapper.errordef = 0.5

            # Create a copy of the starting point
            params = func_wrapper.get_params()

            # Minimize the function
            try:
                for param, value in zip(params, start_point):
                    param.set_value(value)
                start_time = time.time()
                #     # Start bar as a process
                #     p = multiprocessing.Process(target=lambda min=minimizer, p=params, f=func_wrapper: min.minimize(f, p))
                #     p.start()
                #     # Wait for 30 seconds or until process finishes
                #     p.join(30)
                #
                #     # If thread is still active
                #     if p.is_alive():
                #         print(f"{minimizer_name} from {start_point} took too long (more than 30s), terminating...")
                #
                #         # Terminate - may not work if process is stuck for good
                #         p.terminate()
                #         # OR Kill - will work for sure, no chance for process to finish nicely however
                #         # p.kill()
                #
                #         p.join()
                result = minimizer.minimize(func_wrapper, params)
                end_time = time.time()

                converged = result.valid

                # Extract the path from the history
                path = np.array([point[0] for point in func_wrapper.history])

                # Store path data for animation
                all_paths.append(path)
                all_colors.append(color)
                all_labels.append(f"{minimizer_name}")
                all_markers.append(marker)
                all_start_points.append(start_point)

                minlabel = f"{minimizer_name}" if j == 0 else None
                # Plot the path
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    "--",
                    color=color,
                    alpha=0.4,
                    linewidth=1.1,
                    label=minlabel,
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
                    f"Hess: {func_wrapper.hessian_calls}\n"
                    f"Time: {end_time - start_time:.2f}s" + ("\nNot converged" if not converged else ""),
                    xy=(path[0, 0], path[0, 1]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=11,
                    color="black",
                    alpha=0.9,
                )

            except MaxNumberOfEvaluations:
                print(f"Maximum number of evaluations exceeded for {minimizer_name} from {start_point}")
            except Exception as e:
                print(f"Error with {minimizer_name} from {start_point}: {e}")
            finally:
                if pbar is not None:
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

    # Save the static plot
    save_plot(filename)

    # Create and save the animation
    if all_paths:
        return create_animation.remote(
            X,
            Y,
            Z,
            true_minimum,
            all_paths,
            all_colors,
            all_labels,
            all_markers,
            all_start_points,
            x,
            y,
            title,
            filename,
        )
    else:
        return create_fake_animation.remote(filename=filename)


@ray.remote
def create_fake_animation(filename):
    """Create a fake animation for testing purposes."""
    base_name, ext = filename.rsplit(".", 1)
    savepath = outpath / f"{base_name}.gif"

    # create a fake animation with a static image

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.text(0.5, 0.5, "Error when running this minimizer", fontsize=20, ha="center", va="center")
    ax.axis("off")

    def init():
        return (ax,)

    def update(_):
        return (ax,)

    anim = animation.FuncAnimation(fig, update, frames=1, init_func=init, blit=True, interval=200)
    anim.save(savepath, writer="pillow", fps=1, dpi=160)
    plt.close(fig)


@ray.remote
def create_animation(
    X, Y, Z, true_minimum, all_paths, all_colors, all_labels, all_markers, all_start_points, x, y, title, filename
):
    """Create and save an animation of the minimization process.

    Args:
        X, Y, Z: Meshgrid arrays for contour plot
        true_minimum: True minimum of the function
        all_paths: List of paths taken by the minimizers
        all_colors: List of colors for each path
        all_labels: List of labels for each path
        all_markers: List of markers for each starting point
        all_start_points: List of starting points
        x, y: Axis limits
        title: Title for the plot
        filename: Filename to save the animation
    """
    # Find the maximum path length
    max_len = max(len(path) for path in all_paths)

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.set_tight_layout(True)

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
        label="True Min",
    )

    # Set reasonable axis limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # Add labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title + " (Animation)")

    # Plot all starting points
    for i, start_point in enumerate(all_start_points):
        ax.plot(
            start_point[0],
            start_point[1],
            all_markers[i],
            color=all_colors[i % len(all_colors)],
            markersize=10,
            alpha=0.8,
        )

    # Initialize line objects for each path
    lines = []
    points = []

    for i, _path in enumerate(all_paths):
        (line,) = ax.plot([], [], "--", color=all_colors[i], alpha=0.6, linewidth=1.5)
        (point,) = ax.plot([], [], "o", color=all_colors[i], alpha=0.9, markersize=6)
        lines.append(line)
        points.append(point)

    # Create a legend with unique labels
    unique_labels = list(set(all_labels))
    handles = [
        plt.Line2D([0], [0], color=all_colors[all_labels.index(label)], linestyle="--", label=label)
        for label in unique_labels
    ]
    handles.append(plt.Line2D([0], [0], marker="o", color="green", linestyle="", markersize=10, label="True Minimum"))
    ax.legend(handles=handles, loc="upper left")

    # Animation initialization function
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return lines + points

    # Animation update function
    def update(frame):
        for _i, (line, point, path) in enumerate(zip(lines, points, all_paths)):
            if frame < len(path):
                line.set_data(path[: frame + 1, 0], path[: frame + 1, 1])
                point.set_data([path[frame, 0]], [path[frame, 1]])
            else:
                # If this path is shorter than the current frame, show the entire path
                line.set_data(path[:, 0], path[:, 1])
                if len(path) > 0:
                    point.set_data([path[-1, 0]], [path[-1, 1]])
        return lines + points

    fps = 5
    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=min(max_len, 500),  # Limit to certain frames for efficiency
        init_func=init,
        blit=True,
        interval=1000 / fps,  # ms
    )

    # Save the animation
    base_name, ext = filename.rsplit(".", 1)
    anim.save(outpath / f"{base_name}.gif", writer="pillow", fps=fps, dpi=160)
    plt.close()


@lru_cache
def create_meshgrid_arrays(func):
    # Create a grid of x and y values for the contour plot
    x = znp.linspace(-1.5, 1.5, 100)  # Set reasonable bounds
    y = znp.linspace(-1.0, 1.5, 100)  # Set reasonable bounds
    X, Y = znp.meshgrid(x, y)
    Z = np.zeros_like(X)
    # Calculate function values for the contour plot
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = func([X[j, i], Y[j, i]])
    return np.array(X), np.array(Y), np.array(Z), np.array(x), np.array(y)


def plot_minimizers():
    """Generate plots for different minimizers."""
    n_physical_cpus = psutil.cpu_count(logical=False)

    ray.init(num_cpus=n_physical_cpus)
    # ray.init(local_mode=True)
    # Define the function to minimize
    func = complex_rosenbrock

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
        [-0.5, -0.4],  # Bottom left
        [0.5, -0.6],  # Bottom right
    ]
    pbar = tqdm(
        enumerate(minimizer_classes),
        desc="Generating minimizer plots",
        total=len(minimizer_classes) * len(starting_points),
    )
    plot_minimizer_paths_remote = partial(plot_minimizer_paths, pbar=pbar)
    # plot_minimizer_paths_remote = ray.remote(plot_minimizer_paths).remote
    # # Plot the combined minimizer paths
    # plot_minimizer_paths_remote(
    #     minimizer_classes,
    #     starting_points,
    #     func_wrapper,
    #     "Minimizer Paths on Complex Rosenbrock Function",
    #     "minimizer_paths.png",
    # )

    # Plot individual minimizer paths for each category

    # Minuit minimizers
    print("Generating all minimizer plots...")

    @lru_cache
    def create_meshgrid_arrays(func):
        # Create a grid of x and y values for the contour plot
        x = znp.linspace(-1.5, 1.5, 100)  # Set reasonable bounds
        y = znp.linspace(-1.0, 1.5, 100)  # Set reasonable bounds
        X, Y = znp.meshgrid(x, y)
        Z = np.zeros_like(X)
        # Calculate function values for the contour plot
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = func([X[j, i], Y[j, i]])
        return np.array(X), np.array(Y), np.array(Z), np.array(x), np.array(y)

    # Create the meshgrid arrays for the contour plot
    meshgrid_arrays = create_meshgrid_arrays(func)

    remotes = []
    for minimizer_class in minimizer_classes:
        # func_wrapper.reset()
        future = plot_minimizer_paths_remote(
            [minimizer_class],
            starting_points,
            meshgrid_arrays=meshgrid_arrays,
            func=func,
            title=f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
            filename=f"{minimizer_class.__name__.lower()}_paths.png",
        )
        remotes.append(future)

    # Minuit minimizers
    # print("Generating Minuit minimizer plots...")
    # remotes = []
    # for minimizer_class in minuit_minimizers:
    #     func_wrapper.reset()
    #     future = plot_minimizer_paths_remote(
    #         [minimizer_class],
    #         starting_points,
    #         func_wrapper,
    #         f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
    #         f"{minimizer_class.__name__.lower()}_paths.png",
    #     )
    #     remotes.append(future)
    #
    # # Levenberg-Marquardt minimizers
    # print("Generating Levenberg-Marquardt minimizer plots...")
    # for minimizer_class in levenberg_marquardt_minimizers:
    #     func_wrapper.reset()
    #     future = plot_minimizer_paths_remote(
    #         [minimizer_class],
    #         starting_points,
    #         func_wrapper,
    #         f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
    #         f"{minimizer_class.__name__.lower()}_paths.png",
    #     )
    #     remotes.append(future)
    #
    # # Ipyopt minimizers
    # print("Generating Ipyopt minimizer plots...")
    # for minimizer_class in ipyopt_minimizers:
    #     func_wrapper.reset()
    #     future = plot_minimizer_paths_remote(
    #         [minimizer_class],
    #         starting_points,
    #         func_wrapper,
    #         f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
    #         f"{minimizer_class.__name__.lower()}_paths.png",
    #     )
    #     remotes.append(future)
    #
    # # Scipy minimizers
    # print("Generating Scipy minimizer plots...")
    # for minimizer_class in scipy_minimizers:
    #     func_wrapper.reset()
    #     future = plot_minimizer_paths_remote(
    #         [minimizer_class],
    #         starting_points,
    #         func_wrapper,
    #         f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
    #         f"{minimizer_class.__name__.lower()}_paths.png",
    #     )
    #     remotes.append(future)
    #
    # # NLopt minimizers
    # print("Generating NLopt minimizer plots...")
    # for minimizer_class in nlopt_minimizers:
    #     func_wrapper.reset()
    #     future = plot_minimizer_paths_remote(
    #         [minimizer_class],
    #         starting_points,
    #         func_wrapper,
    #         f"{minimizer_class.__name__} Paths on Complex Rosenbrock Function",
    #         f"{minimizer_class.__name__.lower()}_paths.png",
    #     )
    #     remotes.append(future)

    # check all ray remote returns and make sure they all finish
    print("Waiting for all minimizer to finish...")
    all_data = []
    ntot = len(remotes)
    for _ in tqdm(range(ntot), desc="Waiting for minimizers to finish"):
        finished, remotes = ray.wait(remotes)
        data = ray.get(finished)
        all_data.extend(data)

    print("All minimizer plots finished.")


def main():
    """Generate all minimizer plots."""
    print("Generating minimizer plots...")
    print("Static plots will have '_static' appended to their filenames.")
    print("Animated plots will be saved as GIFs with the original filenames.")

    # Generate the plots
    plot_minimizers()

    print("Done generating minimizer plots.")


if __name__ == "__main__":
    main()
