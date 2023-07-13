"""Used to make pytest functions available globally."""
#  Copyright (c) 2023 zfit

import collections.abc
import os
import pathlib
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    import pytest_randomly
except ImportError:
    pass
else:
    import zfit

    pytest_randomly.random_seeder = [zfit.settings.set_seed]
init_modules = sys.modules.keys()


@pytest.fixture(autouse=True)
def setup_teardown():
    import zfit

    old_chunksize = zfit.run.chunking.max_n_points
    old_active = zfit.run.chunking.active
    old_graph_mode = zfit.run.get_graph_mode()
    old_autograd_mode = zfit.run.get_autograd_mode()

    for m in sys.modules.keys():
        if m not in init_modules:
            del sys.modules[m]

    yield

    from zfit.core.parameter import ZfitParameterMixin, _reset_auto_number

    ZfitParameterMixin._existing_params.clear()
    _reset_auto_number()

    from zfit.util.cache import clear_graph_cache

    clear_graph_cache()
    import zfit

    zfit.run.chunking.active = old_active
    zfit.run.chunking.max_n_points = old_chunksize
    zfit.run.set_graph_mode(old_graph_mode)
    zfit.run.set_autograd_mode(old_autograd_mode)
    zfit.run.set_graph_cache_size()
    for m in sys.modules.keys():
        if m not in init_modules:
            del sys.modules[m]

    zfit.settings.set_seed(None)
    import gc

    gc.collect()


def pytest_addoption(parser):
    parser.addoption("--longtests", action="store_true", default=False)
    parser.addoption("--longtests-kde", action="store_true", default=False)
    parser.addoption("--recreate-truth", action="store_true", default=False)


def pytest_configure():
    here = os.path.dirname(os.path.abspath(__file__))
    images_dir = pathlib.Path(here).joinpath(
        "..", "docs", "images", "_generated_by_tests"
    )
    images_dir.mkdir(exist_ok=True)

    def savefig(figure=None, folder=None):
        if figure is None:
            figure = plt.gcf()
        title_sanitized = (
            figure.axes[0]
            .get_title()
            .replace(" ", "_")
            .replace("$", "_")
            .replace("\\", "_")
            .replace("__", "_")
        )
        title_sanitized = (
            title_sanitized.replace("/", "_")
            .replace(".", "_")
            .replace(":", "_")
            .replace(",", "")
        )
        if not title_sanitized:
            raise RuntimeError("Title has to be set for plot that should be saved.")
        foldersave = images_dir
        if folder is not None:
            foldersave = foldersave.joinpath(folder)
        foldersave.mkdir(exist_ok=True, parents=True)
        savepath = foldersave.joinpath(title_sanitized)
        plt.savefig(str(savepath))

    pytest.zfit_savefig = savefig


ARBITRARY_VALUE = "ZFIT_ARBITRARY_VALUE"


def cleanup_recursive(dict1, dict2):
    """Compare two dicts recursively."""
    if not (
        isinstance(dict1, collections.abc.Mapping)
        and isinstance(dict2, collections.abc.Mapping)
    ):
        return dict1, dict2
    dict1, dict2 = dict1.copy(), dict2.copy()
    for key in set(dict1.keys()) | set(dict2.keys()):
        val1 = dict1.get(key)
        val2 = dict2.get(key)
        if isinstance(val1, str) and val1 == ARBITRARY_VALUE:
            dict2[key] = ARBITRARY_VALUE
        elif isinstance(val2, str) and val2 == ARBITRARY_VALUE:
            dict1[key] = ARBITRARY_VALUE
        elif isinstance(val1, collections.abc.Mapping) or isinstance(
            val2, collections.abc.Mapping
        ):
            dict1sub, dict2sub = cleanup_recursive(dict1.get(key), dict2.get(key))
            if dict1sub is not None:
                dict1[key] = dict1sub
            if dict2sub is not None:
                dict2[key] = dict2sub
        elif isinstance(val1, np.ndarray):
            dict1[key] = val1.tolist()
        elif isinstance(val2, np.ndarray):
            dict2[key] = val2.tolist()
        elif key == "value_fn":  # We have a composed parameter
            if isinstance(val1, str):
                dict1[key] = "FUNC_IGNORED_NOT_STABLE"
            if isinstance(val2, str):
                dict2[key] = "FUNC_IGNORED_NOT_STABLE"
    return dict1, dict2


@pytest.helpers.register
def cleanup_hs3(obj1, obj2):
    """Compare two HS3 dicts."""

    if not isinstance(obj1, collections.abc.Mapping) or not isinstance(
        obj2, collections.abc.Mapping
    ):
        raise TypeError(
            f"obj1 and obj2 both need to be of type 'Mapping', are {obj1} and {obj2}"
        )
    missing2 = set(obj1.keys()) - set(obj2.keys())
    missing1 = set(obj2.keys()) - set(obj1.keys())
    if missing1 and missing2:
        raise ValueError(
            f"Both objects are missing keys: {missing1} and {missing2}. "
            f"obj1: {obj1}, obj2: {obj2}"
        )
    return cleanup_recursive(obj1, obj2)


@pytest.helpers.register
def get_truth(folder, filename, request, newval=None):
    """Get the truth value for a given folder and filename.

    Args:
        folder (str): The folder in which the file is located.
        filename (str): The filename of the file.
        newval (Any): If given, the truth value will be overwritten with this value.

    Returns:
        The truth value.
    """
    if newval is None:
        raise ValueError(
            "New value has to be given. This will only be stored and overwrite the current file *if* the --recreate-truth flag is given. "
            "Otherwise, the truth value will be loaded from the file."
        )
    current_dir = pathlib.Path(__file__).parent
    static_dir = current_dir / "truth" / folder
    static_dir.mkdir(exist_ok=True, parents=True)

    filepath = static_dir / (filename)
    # check if need to update value first
    recreate_truth = request.config.getoption("--recreate-truth")
    if recreate_truth:
        if filepath.suffix == ".json":
            import json

            if isinstance(newval, str):
                newval = json.loads(newval)
            with open(filepath, "w") as f:
                json.dump(newval, f)
        elif filepath.suffix == ".yaml":
            import yaml

            with open(filepath, "w") as f:
                yaml.dump(newval, f)

        elif filepath.suffix == ".asdf":
            newval.write_to(filepath)
        else:
            raise ValueError(
                f"Filetype {filepath.suffix} not supported. Needs manual implementation of the truth value in get_truth function."
            )

    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist")
    if filepath.suffix == ".json":
        import json

        with open(filepath, "r") as f:
            return json.load(f)
    elif filepath.suffix == ".yaml":
        import yaml

        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    elif filepath.suffix == ".asdf":
        import asdf

        with asdf.open(filepath, copy_arrays=True) as f:
            return asdf.AsdfFile(f.tree, copy_arrays=True)
    else:
        raise ValueError(f"Filetype {filepath.suffix} not supported")
    raise ValueError(f"Folder {folder} not supported.")
