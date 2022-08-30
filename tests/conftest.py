"""Used to make pytest functions available globally."""

#  Copyright (c) 2022 zfit

import os
import pathlib
import sys

import matplotlib.pyplot as plt
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

    from zfit.core.parameter import ZfitParameterMixin

    ZfitParameterMixin._existing_params.clear()

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
    import gc

    gc.collect()


def pytest_addoption(parser):
    parser.addoption("--longtests", action="store_true", default=False)
    parser.addoption("--longtests-kde", action="store_true", default=False)


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
