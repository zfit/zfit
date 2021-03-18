"""Used to make pytest functions available globally."""

#  Copyright (c) 2021 zfit
import sys

import pytest

import zfit

try:
    import pytest_randomly
except ImportError:
    pass
else:
    pytest_randomly.random_seeder = [zfit.settings.set_seed]
init_modules = sys.modules.keys()


@pytest.fixture(autouse=True)
def setup_teardown():
    for m in sys.modules.keys():
        if m not in init_modules:
            del (sys.modules[m])

    yield
    from zfit.core.parameter import ZfitParameterMixin
    ZfitParameterMixin._existing_params.clear()

    from zfit.util.cache import clear_graph_cache
    clear_graph_cache()
    import zfit
    zfit.run.set_graph_mode()
    zfit.run.set_autograd_mode()
    for m in sys.modules.keys():
        if m not in init_modules:
            del (sys.modules[m])
    import gc
    gc.collect()


def pytest_addoption(parser):
    parser.addoption("--longtests", action="store", default=False)
