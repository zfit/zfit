"""Used to make pytest functions available globally"""

#  Copyright (c) 2021 zfit


import zfit
try:
    import pytest_randomly
except ImportError:
    pass
else:
    pytest_randomly.random_seeder = [zfit.settings.set_seed]



def pytest_addoption(parser):
    parser.addoption("--longtests", action="store", default=False)
