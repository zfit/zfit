"""The setup script."""

#  Copyright (c) 2023 zfit
import os
import sys

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(
    os.path.join(here, "requirements.txt"), encoding="utf-8"
) as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(
    os.path.join(here, "requirements_dev.txt"), encoding="utf-8"
) as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

extras_require = {}
extras_require["ipyopt"] = ["ipyopt>=0.12"]
if sys.version_info[1] < 11:
    extras_require["nlopt"] = ["nlopt>=2.7.1"]
extras_require["hs3"] = ["asdf"]
# Python 3.7 not supported anymore
extras_require["uproot"] = ["awkward-pandas"]
allreq = sum(extras_require.values(), [])

tests_require = [
    "pytest>=3.4.2",  # breaks unittests
    "pytest-runner>=2.11.1",
    "pytest-rerunfailures>=6",
    "pytest-xdist",
    "pytest-ordering",
    "pytest-randomly",
    "scikit-hep-testdata",
    "pytest-timeout>=1",
    "matplotlib",  # for plots in examples
]
extras_require["all"] = allreq
extras_require["tests-nonlinux"] = tests_require + extras_require.get("nlopt", [])
extras_require["tests"] = extras_require["tests-nonlinux"] + extras_require["ipyopt"]
extras_require["dev"] = requirements_dev + extras_require["tests"]
extras_require["dev-nonlinux"] = requirements_dev + extras_require["tests-nonlinux"]
extras_require["alldev"] = list(set(extras_require["all"] + extras_require["dev"]))
alldev_nonlinux = list(set(extras_require["all"] + extras_require["dev-nonlinux"]))
alldev_nonlinux.pop(
    alldev_nonlinux.index(extras_require["ipyopt"][0])
)  # ipyopt is not available on non linux systems
extras_require["alldev-nonlinux"] = alldev_nonlinux
alldev_windows = alldev_nonlinux.copy()
alldev_windows.pop(
    alldev_windows.index("jaxlib")
)  # not available on Windows: https://github.com/google/jax/issues/438#issuecomment-939866186
extras_require["alldev-windows"] = alldev_windows

setup(
    install_requires=requirements,
    tests_require=tests_require,
    extras_require=extras_require,
    use_scm_version=True,
)
