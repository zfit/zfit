"""The setup script."""

#  Copyright (c) 2023 zfit
import os
import sys
import warnings

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
extras_require["tests-darwin"] = tests_require + extras_require.get("nlopt", [])
extras_require["tests-linux"] = (
    extras_require["tests-darwin"] + extras_require["ipyopt"]
)
extras_require["dev-linux"] = requirements_dev + extras_require["tests-linux"]
extras_require["dev-darwin"] = requirements_dev + extras_require["tests-darwin"]
extras_require["alldev-linux"] = list(
    set(extras_require["all"] + extras_require["dev-linux"])
)
alldev_nonlinux = list(set(extras_require["all"] + extras_require["dev-darwin"]))
alldev_nonlinux.pop(
    alldev_nonlinux.index(extras_require["ipyopt"][0])
)  # ipyopt is not available on non linux systems
extras_require["alldev-darwin"] = alldev_nonlinux
alldev_windows = alldev_nonlinux.copy()
alldev_windows.pop(
    alldev_windows.index("jaxlib")
)  # not available on Windows: https://github.com/google/jax/issues/438#issuecomment-939866186
extras_require["alldev-windows"] = alldev_windows
alldev_silicon = alldev_nonlinux.copy()
alldev_silicon.pop(
    alldev_silicon.index("nlopt")  # https://github.com/DanielBok/nlopt-python/issues/13
)  # not available on Silicon:
extras_require["alldev-silicon"] = alldev_silicon

# fill defaults depending on the system
if (platform := sys.platform) == "darwin" and sys.platform.processor() == "arm":
    platform = "silicon"
elif platform == "win32":
    platform = "windows"
if platform in ["linux", "windows", "silicon", "darwin"]:
    extras_require["alldev"] = extras_require[f"alldev-{platform}"]
    extras_require["dev"] = extras_require[f"dev-{platform}"]
    extras_require["tests"] = extras_require[f"tests-{platform}"]
else:
    warnings.warn(
        f"Platform {platform} not recognized, `dev`, `tests` and `alldev` extras are not defined. "
    )


setup(
    install_requires=requirements,
    tests_require=tests_require,
    extras_require=extras_require,
    use_scm_version=True,
)
