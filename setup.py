"""The setup script."""

#  Copyright (c) 2024 zfit
import os
import platform
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
requirements_dev = [req.split("#")[0].strip() for req in requirements_dev]

nlopt_req = "nlopt>=2.7.1"
ipyopt_req = "ipyopt>=0.12"
extras_require = {}
extras_require["ipyopt"] = [ipyopt_req]
extras_require["nlopt"] = [nlopt_req]
extras_require["hs3"] = ["asdf"]
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
nlopt_req = extras_require.get("nlopt", [])
allreq_nonloptipyopt = allreq.copy()
allreq_nonloptipyopt.pop(allreq.index(nlopt_req[0]))
allreq_nonloptipyopt.pop(allreq.index(extras_require["ipyopt"][0]))

extras_require["all-linux"] = allreq
extras_require["all-darwin"] = allreq_nonloptipyopt + extras_require["nlopt"]
extras_require["all-windows"] = allreq_nonloptipyopt + extras_require["nlopt"]
extras_require["all-silicon"] = allreq_nonloptipyopt

extras_require["tests-linux"] = tests_require + nlopt_req + extras_require["ipyopt"]
extras_require["tests-darwin"] = tests_require + nlopt_req
extras_require["tests-windows"] = tests_require + nlopt_req
extras_require["tests-silicon"] = tests_require

extras_require["dev-linux"] = requirements_dev + extras_require["tests-linux"]
extras_require["dev-darwin"] = requirements_dev + extras_require["tests-darwin"]
extras_require["dev-windows"] = requirements_dev + extras_require["tests-windows"]
extras_require["dev-silicon"] = requirements_dev + extras_require["tests-silicon"]

extras_require["alldev-linux"] = list(
    set(extras_require["all-linux"] + extras_require["dev-linux"])
)
extras_require["alldev-darwin"] = list(
    set(extras_require["all-darwin"] + extras_require["dev-darwin"])
)
extras_require["alldev-silicon"] = list(
    set(extras_require["all-silicon"] + extras_require["dev-silicon"])
)
alldev_windows = list(
    set(extras_require["all-windows"] + extras_require["dev-windows"])
)
alldev_windows.pop(
    alldev_windows.index("jaxlib")
)  # not available on Windows: https://github.com/google/jax/issues/438#issuecomment-939866186
extras_require["alldev-windows"] = alldev_windows

# fill defaults depending on the system
if (platf := platform.system().lower()) == "darwin" and platform.processor() == "arm":
    platf = "silicon"
if platf not in ["linux", "windows", "silicon", "darwin"]:
    warnings.warn(
        f"Platform {platf} not recognized, `dev`, `tests` and `alldev` extras contain all requirements. "
    )
    platf = "linux"
extras_require["tests"] = extras_require[f"tests-{platf}"]
extras_require["dev"] = extras_require[f"dev-{platf}"]
extras_require["alldev"] = extras_require[f"alldev-{platf}"]
extras_require["all"] = extras_require[f"all-{platf}"]

if sys.version_info > (3, 11):  # nlopt, ipyopt not available
    for req_name, req in extras_require.items():
        if nlopt_req in req:
            req.remove(nlopt_req)
        if ipyopt_req in req:
            req.remove(ipyopt_req)
setup(
    install_requires=requirements,
    tests_require=tests_require,
    extras_require=extras_require,
    use_scm_version=True,
)
