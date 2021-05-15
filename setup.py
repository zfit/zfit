"""The setup script."""

#  Copyright (c) 2021 zfit
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(here, 'requirements_dev.txt'), encoding='utf-8') as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

extras_require = {'ipyopt': ['ipyopt<0.12']}  # TODO: osx wheels? https://gitlab.com/g-braeunlich/ipyopt/-/issues/4

allreq = sum(extras_require.values(), [])

tests_require = [
    'pytest>=3.4.2,<5.4',  # breaks unittests
    'pytest-runner>=2.11.1',
    'pytest-rerunfailures>=6',
    'pytest-xdist',
    'pytest-ordering',
    'pytest-randomly',
    'scikit-hep-testdata',
    'pytest-timeout>=1',
    'matplotlib'  # for plots in examples
]
extras_require['all'] = allreq
extras_require['tests'] = tests_require + extras_require['ipyopt']
extras_require['dev'] = requirements_dev + extras_require['tests']
extras_require['alldev'] = list(set(extras_require['all'] + extras_require['dev']))

setup(
    author=("Jonas Eschle, "
            "Albert Puig, "
            "Rafael Silva Coutinho, "
            "Matthieu Marinangeli"),
    install_requires=requirements,
    tests_require=tests_require,
    extras_require=extras_require,
    use_scm_version=True,
)
