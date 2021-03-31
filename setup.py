"""The setup script."""

#  Copyright (c) 2021 zfit
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(here, 'requirements_dev.txt'), encoding='utf-8') as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

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
setup(
    author=("Jonas Eschle <Jonas.Eschle@cern.ch>,"
            " Albert Puig <albert.puig.navarro@gmail.com>,"
            "Rafael Silva Coutinho <rsilvaco@cern.ch>,"
            "Matthieu Marinangeli <matthieu.marinangeli@cern.ch>"),
    install_requires=requirements,
    tests_require=tests_require,
    extras_require={
        'dev': requirements_dev + tests_require
    },
    use_scm_version=True,
)
