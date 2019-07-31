#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
#  Copyright (c) 2019 zfit

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
    requirements = requirements_file.read().splitlines()

with open(os.path.join(here, 'requirements_dev.txt'), encoding='utf-8') as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, 'CHANGELOG.rst'), encoding='utf-8') as changelog_file:
    changelog = changelog_file.read()

# split the developer requirements into setup and test requirements
if not requirements_dev.count("") == 1 or requirements_dev.index("") == 0:
    raise SyntaxError("requirements_dev.txt has the wrong format: setup and test "
                      "requirements have to be separated by one blank line.")
requirements_dev_split = requirements_dev.index("")

setup_requirements = requirements_dev[:requirements_dev_split]
test_requirements = requirements_dev[requirements_dev_split + 1:]  # +1: skip empty line

setup(
    author="Jonas Eschle",
    # author_email='zfit@physik.uzh.ch',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Physics',
        ],

    maintainer="zfit",
    maintainer_email='zfit@physik.uzh.ch',
    description="scalable pythonic model fitting for high energy physics",
    install_requires=requirements,
    license="BSD 3-Clause",
    long_description=readme + '\n\n' + changelog,
    include_package_data=True,
    keywords='TensorFlow, model, fitting, scalable, HEP',
    name='zfit',
    python_requires=">=3.6",
    packages=find_packages(include=['zfit', 'zfit.ztf',
                                    'zfit.util', 'zfit.core', "zfit.minimizers", 'zfit.models']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zfit/zfit',
    version='0.3.5',
    zip_safe=False,
    )
