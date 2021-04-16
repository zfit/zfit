#!/usr/bin/env bash

#
# Copyright (c) 2021 zfit
#

mkdir -p ~/test_zfit_tutorials && cd ~/test_zfit_tutorials || exit 1
git clone https://github.com/zfit/zfit-tutorials.git
pip install nbval
pip install -r zfit-tutorials/requirements.txt 2>&1 | tail -n 11 && \
pytest --nbval-lax zfit-tutorials --ignore=zfit-tutorials/experimental || exit 1
cd -
