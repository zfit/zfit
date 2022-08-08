#!/usr/bin/env bash

#
# Copyright (c) 2022 zfit
#

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"


python -m venv .test_tutorials_env
source .test_tutorials_env/bin/activate
pip install -U pip
pip install ../../[all]

mkdir -p tmp_test_zfit_tutorials && cd tmp_test_zfit_tutorials || exit 1
git clone https://github.com/zfit/zfit-tutorials.git
pip install nbval
pip install -r zfit-tutorials/requirements.txt 2>&1 | tail -n 11 &&
pytest --nbval-lax zfit-tutorials --ignore=zfit-tutorials/experimental --ignore=zfit-tutorials/_website --ignore=zfit-tutorials/_unused
cd -
rm -rf tmp_test_zfit_tutorials
deactivate
rm -rf .test_tutorials_env
