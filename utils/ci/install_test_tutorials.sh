#!/usr/bin/env bash

#
# Copyright (c) 2022 zfit
#

#    test build package

BASEDIR=$(dirname -- "$0")
python -m venv "${BASEDIR}"/.test_tutorials_env || exit 1
source "${BASEDIR}/.test_tutorials_env/bin/activate" && pip install -U pip && pip install "${BASEDIR}/../../[all]" && pip install nbval
mkdir -p "${BASEDIR}/.tmp_test_zfit_tutorials" && cd "${BASEDIR}/.tmp_test_zfit_tutorials" || exit 1
git clone https://github.com/zfit/zfit-tutorials.git &&
pip install -r zfit-tutorials/requirements.txt 2>&1 | tail -n 11 &&
pytest --nbval-lax zfit-tutorials --ignore=zfit-tutorials/experimental --ignore=zfit-tutorials/_website --ignore=zfit-tutorials/_unused
cd - && rm -rf "${BASEDIR}/.tmp_test_zfit_tutorials"
deactivate
rm -rf "${BASEDIR}/.test_tutorials_env"
