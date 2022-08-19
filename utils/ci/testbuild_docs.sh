#!/usr/bin/env bash
#
# Copyright (c) 2022 zfit
#

#    test build docs
BASEDIR=$( dirname -- "$0"; )
python -m venv "${BASEDIR}/.test_docs_env"
source "${BASEDIR}/.test_docs_env/bin/activate"
pip install -U pip
pip install "${BASEDIR}/../../[dev]"

echo "============================ Building docs for test ============================"
pip install sphinx sphinx_bootstrap_theme > tmp.txt && echo 'doc utils installed'
bash "${BASEDIR}/../../docs/make_docs.sh"
echo "======================= Finished building docs for test ========================"

deactivate
rm -rf "${BASEDIR}/.test_docs_env"
