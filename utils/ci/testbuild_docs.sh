#!/usr/bin/env bash
#
# Copyright (c) 2022 zfit
#

#    test build docs
pushd "$(dirname "$0")" >/dev/null || exit
MAKE_DOCS_PATH="$(
  cd "$(dirname "$0")" || exit
  pwd -P
)"
MAKE_DOCS_PATH=$(pwd -P)
popd >/dev/null || exit

python -m venv "${MAKE_DOCS_PATH}/.test_docs_env"
source "${MAKE_DOCS_PATH}/.test_docs_env/bin/activate"
pip install -U pip
pip install "${MAKE_DOCS_PATH}/../../[dev]"

echo "============================ Building docs for test ============================"
pip install sphinx sphinx_bootstrap_theme > tmp.txt && echo 'doc utils installed'
bash "${MAKE_DOCS_PATH}/../../docs/make_docs.sh"
echo "======================= Finished building docs for test ========================"

deactivate
rm -rf "${MAKE_DOCS_PATH}/.test_docs_env"
cd -
