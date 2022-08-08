#!/usr/bin/env bash
#
# Copyright (c) 2022 zfit
#

#    test build docs
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

python -m venv .test_docs_env
source .test_docs_env/bin/activate
pip install -U pip
pip install ../../[dev]

echo "============================ Building docs for test ============================"
pip install sphinx sphinx_bootstrap_theme > tmp.txt && echo 'doc utils installed'
bash ../../docs/make_docs.sh
echo "======================= Finished building docs for test ========================"

deactivate
rm -rf .test_docs_env
cd -
