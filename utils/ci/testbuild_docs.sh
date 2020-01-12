#!/usr/bin/env bash
#    test build docs
echo "============================ Building docs for test ============================"
pip install sphinx sphinx_bootstrap_theme > tmp.txt && echo 'doc utils installed'
bash docs/make_docs.sh 2>&1 | tail -n 11 && \
echo "======================= Finished building docs for test ========================"
