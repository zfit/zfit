#!/usr/bin/env bash

#
# Copyright (c) 2024 zfit
#

BASEDIR=$( dirname -- "$0"; )
python -m venv "${BASEDIR}/.test_examples_env"
source "${BASEDIR}/.test_examples_env/bin/activate"
pip install uv && uv pip install "${BASEDIR}/../../[all]" && uv pip install -r ${BASEDIR}/../../examples/example_requirements.txt || { echo "Failed installing zfit"; exit 1; }
#set -e
#export ZFIT_GRAPH_MODE=0
fail=0
for file in ${BASEDIR}/../../examples/*.py; do
  echo "----------------------------------------------------------------------------------------------"
  echo "Running example: $file"
  echo "----------------------------------------------------------------------------------------------"
  python "$file" || { fail=$((fail+1)) && echo "Failed running example: $file"; }
  echo "----------------------------------------------------------------------------------------------"
  echo "Finished example: $file"
  echo "----------------------------------------------------------------------------------------------"
done
deactivate
rm -rf "${BASEDIR}/.test_examples_env"
echo "Cleaned up, finished"
echo "========================================="
if [ $fail -eq 0 ]; then
  echo "all examples run SUCCESSFULLY"
else
  echo " ${fail} examples FAILED"
fi
echo "========================================="
exit $fail
