#!/usr/bin/env bash

#
# Copyright (c) 2023 zfit
#

BASEDIR=$( dirname -- "$0"; )
python -m venv "${BASEDIR}/.test_examples_env"
source "${BASEDIR}/.test_examples_env/bin/activate"
pip install -U pip
pip install "${BASEDIR}/../../[all]"
pip install -r "${BASEDIR}/../../examples/example_requirements.txt"
#set -e
fail=0
for file in ${BASEDIR}/../../examples/*.py; do
  python "$file" || fail=1;
done
deactivate
rm -rf "${BASEDIR}/.test_examples_env"
echo "Cleaned up, finished"
echo "========================================="
if [ $fail -eq 0 ]; then
  echo "all examples run SUCCESSFULLY"
else
  echo "some examples FAILED"
fi
echo "========================================="
exit $fail
