#!/usr/bin/env bash

#
# Copyright (c) 2022 zfit
#

BASEDIR=$( dirname -- "$0"; )
python -m venv "${BASEDIR}/.test_examples_env"
source "${BASEDIR}/.test_examples_env/bin/activate"
pip install -U pip
pip install -U "${BASEDIR}/../../[all]"
pip install -r "${BASEDIR}/../../examples/example_requirements.txt"
#set -e
for file in ${BASEDIR}/../../examples/*.py; do
  sucess=0
  python "$file" && sucess=1 || break
  #    below needed?
  #    python $file 2>&1 | tail -n 11 && echo "file $file run sucessfully" || exit 1;
done
echo "========================================="
if [ $sucess -eq 1 ]; then
  echo "all examples run SUCCESSFULLY"
else
  echo "some examples FAILED"
fi
echo "========================================="
deactivate
rm -rf "${BASEDIR}/.test_examples_env"
echo "Cleaned up, finished"
