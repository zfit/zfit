#!/usr/bin/env bash

#
# Copyright (c) 2022 zfit
#
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

python -m venv .test_examples_env
source .test_examples_env/bin/activate
pip install -U pip
pip install ../../[all]
pip install -r "../../examples/example_requirements.txt"
#set -e
for file in ../../examples/*.py; do
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
rm -rf .test_examples_env
cd -
echo "Cleaned up, finished"
