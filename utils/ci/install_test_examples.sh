#!/usr/bin/env bash

#
# Copyright (c) 2022 zfit
#
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

python -m venv test_examples_env
source test_examples_env/bin/activate
pip install -U pip
pip install ../../
pip install -r "../../examples/example_requirements.txt"
#set -e
for file in ../../examples/*; do
  python "$file"
  #    below needed?
  #    python $file 2>&1 | tail -n 11 && echo "file $file run sucessfully" || exit 1;
done
deactivate
rm -rf test_examples_env
cd -
