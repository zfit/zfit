#!/bin/bash
#
# Copyright (c) 2022 zfit
#

# script has to be executed inside folder `docs`
# get current directory name
pushd "$(dirname "$0")" >/dev/null || exit
MAKE_DOCS_PATH="$(
  cd "$(dirname "$0")" || exit
  pwd -P
)"
MAKE_DOCS_PATH=$(pwd -P)
popd >/dev/null || exit

echo "Invoking docformatter"
docformatter "${MAKE_DOCS_PATH}/../zfit/" -r --in-place --wrap-descriptions 120 --wrap-summaries 120
echo "Replacing auto docs for duplicated args"
find "${MAKE_DOCS_PATH}/../zfit/" -type f -name '*.py' -exec "${MAKE_DOCS_PATH}/../utils/api/replace_argdocs.py" {} ';'
