#!/bin/bash
#
# Copyright (c) 2022 zfit
#

# script has to be executed inside folder `docs`
# get current directory name
pushd "$(dirname "$0")" >/dev/null || exit

MAKE_DOCS_PATH=$( dirname -- "$0"; )
# generate the ReST files
#echo "debug"
#echo ${MAKE_DOCS_PATH}/../zfit
#ls ${MAKE_DOCS_PATH}
make clean
bash "${MAKE_DOCS_PATH}/prepare_apidocs.sh"
make -C "${MAKE_DOCS_PATH}" clean && make -C "${MAKE_DOCS_PATH}" html -j8 &&
  echo "Documentation successfully built!" || echo "FAILED to build Documentation"
popd >/dev/null || exit
