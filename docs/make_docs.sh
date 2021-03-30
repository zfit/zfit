#!/bin/bash
#
# Copyright (c) 2021 zfit
#

# script has to be executed inside folder `docs`
# get current directory name
pushd "$(dirname "$0")" >/dev/null || exit
MAKE_DOCS_PATH="$(
    cd "$(dirname "$0")" || exit
    pwd -P
)"
#MAKE_DOCS_PATH=$(pwd -P)
popd >/dev/null || exit

# generate the ReST files
#echo "debug"
#echo ${MAKE_DOCS_PATH}/../zfit
#ls ${MAKE_DOCS_PATH}
make clean
bash prepare_apidocs.sh
make -C "${MAKE_DOCS_PATH}" clean && make -C "${MAKE_DOCS_PATH}" html -j8 &&
    echo "Documentation successfully built!" || echo "FAILED to build Documentation"
