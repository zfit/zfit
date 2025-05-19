#!/usr/bin/env bash
#
# Copyright (c) 2025 zfit
#

# script has to be executed inside folder `docs`
# get current directory name
pushd "$(dirname "$0")" >/dev/null || exit

MAKE_DOCS_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# generate the ReST files
#echo "debug"
#echo ${MAKE_DOCS_PATH}/../zfit
#ls ${MAKE_DOCS_PATH}
CURRENT_DIR=$(pwd)
cd "${MAKE_DOCS_PATH}" || exit 1
make clean
bash "${MAKE_DOCS_PATH}/prepare_apidocs.sh"
make -C "${MAKE_DOCS_PATH}" clean && make -C "${MAKE_DOCS_PATH}" html -j12 &&
  (echo "Documentation successfully built!") || (echo "FAILED to build Documentation" && cd "${CURRENT_DIR}" && exit 1)
cd "${CURRENT_DIR}" || exit 1
popd >/dev/null || exit
