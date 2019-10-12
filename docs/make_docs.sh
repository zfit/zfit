#!/bin/bash
# script has to be executed inside folder `docs`
# get current directory name
pushd `dirname $0` > /dev/null
MAKE_DOCS_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
#MAKE_DOCS_PATH=$(pwd -P)
popd > /dev/null

# generate the ReST files
#echo "debug"
#echo ${MAKE_DOCS_PATH}/../zfit
#ls ${MAKE_DOCS_PATH}
#SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,inherited-members sphinx-apidoc -o ${MAKE_DOCS_PATH}/api ${MAKE_DOCS_PATH}/../zfit  -fMeT && \
#python3 ${MAKE_DOCS_PATH}/api/tools/change_headline.py ${MAKE_DOCS_PATH}/api/zfit.* && \
make -C ${MAKE_DOCS_PATH} clean && make -C ${MAKE_DOCS_PATH} html -j4 && \
echo "Documentation successfully built!" || echo "FAILED to build Documentation"
