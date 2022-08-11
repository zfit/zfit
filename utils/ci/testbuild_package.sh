#!/usr/bin/env bash
#
# Copyright (c) 2022 zfit
#

#    test build package
BASEDIR=$( dirname -- "$0"; )
echo "============================ Building package for test ============================"
python "${BASEDIR}/../../setup.py" sdist bdist_wheel  2>&1 | tail -n 15
twine check "${BASEDIR}/../../dist/*"
python "${BASEDIR}/../../setup.py" clean --dist --eggs # cleanup
rm -r "${BASEDIR}/../../build"
echo "======================= Finished building package for test ========================"
