#!/usr/bin/env bash
#    test build package
echo "============================ Building package for test ============================"
mkdir tmp_testbuild_zfit && cd tmp_testbuild_zfit
python setup.py sdist bdist_wheel  2>&1 | tail -n 15
twine check dist/*
cd ..
rm -r tmp_testbuild_zfit  # cleanup
echo "======================= Finished building package for test ========================"
