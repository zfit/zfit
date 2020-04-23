#!/usr/bin/env bash
#    test build package
echo "============================ Building package for test ============================"
python setup.py sdist bdist_wheel  2>&1 | tail -n 15
twine check dist/*
rm -r dist  # cleanup
echo "======================= Finished building package for test ========================"
