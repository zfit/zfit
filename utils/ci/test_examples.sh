#!/usr/bin/env bash

#
# Copyright (c) 2021 zfit
#

pip install matplotlib mplhep
set -e
for file in examples/*; do
    python "$file"
#    below needed?
#    python $file 2>&1 | tail -n 11 && echo "file $file run sucessfully" || exit 1;
done
