#!/bin/bash

mkdir -p ~/test_zfit_tutorials && cd ~/test_zfit_tutorials
git clone https://github.com/zfit/zfit-tutorials.git
pip install nbval
pip install -r zfit-tutorials/requirements.txt 2>&1 | tail -n 11 && \
pytest zfit-tutorials --ignore=zfit-tutorials/experimental --nbval-lax
cd -


