#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `zfit` package."""
from __future__ import print_function, division, absolute_import

suppress_gpu = True
if suppress_gpu:
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

import zfit


def test_empty():
    """Empty test."""
    assert True
