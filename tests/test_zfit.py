#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `zfit` package."""

import pytest


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    import requests
    return requests.get('https://github.com/zfit/zfit')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    from bs4 import BeautifulSoup
    assert 'GitHub' in BeautifulSoup(response.content, "lxml").title.string
