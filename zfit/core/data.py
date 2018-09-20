from __future__ import print_function, division, absolute_import
#
class AbstractData(object):
    """Data class holding data."""

    def __init__(self, lower, upper):
        self.lower_limit = None
