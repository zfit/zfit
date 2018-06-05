"""
Definition of the pdf interface, base etc.
"""

from __future__ import print_function, division, absolute_import


class BaseDistribution(object):


    def sample(self):
        pass

    def log_prob(self):
        pass

    def batch_shape_tensor(self):
        pass

    def event_shape_tensor(self):
        pass
