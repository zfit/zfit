"""
Definition of the pdf interface, base etc.
"""

from __future__ import print_function, division, absolute_import


class BaseDistribution(object):

    def sample(self, sample_shape=(), seed=None, name='sample'):
        pass

    def log_prob(self, value, name='log_prob'):
        pass

    def batch_shape_tensor(self, name='batch_shape_tensor'):
        pass

    def event_shape_tensor(self, name='event_shape_tensor'):
        pass
