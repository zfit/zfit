from __future__ import print_function, division, absolute_import

import tensorflow_probability as tfp

from zfit.core.basepdf import WrapDistribution


class Normal(WrapDistribution):
    def __init__(self, loc, scale, name="Normal"):
        distribution = tfp.distributions.Normal(loc=loc, scale=scale, name=name + "_tf")
        super(Normal, self).__init__(distribution=distribution, name=name)
