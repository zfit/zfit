from __future__ import print_function, division, absolute_import

import tensorflow as tf


def beta(q2, ml):
    return tf.sqrt(1.0 - (4. * tf.square(ml) / q2))


def Lambda(ma2, mb2, mc2):
    return (tf.square(ma2) + tf.square(mb2) + tf.square(mc2)
            - 2.0 * ma2 * (mb2 + mc2) - 2.0 * mc2 * mb2)
