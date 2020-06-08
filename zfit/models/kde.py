#  Copyright (c) 2020 zfit


import tensorflow as tf

from .. import z, ztypes


def bandwidth_rule_of_thumb(data, factor=0.9):
    return tf.math.reduce_std(data) * tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.) * factor


def bandwidth_silverman(data):
    return bandwidth_rule_of_thumb(data=data, factor=0.9)


def bandwidth_scott(data):
    return bandwidth_rule_of_thumb(data=data, factor=1.059)


def bandwidth_adaptiveV1(data, bandwidth, func):
    from .. import run
    run.assert_executing_eagerly()
    data = z.convert_to_tensor(data)
    # bandwidth = z.convert_to_tensor(bandwidth)
    bandwidth = z.sqrt(tf.math.reduce_std(data) / func(data))
    bandwidth *= tf.cast(tf.shape(data)[0], ztypes.float) ** (-1 / 5.) * 1.059
    return bandwidth


if __name__ == '__main__':
    data = z.random.normal(shape=(100,))
    data = tf.sort(data)
    bandwidth = bandwidth_rule_of_thumb(data=data)


    def kde_func(data):
        return z.reduce_sum(gauss(data[:, None], bandwidth, data),
                            axis=-1) / tf.cast(tf.shape(data)[0], ztypes.float)


    def gauss(x, bandwidth, data):
        return (1. / bandwidth) * z.exp(-0.5 * ((data - x) ** 2 / bandwidth ** 2))


    bw_adapt = bandwidth_adaptiveV1(data, bandwidth, kde_func)
    print(bandwidth)
    print(bw_adapt)
