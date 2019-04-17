#  Copyright (c) 2019 zfit

from zfit.core.testing import setup_function, teardown_function

import pytest
import numpy as np
import tensorflow as tf

import zfit
from zfit import ztf


@pytest.mark.flaky(2)
def test_counts_multinomial():
    probs = [0.1, 0.3, 0.6]
    total_count = 1000.
    total_count_var = tf.Variable(total_count, use_resource=True, trainable=False)
    zfit.run(total_count_var.initializer)
    counts = ztf.random.counts_multinomial(total_count=total_count_var, probs=probs)

    counts_np = [zfit.run(counts) for _ in range(20)]
    mean = np.mean(counts_np, axis=0)
    std = np.std(counts_np, axis=0)

    assert (3,) == counts_np[0].shape
    assert np.sum(counts_np, axis=1) == pytest.approx(total_count, rel=0.05)
    assert np.array(probs) * total_count == pytest.approx(mean, rel=0.05)
    assert [9, 14, 16] == pytest.approx(std, rel=0.5)  # flaky

    total_count2 = 5000
    total_count_var.load(total_count2, session=zfit.run.sess)
    counts_np2 = [zfit.run(counts) for _ in range(3)]
    mean2 = np.mean(counts_np2, axis=0)

    assert pytest.approx(total_count2, rel=0.05) == np.sum(counts_np2, axis=1)
    assert pytest.approx(mean2, rel=0.05) == np.array(probs) * total_count2

    total_count3 = 3000.
    counts3 = ztf.random.counts_multinomial(total_count=total_count3, logits=np.array(probs) * 30)

    counts_np3 = [zfit.run(counts3) for _ in range(5)]
    mean3 = np.mean(counts_np3, axis=0)

    assert pytest.approx(total_count3, rel=0.05) == np.sum(counts_np3, axis=1)
    assert pytest.approx(mean3, rel=0.05) == np.array(probs) * total_count3
