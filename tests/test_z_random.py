#  Copyright (c) 2022 zfit

import numpy as np
import pytest
import tensorflow as tf

from zfit import z


@pytest.mark.flaky(2)
def test_counts_multinomial():
    probs = [0.1, 0.3, 0.6]
    total_count = 1000.0
    total_count_var = tf.Variable(total_count, trainable=False)
    counts_np = [
        z.random.counts_multinomial(total_count=total_count_var, probs=probs).numpy()
        for _ in range(20)
    ]
    mean = np.mean(counts_np, axis=0)
    std = np.std(counts_np, axis=0)

    assert (3,) == counts_np[0].shape
    assert all(total_count == np.sum(counts_np, axis=1))
    probs_scaled = np.array(probs) * total_count
    assert probs_scaled == pytest.approx(mean, rel=0.05)
    assert [9, 14, 16] == pytest.approx(std, rel=0.5)  # flaky

    total_count2 = 5000
    counts_np2 = [
        z.random.counts_multinomial(total_count=total_count2, probs=probs).numpy()
        for _ in range(3)
    ]
    mean2 = np.mean(counts_np2, axis=0)

    assert all(total_count2 == np.sum(counts_np2, axis=1))
    probs_scaled2 = np.array(probs) * total_count2
    assert pytest.approx(mean2, rel=0.15) == probs_scaled2

    total_count3 = 3000.0

    counts_np3 = [
        z.random.counts_multinomial(
            total_count=total_count3,  # sigmoid: inverse of logits (softmax)
            logits=probs * 30,
        ).numpy()
        for _ in range(5)
    ]

    assert all(total_count3 == np.sum(counts_np3, axis=1))
