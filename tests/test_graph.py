#  Copyright (c) 2022 zfit
import pytest
import tensorflow as tf

import zfit
from zfit import z


@pytest.mark.skipif(
    zfit.run.get_graph_mode() is False,
    reason="If not using graph mode, we cannot test if graphs work or not.",
)
@pytest.mark.parametrize("cachesize", [1, 10, 1000, None])
def test_modes(cachesize):
    counts = 0

    zfit.run.set_graph_cache_size(cachesize)
    if cachesize is not None:
        assert all(
            register.cachesize == cachesize
            for register in zfit.z.zextension.FunctionWrapperRegistry.registries
        )

    @z.function(wraps="42")
    def func(x):
        nonlocal counts
        counts += 1
        return tf.random.uniform(shape=()) * x

    func(5)
    func(5)
    func(5)
    func(5)
    assert counts == 1
    zfit.run.set_graph_mode(False)
    func(5)
    assert counts == 2
    zfit.run.set_mode_default()
    func(5)
    func(5)
    func(5)
    assert counts == 2
    zfit.run.clear_graph_cache()
    func(5)
    func(5)
    func(5)
    assert counts == 3

    # use with everywhere in orther to reset it correctly
    with zfit.run.set_autograd_mode(False):
        assert zfit.settings.options.numerical_grad
        with zfit.run.set_mode_default():
            assert not zfit.settings.options.numerical_grad

            with zfit.run.set_graph_mode(True):
                assert zfit.run.get_graph_mode()
                with zfit.run.set_graph_mode(False):
                    assert not zfit.run.get_graph_mode()
                assert zfit.run.get_graph_mode()

            with zfit.run.set_autograd_mode(True):
                assert zfit.run.get_autograd_mode()
                with zfit.run.set_autograd_mode(False):
                    assert not zfit.run.get_autograd_mode()
                assert zfit.run.get_autograd_mode()
