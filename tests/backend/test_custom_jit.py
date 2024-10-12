#  Copyright (c) 2024 zfit
import zfit


def test_donotjit():
    from zfit import z
    import zfit.z.numpy as znp
    nraises = 0

    @z.function()
    def jittedfunc(x):
        if not zfit.run.executing_eagerly():
            nonlocal nraises
            nraises += 1
            assert nraises == 1
            raise z.DoNotCompile
        else:
            _ = bool(x > 0)  # sanity check, this will raise an error if it's compiled!
            return x ** 2

    assert jittedfunc(znp.array(2)) == 2 ** 2
    assert nraises == 1

def test_donotjit_decorated():
    from zfit import z
    import zfit.z.numpy as znp
    nraises = 0

    @z.function(force_eager=True)
    def nonjitfunc(x):
        assert zfit.run.executing_eagerly()
        return x ** 3


    @z.function()
    def jittedfunc(x):
        nonlocal nraises
        nraises += 1
        return nonjitfunc(x)

    assert jittedfunc(znp.array(2)) == 2 ** 3
    assert nraises == 2
    assert jittedfunc(znp.array(4)) == 4 ** 3
    assert nraises == 3  # it has learnt that it should not jit
