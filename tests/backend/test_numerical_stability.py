#  Copyright (c) 2024 zfit

import zfit.z.numpy as znp

def test_type_casting():

    one = 1.0
    oneplus = one + 1e-15

    assert one != oneplus, "Test is flawed"
    assert znp.asarray(one) != znp.asarray(oneplus)
    assert znp.asarray(one, znp.float32) == znp.asarray(oneplus, znp.float32)
