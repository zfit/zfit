#  Copyright (c) 2022 zfit
import pprint

import numpy as np


def test_serial_space():
    import zfit

    space = zfit.Space("x", (-1, 1))
    json1 = space.to_json()
    print(json1)
    space2 = zfit.Space.repr.parse_raw(json1).to_orm()
    print(space2)
    assert space == space2
    json2 = space2.to_json()
    assert json1 == json2
    space3 = zfit.Space.repr.parse_raw(json2).to_orm()
    assert space2 == space3
    assert space == space3


def test_serial_param():
    import zfit

    param = zfit.Parameter("mu", 0.1, -1, 1)

    json1 = param.to_json()
    print(json1)
    param2 = zfit.Parameter.repr.parse_raw(json1).to_orm()
    print(param2)
    assert param == param2
    json2 = param2.to_json()
    assert json1 == json2
    param3 = zfit.Parameter.repr.parse_raw(json2).to_orm()
    assert param2 == param3
    assert param == param3


def test_serial_gauss():
    import zfit
    import zfit.z.numpy as znp

    mu = zfit.Parameter("mu", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    json1 = gauss.to_json()
    print("json1", json1)
    gauss2_raw = zfit.pdf.Gauss.repr.parse_raw(json1)
    print("gauss2_raw", gauss2_raw)
    gauss2 = gauss2_raw.to_orm()
    print(gauss2)
    assert str(gauss) == str(gauss2)
    json2 = gauss2.to_json()
    assert json1 == json2
    gauss3 = zfit.pdf.Gauss.repr.parse_raw(json2).to_orm()
    x = znp.linspace(-3, 3, 100)
    assert np.allclose(gauss.pdf(x), gauss3.pdf(x))
    assert np.allclose(gauss2.pdf(x), gauss3.pdf(x))
    mu.set_value(0.6)
    assert np.allclose(gauss.pdf(x), gauss3.pdf(x))
    assert np.allclose(gauss2.pdf(x), gauss3.pdf(x))


def test_serial_hs3_gauss():
    import zfit
    import zfit.z.numpy as znp
    import zfit.serialization as zserial

    mu = zfit.Parameter("mu", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    hs3json = zserial.Serializer.to_hs3(gauss)
    pprint.pprint(hs3json)
