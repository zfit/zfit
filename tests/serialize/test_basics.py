#  Copyright (c) 2022 zfit
import copy
import pprint

import pytest
from frozendict import frozendict
import numpy as np

import zfit


def test_serial_space():
    import zfit

    space = zfit.Space("x", (-1, 1))
    json1 = space.to_json()
    print(json1)
    space2 = zfit.Space.get_repr().parse_raw(json1).to_orm()
    print(space2)
    assert space == space2
    json2 = space2.to_json()
    assert json1 == json2
    space3 = zfit.Space.get_repr().parse_raw(json2).to_orm()
    assert space2 == space3
    assert space == space3


def test_serial_param():
    import zfit

    param = zfit.Parameter("mu", 0.1, -1, 1)

    json1 = param.to_json()
    print(json1)
    param2 = zfit.Parameter.get_repr().parse_raw(json1).to_orm()
    print(param2)
    assert param == param2
    json2 = param2.to_json()
    assert json1 == json2
    param3 = zfit.Parameter.get_repr().parse_raw(json2).to_orm()
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
    gauss2_raw = zfit.pdf.Gauss.get_repr().parse_raw(json1)
    print("gauss2_raw", gauss2_raw)
    gauss2 = gauss2_raw.to_orm()
    print(gauss2)
    assert str(gauss) == str(gauss2)
    json2 = gauss2.to_json()
    assert json1 == json2
    gauss3 = zfit.pdf.Gauss.get_repr().parse_raw(json2).to_orm()
    x = znp.linspace(-3, 3, 100)
    assert np.allclose(gauss.pdf(x), gauss3.pdf(x))
    assert np.allclose(gauss2.pdf(x), gauss3.pdf(x))
    mu.set_value(0.6)
    assert np.allclose(gauss.pdf(x), gauss3.pdf(x))
    assert np.allclose(gauss2.pdf(x), gauss3.pdf(x))


def gauss():
    import zfit
    mu = zfit.Parameter("mu", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)


def cauchy():
    import zfit
    m = zfit.Parameter("m", 0.1, -1, 1)
    gamma = zfit.Parameter("gamma", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.Cauchy(m=m, gamma=gamma, obs=obs)


def exponential():
    import zfit
    lam = zfit.Parameter("lambda", 0.1, -1, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.Exponential(lam=lam, obs=obs)


def crystalball():
    import zfit
    alpha = zfit.Parameter("alpha", 0.1, -1, 1)
    n = zfit.Parameter("n", 0.1, 0, 1)
    mu = zfit.Parameter("mu", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.CrystalBall(alpha=alpha, n=n, mu=mu, sigma=sigma, obs=obs)


def doublecb():
    import zfit
    alphaL = zfit.Parameter("alphaL", 0.1, -1, 1)
    nL = zfit.Parameter("nL", 0.1, 0, 1)
    alphaR = zfit.Parameter("alphaR", 0.1, -1, 1)
    nR = zfit.Parameter("nR", 0.1, 0, 1)
    mu = zfit.Parameter("mu", 0.1, -1, 1)
    sigma = zfit.Parameter("sigma", 0.1, 0, 1)
    obs = zfit.Space("obs", (-3, 3))
    return zfit.pdf.DoubleCB(alphaL=alphaL, nL=nL, alphaR=alphaR, nR=nR, mu=mu, sigma=sigma, obs=obs)


@pytest.mark.parametrize("pdf", [gauss, cauchy, exponential, crystalball])
def test_serial_hs3_gauss(pdf):
    import zfit
    import zfit.z.numpy as znp
    import zfit.serialization as zserial

    pdf = pdf()

    hs3json = zserial.Serializer.to_hs3(pdf)
    pprint.pprint(hs3json)
    loaded = zserial.Serializer.from_hs3(hs3json)
    pprint.pprint(loaded)
    loaded_pdf = list(loaded['pdfs'].values())[0]
    assert str(pdf) == str(loaded_pdf)
    x = znp.linspace(-3, 3, 100)
    assert np.allclose(pdf.pdf(x), loaded_pdf.pdf(x))


def test_replace_matching():
    import zfit
    import zfit.serialization as zserial
    original_dict = {'metadata': {'HS3': {'version': 'experimental'},
                                  'serializer': {'lib': 'zfit',
                                                 'version': '0.10.2.dev9+gfc70df47'}},
                     'pdfs': {'Gauss': {'mu': {'floating': True,
                                               'max': 1.0,
                                               'min': -1.0,
                                               'name': 'mu',
                                               'step_size': 0.001,
                                               'type': 'Parameter',
                                               'value': 0.10000000149011612},
                                        'sigma': {'floating': True,
                                                  'max': 1.0,
                                                  'min': 0.0,
                                                  'name': 'sigma',
                                                  'step_size': 0.001,
                                                  'type': 'Parameter',
                                                  'value': 0.10000000149011612},
                                        'type': 'Gauss',
                                        'x': {'max': 3.0,
                                              'min': -3.0,
                                              'name': 'obs',
                                              'type': 'Space'}}},
                     'variables': {'mu': {'floating': True,
                                          'max': 1.0,
                                          'min': -1.0,
                                          'name': 'mu',
                                          'step_size': 0.001,
                                          'type': 'Parameter',
                                          'value': 0.10000000149011612},
                                   'obs': {'max': 3.0, 'min': -3.0, 'name': 'obs', 'type': 'Space'},
                                   'sigma': {'floating': True,
                                             'max': 1.0,
                                             'min': 0.0,
                                             'name': 'sigma',
                                             'step_size': 0.001,
                                             'type': 'Parameter',
                                             'value': 0.10000000149011612}}}

    target_dict = {'metadata': {'HS3': {'version': 'experimental'},
                                'serializer': {'lib': 'zfit',
                                               'version': '0.10.2.dev9+gfc70df47'}},
                   'pdfs': {'Gauss': {'mu': 'mu',
                                      'sigma': 'sigma',
                                      'type': 'Gauss',
                                      'x': 'obs'}},
                   'variables': {'mu': {'floating': True,
                                        'max': 1.0,
                                        'min': -1.0,
                                        'name': 'mu',
                                        'step_size': 0.001,
                                        'type': 'Parameter',
                                        'value': 0.10000000149011612},
                                 'obs': {'max': 3.0, 'min': -3.0, 'name': 'obs', 'type': 'Space'},
                                 'sigma': {'floating': True,
                                           'max': 1.0,
                                           'min': 0.0,
                                           'name': 'sigma',
                                           'step_size': 0.001,
                                           'type': 'Parameter',
                                           'value': 0.10000000149011612}}}

    parameter = frozendict({'name': None, 'min': None, 'max': None})
    replace_forward = {parameter: lambda x: x['name']}
    target_test = copy.deepcopy(original_dict)
    target_test['pdfs'] = zserial.serializer.replace_matching(target_test['pdfs'], replace_forward)
    pprint.pprint(target_test)
    assert target_test == target_dict
    replace_backward = {k: lambda x=k: target_dict['variables'][x] for k in target_dict['variables'].keys()}
    original_test = copy.deepcopy(target_dict)
    original_test['pdfs'] = zserial.serializer.replace_matching(original_test['pdfs'], replace_backward)
    pprint.pprint(original_test)
    assert original_test == original_dict
