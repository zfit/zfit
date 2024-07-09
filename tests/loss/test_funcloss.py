#  Copyright (c) 2024 zfit
import matplotlib.pyplot as plt
import numpy as np
import pytest

import zfit
import zfit.z.numpy as znp
from zfit.core.interfaces import ZfitData

@pytest.mark.parametrize("uncertainty", [True, False])
def test_funcloss_base(uncertainty):
    class Polynomial(zfit.core.basefunc.BaseFuncV2):
        def __init__(self, obs, output, params=None):
            super().__init__(obs=obs, output=output, params=params)

        @zfit.supports()
        def _func(self, x: ZfitData, params: dict):
            params = self.params
            x0 = x[0]
            x1 = x[1]
            x2 = x[2]
            c0 = params['c0']
            c1 = params['c1']
            c2 = params['c2']
            c3 = params['c3']
            c4 = params['c4']
            y1 = c0 + c1 * x0 + c2 * x1 + c3 * x2 + c4 * x0 * x1 + c3 * x1 * x2
            y2 = c4 * x0 ** 2 + c1 * x1 ** 2 + c2 * x2 ** 2
            return znp.stack([y1, y2], axis=-1)

    obs = zfit.Space('obs1', -1, 1) * zfit.Space('obs2', -1, 1) * zfit.Space('obs3', -1, 1)
    params = {f'c{i}': zfit.Parameter(f'c{i}', 1) for i in range(5)}
    output = zfit.Space('output1', -1e10, 1e10) * zfit.Space("output2", -1e10, 1e10)
    poly = Polynomial(obs=obs, output=output, params=params)
    inputshape = (500, obs.n_obs + output.n_obs + int(uncertainty))  # sigma
    datadims = obs * output
    sigma = None
    if uncertainty:
        sigma = zfit.Space("uncertainty", -1e10, 1e10)
        datadims *= sigma
    data = zfit.Data(obs=datadims,
                     data=(np.sort(np.random.uniform(size=inputshape), axis=0) + np.random.uniform(size=inputshape) * 0.1) ** 2)
    pred = poly(data, params=params)
    print(pred)
    loss = zfit.loss.Chi2(func=poly, data=data, uncertainty=sigma)
    print(loss.value())
    loss.errordef = 0.5
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=loss, params=loss.get_params())
    predfitted = poly(data, params=params)
    print(result)
    # plt.figure()
    # plt.plot(data['obs1'], data[output], '.', label='Data')
    # plt.plot(data['obs1'], predfitted[output], '.', label='fitted')
    # plt.legend()
    # plt.show()
