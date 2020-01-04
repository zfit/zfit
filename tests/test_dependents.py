#  Copyright (c) 2020 zfit

import zfit

# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester


def test_get_dependents_ordered():
    params = [zfit.Parameter(f'param{i}', i) for i in range(4)]
    obs = zfit.Space('obs1', (-3, 2))

    def create_pdf(params):
        param1, param2, param3, param4 = params
        gauss1 = zfit.pdf.Gauss(param1, param2, obs=obs)
        gauss2 = zfit.pdf.Gauss(param2, param2, obs=obs)
        gauss3 = zfit.pdf.Gauss(param2, param4, obs=obs)
        gauss4 = zfit.pdf.Gauss(param3, param1, obs=obs)
        sum_pdf = zfit.pdf.SumPDF([gauss1, gauss2, gauss3, gauss4], fracs=[0.1, 0.3, 0.4])
        return sum_pdf

    is_different = 0
    shuffled_param_list = params, params[::-1]
    # if error below, the test is at flaw
    params[0] == params[1]
    assert ((list(shuffled_param_list[0]) != list(set(shuffled_param_list[0]))) or
            (list(shuffled_param_list[1]) != list(set(shuffled_param_list[1]))))  # check that set is different
    for params_shuffled in shuffled_param_list:
        pdf = create_pdf(params_shuffled)
        deps = pdf.get_dependents()
        is_different += list(deps) != list(set(deps))

    assert is_different > 0, "set and OrderedSet return the same order, cannot be."
