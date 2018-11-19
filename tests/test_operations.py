import tensorflow as tf

from zfit import Parameter, ztf
from zfit.core.parameter import ComposedParameter
from zfit.pdfs.functions import SimpleFunction


def test_composed_param():
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    param_a = ComposedParameter('param_a', tensor=a)
    assert isinstance(param_a.get_dependents(only_floating=True), set)
    assert param_a.get_dependents(only_floating=True) == {param1, param2}
    assert param_a.get_dependents(only_floating=False) == {param1, param2, param3}

def test_param_func():
    param1 = Parameter('param1', 1.)
    param2 = Parameter('param2', 2.)
    param3 = Parameter('param3', 3., floating=False)
    param4 = Parameter('param4', 4.)
    a = ztf.log(3. * param1) * tf.square(param2) - param3
    func = SimpleFunction(func=lambda x: x*a)

    new_func = param4 * func

