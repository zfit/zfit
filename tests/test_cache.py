#  Copyright (c) 2020 zfit
import pytest

import zfit
from zfit import z
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.util.cache import Cachable, invalidates_cache


# class Test1(Cachable):
#
#     def value(self):
#         value = self._cache.get("value")
#         if value is None:
#             self._cache['value'] = np.random.random()
#         return self._cache['value']
#
#     @invalidates_cache
#     def change_param(self, new_param):
#         # invalidates cache
#         return None
#
#
# class MotherTest1(Cachable):
#
#     def __init__(self, test1, test2):
#         super().__init__()
#         self.add_cache_dependents([test1, test2])
#         self.test1 = test1
#         self.test2 = test2
#
#     def mother_value(self):
#         value = self._cache.get("mother_value")
#         if value is None:
#             value = self.test1.value() + self.test2.value()
#         self._cache['mother_value'] = value
#         return value
#
#     @invalidates_cache
#     def change_param(self):
#         return None
#
#
# def test_simple_cache():
#     test1 = Test1()
#     assert test1.value() == test1.value()
#     value1 = test1.value()
#     test1.change_param(24)
#     assert value1 != test1.value()
#
#
# def test_mother_cache():
#     test1, test2 = Test1(), Test1()
#     mother_test = MotherTest1(test1, test2)
#     assert mother_test.mother_value() == mother_test.mother_value()
#     mother_value = mother_test.mother_value()
#     test1.change_param(12)
#     assert mother_test.mother_value() != mother_value
#     assert mother_test.mother_value() == mother_test.mother_value()

class GraphCreator1(Cachable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 42
        self.retrace_runs = 0

    @z.function
    def calc(self, x):
        self.retrace_runs += 1
        return x + self.value

    @invalidates_cache
    def change_value(self, value):
        self.value = value

    def change_value_no_invalidation(self, value):
        self.value = value


@pytest.mark.skipif(not zfit.z.zextension.FunctionWrapperRegistry.do_jit,
                    reason="no caching in eager mode expected")  # currently, importance sampling is not working, odd deadlock in TF
def test_graph_cache():
    graph1 = GraphCreator1()
    initial = 42
    add = 5
    new_value = 8
    result = 47
    assert graph1.calc(add).numpy() == result
    assert graph1.retrace_runs > 0  # simple
    graph1.retrace_runs = 0  # reset
    assert graph1.calc(add).numpy() == result
    assert graph1.retrace_runs == 0  # no retracing must have occurred

    graph1.change_value_no_invalidation(10)
    assert graph1.calc(add).numpy() == result
    assert graph1.retrace_runs == 0  # no retracing must have occurred
    graph1.change_value(new_value)
    assert graph1.calc(add).numpy() == new_value + add
    assert graph1.retrace_runs > 0
    graph1.retrace_runs = 0  # reset

    graph1.change_value_no_invalidation(10)
    assert graph1.calc(add).numpy() == new_value + add
    assert graph1.retrace_runs == 0  # no retracing must have occurred
