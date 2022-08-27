#  Copyright (c) 2022 zfit
import numpy as np
import pytest

import zfit
from zfit import z
from zfit.util.cache import GraphCachable, clear_graph_cache, invalidate_graph
from zfit.z.zextension import FunctionWrapperRegistry


class Example1(GraphCachable):
    def value(self):
        value = self._cache.get("value")
        if value is None:
            self._cache["value"] = np.random.random()
        return self._cache["value"]

    @invalidate_graph
    def change_param(self, new_param):
        # invalidates cache
        return None


class MotherExample1(GraphCachable):
    def __init__(self, test1, test2):
        super().__init__()
        self.add_cache_deps(
            [test1, test2],
        )
        self.test1 = test1
        self.test2 = test2

    def mother_value(self):
        value = self._cache.get("mother_value")
        if value is None:
            value = self.test1.value() + self.test2.value()
        self._cache["mother_value"] = value
        return value

    @invalidate_graph
    def change_param(self):
        return None


def test_simple_cache():
    test1 = Example1()
    assert test1.value() == test1.value()
    value1 = test1.value()
    test1.change_param(24)
    assert value1 != test1.value()


def test_mother_cache():
    test1, test2 = Example1(), Example1()
    mother_test = MotherExample1(test1, test2)
    assert mother_test.mother_value() == mother_test.mother_value()
    mother_value = mother_test.mother_value()
    test1.change_param(12)
    assert mother_test.mother_value() != mother_value
    assert mother_test.mother_value() == mother_test.mother_value()


CONST = 40


class GraphCreator1(GraphCachable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 42
        self.retrace_runs = 0

    @z.function
    def calc(self, x):
        self.retrace_runs += 1
        return x + self.value + CONST

    @z.function(wraps="tensor")
    def calc_variable(self, x):
        return x + self.value + CONST

    def calc_no_cache(self, x):
        self.retrace_runs += 1
        return x + self.value + CONST

    @invalidate_graph
    def change_value(self, value):
        self.value = value

    def change_value_no_invalidation(self, value):
        self.value = value


class GraphCreator2(GraphCreator1):
    @z.function()
    def calc(self, x):
        self.retrace_runs += 1
        return x + self.value + CONST


graph_creators = [
    GraphCreator2,
    GraphCreator1,
]


@pytest.mark.skipif(
    zfit.run.mode["graph"] is False, reason="no caching in eager mode expected"
)
@pytest.mark.parametrize("graph_holder", graph_creators)
def test_graph_cache(graph_holder):
    graph1 = graph_holder()
    global CONST
    CONST = 40
    add = 5
    new_value = 8
    result = 47 + CONST
    assert FunctionWrapperRegistry.do_jit_types["tensor"]  # should be true by default
    assert graph1.calc(add).numpy() == result
    assert graph1.calc_variable(add).numpy() == result
    assert graph1.retrace_runs > 0  # simple
    graph1.retrace_runs = 0  # reset
    assert graph1.calc(add).numpy() == result
    assert graph1.calc_variable(add).numpy() == result
    assert graph1.retrace_runs == 0  # no retracing must have occurred

    graph1.change_value_no_invalidation(10)
    assert graph1.calc(add).numpy() == result
    assert graph1.calc_variable(add).numpy() == result
    assert graph1.retrace_runs == 0  # no retracing must have occurred
    FunctionWrapperRegistry.do_jit_types["tensor"] = False
    assert graph1.calc_variable(add) == 10 + add + CONST
    FunctionWrapperRegistry.do_jit_types["tensor"] = True
    graph1.change_value(new_value)
    assert graph1.calc(add).numpy() == new_value + add + CONST
    assert graph1.calc_variable(add).numpy() == new_value + add + CONST
    assert graph1.retrace_runs > 0
    CONST = 50
    assert graph1.calc(add).numpy() == new_value + add + 40  # old const
    assert graph1.calc_variable(add).numpy() == new_value + add + 40  # old const
    clear_graph_cache()
    FunctionWrapperRegistry.do_jit_types["something"] = False
    assert graph1.calc_no_cache(add) == new_value + add + CONST
    assert graph1.calc_variable(add) == new_value + add + CONST
    assert graph1.calc(add).numpy() == new_value + add + CONST
    graph1.retrace_runs = 0  # reset

    graph1.change_value_no_invalidation(10)
    assert graph1.calc(add).numpy() == new_value + add + CONST
    assert graph1.retrace_runs == 0  # no retracing must have occurred
    CONST = 40
    FunctionWrapperRegistry.do_jit_types[
        "something"
    ] = True  # should be true by default
