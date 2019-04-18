#  Copyright (c) 2019 zfit

from zfit.core.testing import setup_function, teardown_function, tester



from zfit.util.cache import Cachable, invalidates_cache

import numpy as np

from zfit.core.testing import setup_function, teardown_function, tester


class Test1(Cachable):

    def value(self):
        value = self._cache.get("value")
        if value is None:
            self._cache['value'] = np.random.random()
        return self._cache['value']

    @invalidates_cache
    def change_param(self, new_param):
        # invalidates cache
        return None


class MotherTest1(Cachable):

    def __init__(self, test1, test2):
        super().__init__()
        self.add_cache_dependents([test1, test2])
        self.test1 = test1
        self.test2 = test2

    def mother_value(self):
        value = self._cache.get("mother_value")
        if value is None:
            value = self.test1.value() + self.test2.value()
        self._cache['mother_value'] = value
        return value

    @invalidates_cache
    def change_param(self):
        return None


def test_simple_cache():
    test1 = Test1()
    assert test1.value() == test1.value()
    value1 = test1.value()
    test1.change_param(24)
    assert value1 != test1.value()


def test_mother_cache():
    test1, test2 = Test1(), Test1()
    mother_test = MotherTest1(test1, test2)
    assert mother_test.mother_value() == mother_test.mother_value()
    mother_value = mother_test.mother_value()
    test1.change_param(12)
    assert mother_test.mother_value() != mother_value
    assert mother_test.mother_value() == mother_test.mother_value()
