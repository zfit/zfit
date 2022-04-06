#  Copyright (c) 2022 zfit


from zfit.util.temporary import TemporarilySet


def test_simple_x():
    class SimpleX:
        def __init__(self):
            self.x = None

        def _set_x(self, x):
            self.x = x

        def get_x(self):
            return self.x

        def set_x(self, x):
            return TemporarilySet(value=x, setter=self._set_x, getter=self.get_x)

    simple_x = SimpleX()
    simple_x.set_x(42)
    assert simple_x.get_x() == 42
    list1 = [1, 5]
    with simple_x.set_x(list1) as value:
        assert value == list1
        assert simple_x.get_x() == list1
    assert simple_x.get_x() == 42
