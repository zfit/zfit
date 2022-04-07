#  Copyright (c) 2022 zfit

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TemporarilySet:
    def __init__(self, value: Any, setter: Callable, getter: Callable):
        """Temporarily set `value` with `setter` and reset to the old value after leaving the context.

         This class can be used to have a setter that can permanently set a value *as well as* just
         for the time inside a context manager. The usage is as follows:

         >>> class SimpleX:
         >>>     def __init__(self):
         >>>          self.x = None
         >>>     def _set_x(self, x):
         >>>         self.x = x
         >>>     def get_x(self):
         >>>         return self.x
         >>>     def set_x(self, x):
         >>>         return TemporarilySet(value=x, setter=self._set_x, getter=self.get_x)

         >>> simple_x = SimpleX()
         Now we can either set x permanently
         >>> simple_x.set_x(42)
         >>> print(simple.x)
         42

         or temporarily
         >>> with simple_x.set_x(13) as value:
         >>>     print("Value from contextmanager:", value)
         >>>     print("simple_x.get_x():", simple_x.get_x())
         13
         13

         and is afterwards unset again
         >>> print(simple.x)
         42

        Args:
            value: The value to be (temporarily) set (and returned if a context manager is applied).
            setter: The setter function with a signature that is compatible to the call:
                `setter(value, *setter_args, **setter_kwargs)`
            getter: The getter function with a signature that is compatible to the call:
                `getter(*getter_args, **getter_kwargs)`
        """
        self.setter = setter
        self.getter = getter
        self.value = value

        self.old_value = self.getter()
        self.setter(self.value)

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.setter(self.old_value)
