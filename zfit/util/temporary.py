from typing import Callable, Any


class TemporarilySet:

    def __init__(self, value: Any, setter: Callable, getter: Callable, setter_args=None, setter_kwargs=None,
                 getter_args=None, getter_kwargs=None):
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
            value (Any): The value to be (temporarily) set (and returned if a context manager is applied).
            setter (Callable): The setter function with a signature that is compatible to the call:
                `setter(value, *setter_args, **setter_kwargs)`
            getter (Callable): The getter function with a signature that is compatible to the call:
                `getter(*getter_args, **getter_kwargs)`
            setter_args (List): A list of arguments given to the setter
            setter_kwargs (Dict): A dict of keyword-arguments given to the setter
            getter_args (List): A list of arguments given to the getter
            getter_kwargs (Dict): A dict of keyword-arguments given to the getter
         """
        self.setter = setter
        self.setter_args = [] if setter_args is None else setter_args
        self.setter_kwargs = {} if setter_kwargs is None else setter_kwargs
        self.getter_args = [] if getter_args is None else getter_args
        self.getter_kwargs = {} if getter_kwargs is None else getter_kwargs
        self.getter = getter
        self.value = value

        self.old_value = self.getter(*self.getter_args, **self.getter_kwargs)
        self.setter(self.value, *self.setter_args, **self.setter_kwargs)

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.setter(self.old_value, *self.setter_args, **self.setter_kwargs)
