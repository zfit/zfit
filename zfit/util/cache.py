"""Module for caching.

The basic concept of caching in Zfit builds on a "cacher", that caches a certain value and that
is dependent of "cache_dependents". By implementing `ZfitCachable`, an object will be able to play both
roles. And most importantly, it has a `_cache` dict, that contains all the cache.

Basic principle
===============

A "cacher" adds any dependents that it may comes across with `add_cache_dependents`. For example,
for a loss this would be all pdfs and data. Since :py:class:`~zfit.Space` is immutable, there is no need to add this
as a dependent. This leads to the "cache_dependent" to register the "cacher" and to remember it.

In case, any "cache_dependent" changes in a way the cache of itself (and any "cacher") is invalid,
which is done in the simplest case by decorating a method with `@invalidates_cache`, the "cache_dependent":

 * clears it's own cache with `reset_cache_self` and
 * "clears" any "cacher"s cache with `reset_cache(reseter=self)`, telling the "cacher" that it should
   reset the cache. This is also where more fine-grained control (depending on which "cache_dependent"
   calls `reset_cache`) can be brought into play.

Example with a pdf that caches the normalization:

.. code:: python

    class Parameter(Cachable):
        def load(new_value):  # does not require to build a new graph
            # do something

        @invalidates_cache
        def change_limits(new_limits):  # requires to build a new graph (as an example)
            # do something

    # create param1, param2 from `Parameter`

    class MyPDF(Cachable):
        def __init__(self, param1, param2):
            self.add_cache_dependents([param1, param2])

        def cached_func(...):
            if self._cache.get('my_name') is None:
                result = ...  # calculations here
                self._cache['my_name']
            else:
                result = self._cache['my_name']
            return result
"""

#  Copyright (c) 2021 zfit

import functools
import weakref
from abc import abstractmethod
from typing import Iterable, Mapping, Union

import numpy as np
import tensorflow as tf

from . import ztyping
from .container import convert_to_container


class ZfitGraphCachable:

    @abstractmethod
    def register_cacher(self, cacher: "ZfitGraphCachable"):
        raise NotImplementedError

    @abstractmethod
    def add_cache_deps(self, cache_dependents, allow_non_cachable):
        """Add dependents that render the cache invalid if they change.

        Args:
            cache_dependents:
            allow_non_cachable: If `True`, allow `cache_dependents` to be non-cachables.
                If `False`, any `cache_dependents` that is not a `ZfitCachable` will raise an error.

        Raises:
            TypeError: if one of the `cache_dependents` is not a `ZfitCachable` _and_ `allow_non_cachable`
                if `False`.
        """
        pass

    @abstractmethod
    def reset_cache_self(self):
        """Clear the cache of self and all dependent cachers."""
        pass

    @abstractmethod
    def reset_cache(self, reseter):
        pass


class GraphCachable(ZfitGraphCachable):
    graph_caching_methods = []
    instances = weakref.WeakSet()

    def __init__(self, *args, **kwargs):
        self._cache = {}
        self._cachers = weakref.WeakKeyDictionary()
        self.reset_cache_self()
        self.instances.add(self)
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        graph_caching_methods = []
        for func_name in dir(cls):
            if not func_name.startswith("__"):
                func = getattr(cls, func_name)
                if callable(func) and hasattr(func, 'zfit_graph_cache_registered'):
                    # assert hasattr(func, "_descriptor_cache"), "TensorFlow internals have changed. Need to update cache"
                    func.zfit_graph_cache_registered = True
                    graph_caching_methods.append(func)
        cls.graph_caching_methods = graph_caching_methods

    def register_cacher(self, cacher: ztyping.CacherOrCachersType):
        """Register a `cacher` that caches values produces by this instance; a dependent.

        Args:
            cacher:
        """
        if not isinstance(cacher, ZfitGraphCachable):
            raise TypeError(f"`cacher` is not a `ZfitCachable` but {type(cacher)}")
        if not cacher in self._cachers:
            self._cachers[cacher] = None  # could we have a more useful value?

    def add_cache_deps(self, cache_deps: ztyping.CacherOrCachersType, allow_non_cachable: bool = True):
        """Add dependencies that render the cache invalid if they change.

        Args:
            cache_deps:
            allow_non_cachable: If `True`, allow `cache_dependents` to be non-cachables.
                If `False`, any `cache_dependents` that is not a `ZfitCachable` will raise an error.

        Raises:
            TypeError: if one of the `cache_dependents` is not a `ZfitCachable` _and_ `allow_non_cachable`
                if `False`.
        """
        cache_deps = convert_to_container(cache_deps)
        for cache_dep in cache_deps:
            if isinstance(cache_dep, ZfitGraphCachable):
                cache_dep.register_cacher(self)
            elif not allow_non_cachable:
                raise TypeError("cache_dependent {} is not a `ZfitCachable` but {}".format(cache_dep,
                                                                                           type(cache_dep)))

    def reset_cache_self(self):
        """Clear the cache of self and all dependent cachers."""
        self._clean_cache()
        self._inform_cachers()

    def reset_cache(self, reseter: 'ZfitGraphCachable'):
        self.reset_cache_self()

    def _clean_cache(self):
        # for func_holder in self.graph_caching_methods:
        #     func_holder.reset
        self._cache = {}
        return

    def _inform_cachers(self):
        for cacher in self._cachers:
            cacher.reset_cache(reseter=self)


def invalidate_graph(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        self = args[0]
        if not isinstance(self, ZfitGraphCachable):
            raise TypeError("Decorator can only be used in a subclass of `ZfitCachable`")
        self.reset_cache(reseter=self)

        return func(*args, **kwargs)

    return wrapped_func


class FunctionCacheHolder(GraphCachable):
    IS_TENSOR = object()

    def __init__(self, func, wrapped_func,
                 cachables: Union[ZfitGraphCachable, object, Iterable[Union[ZfitGraphCachable, object]]] = None,
                 cachables_mapping=None):
        """`tf.function` decorated function holder with caching dependencies on inputs.

        A `tf.function` creates a new graph for every signature that is encountered. It automatically caches them but
        thereby assumes that Python objects are immutable. Any mutation won't be detected. Therefore, an extra wrapper
        is needed. The input signature is compared with firstly checking whether the function is the same and then
        doing an equal comparison of the arguments (maybe too costly?).

        The `FunctionCacheHolder` holds the
         - original python function which serves as the hash of the object
         - wrapped python function, `wrapped_func`
         - the (keyword-)arguments

        If any of the keyword arguments changes in a way that the graph cache is invalid, this holder will have
        `is_valid` set to False and the `wrapped_func` cannot be used anymore, instead a new `tf.function` should
        be created as a call to the `wrapped_func` with the given arguments will result in an outdated graph.

        Args:
            func: Python function that serves as a hash of the holder. Notice that equality is different
                defined.
            wrapped_func: Wrapped `func` with `tf.function`. The holder signals via
                `is_valid` whether this function is still valid to be used.
            cachables: objects that are cached. If they change, the cache is invalidated
            cachables_mapping: keyword arguments to the function. If the values change, the cache is
                invalidated.
        """
        # cache = {} if cache is None else cache
        self.delete_from_cache = False
        self.wrapped_func = wrapped_func
        # self.parent_cache = cache
        self.python_func = func
        self._hash_value = hash(self.python_func)
        if cachables is None and cachables_mapping is None:
            raise ValueError("Both `cachables and `cachables_mapping` are None. One needs to be different from None.")
        if cachables is None:
            cachables = []
        if cachables_mapping is None:
            cachables_mapping = {}
        cachables = convert_to_container(cachables, container=list)
        cachables_values = convert_to_container(cachables_mapping.values(), container=list)
        cachables_all = cachables + cachables_values
        self.immutable_representation = self.create_immutable(cachables, cachables_mapping)
        # self._hash_value = hash(self.immutable_representation)
        super().__init__()  # resets the cache
        self.add_cache_deps(cachables_all)
        self.is_valid = True  # needed to make the cache valid again

    def reset_cache_self(self):
        self.is_valid = False

    def create_immutable(self, args, kwargs):
        """Create a tuple of the args and kwargs by combining them as args + kwargs.keys() + kwargs.values()`

        Args:
            args: list like
            kwargs: dict-like

        Returns:
        """
        # is initialized before the core
        from ..core.interfaces import ZfitData, ZfitParameter, ZfitSpace

        args = list(args)
        kwargs = list(kwargs.keys()) + list(kwargs.values())
        combined = []
        if args != []:
            combined += args
        if kwargs != []:
            combined += args
        combined_cleaned = []
        for obj in combined:
            if isinstance(obj, ZfitData):
                obj = (id(obj),)

            elif isinstance(obj, ZfitParameter):
                obj = (ZfitParameter, obj.name)
            elif isinstance(obj, ZfitSpace):
                obj = (id(obj),)
            elif tf.is_tensor(obj):
                obj = self.IS_TENSOR
            elif isinstance(obj, np.ndarray):
                obj = (obj,) if sum(obj.shape) < 20 else id(obj)
            combined_cleaned.append(obj)

        return tuple(combined_cleaned)

    def __hash__(self) -> int:
        return self._hash_value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionCacheHolder):
            return False
        # return all(obj1 == obj2 for obj1, obj2 in zip(self.immutable_representation, other.immutable_representation))
        array_repr_self = np.array(self.immutable_representation, dtype=object)
        array_repr_other = np.array(other.immutable_representation, dtype=object)
        try:
            return all(np.equal(array_repr_self, array_repr_other))
        except ValueError:  # broadcasting does not work
            return False
        except TypeError:  # OperatorNotAllowedError inherits from this
            return False
        # TODO: activate the below? costly, but runs?
        # except OperatorNotAllowedInGraphError:  # we have to assume they're not the same
        #     return False

    def __repr__(self) -> str:
        return f"<FunctionCacheHolder: {self.python_func}, valid={self.is_valid}>"


def clear_graph_cache():
    from zfit.z.zextension import FunctionWrapperRegistry

    for registry in FunctionWrapperRegistry.registries:
        for all_meth in registry.function_cache.values():
            for wrapped_meth in all_meth:
                wrapped_meth = wrapped_meth.wrapped_func
                wrapped_meth._created_variables = None
                wrapped_meth._stateful_fn = None
                wrapped_meth._stateless_fn = None
                wrapped_meth._descriptor_cache.clear()

    for registry in FunctionWrapperRegistry.registries:
        registry.reset()
    for instance in GraphCachable.instances:
        instance.reset_cache('global')
    # Cachable.graph_caching_methods.clear()
    tf.compat.v1.reset_default_graph()
