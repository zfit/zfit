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

#  Copyright (c) 2020 zfit

import abc
from abc import abstractmethod
import functools
from contextlib import suppress
from types import MethodType
from typing import Iterable, Union, Mapping

import tensorflow as tf

from . import ztyping
from .container import convert_to_container


class ZfitCachable:

    @abstractmethod
    def register_cacher(self, cacher: "ZfitCachable"):
        raise NotImplementedError

    @abstractmethod
    def add_cache_dependents(self, cache_dependents, allow_non_cachable):
        """Add dependents that render the cache invalid if they change.

        Args:
            cache_dependents (ZfitCachable):
            allow_non_cachable (bool): If `True`, allow `cache_dependents` to be non-cachables.
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


class Cachable(ZfitCachable):
    graph_caching_methods = []
    old_graph_caching_methods = []

    def __init__(self, *args, **kwargs):
        self._cache = {}
        self._cachers = {}
        self.reset_cache_self()
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        graph_caching_methods = []
        for func_name in dir(cls):
            if not func_name.startswith("__"):
                func = getattr(cls, func_name)
                if callable(func) and hasattr(func, 'zfit_graph_cache_registered'):
                    assert hasattr(func, "_descriptor_cache"), "TensorFlow internals have changed. Need to update cache"
                    func.zfit_graph_cache_registered = True
                    graph_caching_methods.append(func)
        cls.graph_caching_methods = graph_caching_methods

    def register_cacher(self, cacher: ztyping.CacherOrCachersType):
        """Register a `cacher` that caches values produces by this instance; a dependent.

        Args:
            cacher ():
        """
        if not isinstance(cacher, ZfitCachable):
            raise TypeError("`cacher` is not a `ZfitCachable` but {}".format(type(cacher)))
        if not cacher in self._cachers:
            self._cachers[cacher] = None  # could we have a more useful value?

    def add_cache_dependents(self, cache_dependents: ztyping.CacherOrCachersType,
                             allow_non_cachable: bool = True):
        """Add dependents that render the cache invalid if they change.

        Args:
            cache_dependents (ZfitCachable):
            allow_non_cachable (bool): If `True`, allow `cache_dependents` to be non-cachables.
                If `False`, any `cache_dependents` that is not a `ZfitCachable` will raise an error.

        Raises:
            TypeError: if one of the `cache_dependents` is not a `ZfitCachable` _and_ `allow_non_cachable`
                if `False`.
        """
        cache_dependents = convert_to_container(cache_dependents)
        for cache_dependent in cache_dependents:
            if isinstance(cache_dependent, ZfitCachable):
                cache_dependent.register_cacher(self)
            elif not allow_non_cachable:
                raise TypeError("cache_dependent {} is not a `ZfitCachable` but {}".format(cache_dependent,
                                                                                           type(cache_dependent)))

    def reset_cache_self(self):
        """Clear the cache of self and all dependent cachers."""
        self._clean_cache()
        self._inform_cachers()

    def reset_cache(self, reseter: 'ZfitCachable'):
        self.reset_cache_self()

    def _clean_cache(self):
        # pass
        self._cache = {}
        return
        for method in self.graph_caching_methods:
            # on first run
            if method._created_variables is None and method._stateful_fn is None and method._stateless_fn is None:
                continue
            method._created_variables = None
            method._stateful_fn = None
            method._stateless_fn = None
            self.old_graph_caching_methods.append(method._descriptor_cache.copy())
            method._descriptor_cache.clear()
            # method._function_cache.clear()
            continue

            funcs = [method._stateful_fn, method._stateless_fn]
            for func in funcs:
                # funcs
                if func is not None:
                    for garbage_collector in funcs._function_cache._garbage_collectors:
                        garbage_collector._cache.clear()
            # continue
            # from tensorflow_core.python.eager.function import FunctionCache
            # method._function_cache = FunctionCache()
        #     self.old_graph_caching_methods.append(method)  # to prevent graphs from being garbace collected
        #     method_tf = method.zfit_func_to_graph(method.zfit_python_func)
        #     bound_method = MethodType(method_tf, self)
        #     setattr(self, method.__name__, bound_method)
        # pass
        # method._descriptor_cache.clear()

    def _inform_cachers(self):
        for cacher in self._cachers:
            cacher.reset_cache(reseter=self)


def invalidates_cache(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        print("DEBUG, invalidating cache, I am:", func)

        self = args[0]
        if not isinstance(self, ZfitCachable):
            raise TypeError("Decorator can only be used in a subclass of `ZfitCachable`")
        self.reset_cache(reseter=self)

        return func(*args, **kwargs)

    return wrapped_func


# class MutableCacheHolder(Cachable):
#     next_cache_number = 0
#     IS_TENSOR = object()
#
#     def __init__(self, cachables: Union[ZfitCachable, object, Iterable[Union[ZfitCachable, object]]] = None,
#                  cachables_mapping=None, cache: Mapping = None):
#         """
#         Args:
#             cachables: objects that are cached. If they change, the cache is invalidated
#             cache: The cache where the objects are stored with the `cache_number`. If given, will delete itself from
#                 the cache once invalidated.
#         """
#         if cachables is None and cachables_mapping is None:
#             raise ValueError("Both `cachables and `cachables_mapping` are None. One needs to be different from None.")
#         if cachables is None:
#             cachables = []
#         if cachables_mapping is None:
#             cachables_mapping = {}
#         self.delete_from_cache = False
#         self.parent_cache = cache
#         self.is_valid = True
#         self._cache_number = self.next_cache_number
#         self.next_cache_number += 1
#         cachables = convert_to_container(cachables, container=list)
#         cachables_values = convert_to_container(cachables_mapping.values(), container=list)
#         cachables_all = cachables + cachables_values
#         self.immutable_representation = self.create_immutable(*cachables, **cachables_mapping)
#
#         self.hash_value = hash(self.immutable_representation)
#         super().__init__()
#         self.add_cache_dependents(cachables_all)
#         self.delete_from_cache = True
#
#     @property
#     def cache_number(self):
#         return self._cache_number
#
#     def create_immutable(self, *args, **kwargs):
#         args = list(args)
#         kwargs = list(kwargs.keys()) + list(kwargs.values())
#         combined = []
#         if args != []:
#             combined += args
#         if kwargs != []:
#             combined += args
#         combined = [self.IS_TENSOR if isinstance(obj, (tf.Tensor, tf.Variable)) else obj for obj in combined]
#         return tuple(combined)
#
#     def reset_cache_self(self):
#         self.is_valid = False
#         if self.parent_cache is not None and self.delete_from_cache:
#             with suppress(KeyError):
#                 del self.parent_cache[self]
#
#     def __hash__(self) -> int:
#         return self.hash_value
#
#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, MutableCacheHolder):
#             return False
#         return all(obj1 == obj2 for obj1, obj2 in zip(self.immutable_representation, other.immutable_representation))
#
#     def __repr__(self) -> str:
#         return f"<MutableCacheHolder: {self.immutable_representation}>"


class FunctionCacheHolder(Cachable):
    IS_TENSOR = object()

    def __init__(self, func, wrapped_func,
                 cachables: Union[ZfitCachable, object, Iterable[Union[ZfitCachable, object]]] = None,
                 cachables_mapping=None):
        """
        Args:
            cachables: objects that are cached. If they change, the cache is invalidated
            cache: The cache where the objects are stored with the `cache_number`. If given, will delete itself from
                the cache once invalidated.
        """
        # cache = {} if cache is None else cache
        self.delete_from_cache = False
        self.wrapped_func = wrapped_func
        # self.parent_cache = cache
        self.python_func = func
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
        super().__init__()
        self.add_cache_dependents(cachables_all)
        self.is_valid = True

    def reset_cache_self(self):
        self.is_valid = False
        # if self.parent_cache and self.delete_from_cache:
        #     with suppress(KeyError):
        #         del self.parent_cache[self.caching_func]

    def create_immutable(self, args, kwargs):
        args = list(args)
        kwargs = list(kwargs.keys()) + list(kwargs.values())
        combined = []
        if args != []:
            combined += args
        if kwargs != []:
            combined += args
        combined = [self.IS_TENSOR if isinstance(obj, (tf.Tensor, tf.Variable)) else obj for obj in combined]
        return tuple(combined)

    def __hash__(self) -> int:
        return hash(self.python_func)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionCacheHolder):
            return False
        # return all(obj1 == obj2 for obj1, obj2 in zip(self.immutable_representation, other.immutable_representation))
        import numpy as np
        try:
            return all(np.equal(self.immutable_representation, other.immutable_representation))
        except ValueError:  # broadcasting does not work
            return False

    def __repr__(self) -> str:
        return f"<FunctionCacheHolder: {self.python_func}, valid={self.is_valid}>"
