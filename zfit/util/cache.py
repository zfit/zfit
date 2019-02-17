import abc
from abc import abstractmethod
import functools

import pep487

from zfit.util import ztyping
from zfit.util.container import convert_to_container


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

    def __init__(self, *args, **kwargs):
        self._cache = {}
        self._cachers = {}
        self.reset_cache_self()
        super().__init__(*args, **kwargs)

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
        self._cache = {}

    def _inform_cachers(self):
        for cacher in self._cachers:
            cacher.reset_cache(reseter=self)


def invalidates_cache(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        self = args[0]
        if not isinstance(self, ZfitCachable):
            raise TypeError("Decorator can only be used in a subclass of `ZfitCachable`")
        self.reset_cache(reseter=self)

        return func(*args, **kwargs)

    return wrapped_func
