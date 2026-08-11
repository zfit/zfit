#  Copyright (c) 2025 zfit

from __future__ import annotations

import functools
import typing
from collections.abc import Iterable

import hist
import numpy as np
import zfit_interface as zinterface

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

# @tfp.experimental.auto_composite_tensor()
# class Regular(hist.axis.Regular, tfp.experimental.AutoCompositeTensor, family='zfit'):
#     pass
from .._interfaces import ZfitRectBinning


class Variable(zinterface.variables.ZfitVar):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class SpaceV2:
    def __init__(self, axes):
        self.axes = axes

    def __getitem__(self, key):
        key = to_var_str(key)
        for axis in self.axes:
            if axis.name == key:
                return axis
        msg = f"{key} not in {self}."
        raise KeyError(msg)

    def __iter__(self):
        yield from self.axes

    @property
    def names(self):
        return [axis.name for axis in self]


def to_var_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, zinterface.variables.ZfitVar):
        return value.name
    return None


class Axis(Variable):
    def __init__(self, name):
        super().__init__(name=name)


class UnbinnedAxis(Axis):
    def __init__(self, name, lower=None, upper=None):
        super().__init__(name)
        self.lower = lower
        self.upper = upper


def _to_pyfloat(value) -> float:
    """Convert a python number, numpy array/scalar or (eager) tensor with a single element to a python float."""
    return float(np.asarray(value).reshape(-1)[0])


class HashableAxisMixin:
    """Validates the axis name and provides edges-based equality/hashing for zfit-native binnings."""

    def __init__(self, *, name: str) -> None:
        """Store and validate the axis name.

        Args:
            name: Name of the axis; must be non-empty.
        """
        if name == "":
            msg = "Currently, a binning has to have a name coinciding with the obs."
            raise ValueError(msg)
        self.name = name

    def __eq__(self, other):
        """Compare by type, name, and edges."""
        if type(self) is not type(other):
            return NotImplemented
        return self.name == other.name and np.array_equal(self.edges, other.edges)

    def __hash__(self):
        return hash((type(self).__name__, self.name, tuple(self.edges)))


class BinningBase(HashableAxisMixin, ZfitRectBinning):
    """Shared behavior for zfit-native 1D binnings, computed purely from ``self.edges``."""

    transform = None  # hist.axis compatibility: zfit binnings never carry a coordinate transform

    @property
    def size(self) -> int:
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    def get_edges(self):
        return self.edges

    def __len__(self) -> int:
        """Number of bins, per the UHI ``PlottableAxis`` protocol (used e.g. by ``mplhep``)."""
        return self.size

    def __getitem__(self, i: int) -> tuple[float, float]:
        """Bin ``i`` as a ``(lower, upper)`` edge pair, per the UHI ``PlottableAxis`` protocol."""
        return self.edges[i], self.edges[i + 1]

    def __repr__(self):
        """String representation showing the name and edges."""
        return f"{type(self).__name__}(name={self.name!r}, edges={self.edges})"


class RegularBinning(BinningBase):
    def __init__(self, bins: int, start: float, stop: float, *, name: str) -> None:
        self.edges = np.linspace(_to_pyfloat(start), _to_pyfloat(stop), bins + 1)
        super().__init__(name=name)


class VariableBinning(BinningBase):
    def __init__(self, edges: Iterable[float], *, name: str) -> None:
        self.edges = np.asarray(list(edges), dtype=float)
        super().__init__(name=name)


class _ArrayTuple(tuple):
    """
    Tuple of (possibly differently shaped, sparse-broadcastable) arrays.

    Mirrors ``boost_histogram``'s ``ArrayTuple``: reductions like ``np.prod(t, axis=0)`` dispatch to
    ``t.prod`` (numpy tries the method before falling back to ``np.multiply.reduce``), which here
    broadcasts the members to a common dense shape first -- required since e.g. ``binning.widths`` holds
    per-axis arrays of shape ``(n_i, 1, ..., 1)`` that a plain tuple can't be turned into a single ndarray from.
    """

    __slots__ = ()
    _REDUCTIONS = frozenset(("sum", "any", "all", "min", "max", "prod"))

    def __getattr__(self, name):
        if name in self._REDUCTIONS:
            return functools.partial(getattr(np, name), np.broadcast_arrays(*self))
        return self.__class__(getattr(a, name) for a in self)


class Binnings(tuple):
    """A tuple of zfit-native binning axes with hist-like container-level convenience properties."""

    __slots__ = ()

    def _index_by_name(self, key):
        if not isinstance(key, str):
            return key
        for i, ax in enumerate(self):
            if ax.name == key:
                return i
        msg = f"{key} not found in axes"
        raise KeyError(msg)

    def __getitem__(self, item):
        """Index by position, slice, or axis name."""
        if isinstance(item, slice):
            item = slice(self._index_by_name(item.start), self._index_by_name(item.stop), item.step)
        else:
            item = self._index_by_name(item)
        result = super().__getitem__(item)
        return Binnings(result) if isinstance(result, tuple) else result

    @property
    def name(self) -> tuple[str, ...]:
        return tuple(ax.name for ax in self)

    @property
    def size(self) -> tuple[int, ...]:
        return tuple(ax.size for ax in self)

    @property
    def edges(self) -> _ArrayTuple:
        return _ArrayTuple(np.meshgrid(*(ax.edges for ax in self), sparse=True, indexing="ij"))

    @property
    def centers(self) -> _ArrayTuple:
        return _ArrayTuple(np.meshgrid(*(ax.centers for ax in self), sparse=True, indexing="ij"))

    @property
    def widths(self) -> _ArrayTuple:
        return _ArrayTuple(np.meshgrid(*(ax.widths for ax in self), sparse=True, indexing="ij"))


def new_from_axis(axis):
    if isinstance(axis, RegularBinning | VariableBinning):
        return axis
    if isinstance(axis, hist.axis.Regular):
        lower, upper = axis.edges[0], axis.edges[-1]
        if axis.transform is not None:
            msg = (
                "Transformed axes are not supported. Please convert it explicitly to a Variable axis using the edges."
                "Example: ax2 = hist.axis.Variable(ax1.edges, name='x')."
                "If this is an issue or you prefer to have this automatically converted, please open an issue on github with zfit."
            )
            raise ValueError(msg)
        return RegularBinning(axis.size, lower, upper, name=axis.name)
    if isinstance(axis, hist.axis.Variable):
        return VariableBinning(axis.edges, name=axis.name)
    msg = f"{axis} is not a valid axis."
    raise ValueError(msg)


def histaxes_to_binning(binnings):
    new_binnings = []
    for binning in binnings:
        new_binnings.append(new_from_axis(binning))
    return Binnings(new_binnings)


def binning_to_histaxes(binnings):
    histaxes = []
    for binning in binnings:
        if isinstance(binning, RegularBinning):
            histaxes.append(
                hist.axis.Regular(binning.size, binning.edges[0], binning.edges[-1], name=binning.name, flow=False)
            )
        elif isinstance(binning, VariableBinning):
            histaxes.append(hist.axis.Variable(binning.edges, name=binning.name, flow=False))
        elif isinstance(binning, hist.axis.Regular | hist.axis.Variable):
            histaxes.append(binning)
        else:
            msg = f"{binning} is not a valid binning."
            raise ValueError(msg)
    return histaxes
