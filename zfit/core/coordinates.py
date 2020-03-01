#  Copyright (c) 2020 zfit
from typing import Optional, Tuple, Union, List

import numpy as np
import tensorflow as tf

from zfit import z
from .interfaces import ZfitOrderableDimensional, ZfitSpace
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import OverdefinedError, CoordinatesUnderdefinedError, CoordinatesIncompatibleError, \
    AxesIncompatibleError, ObsIncompatibleError, WorkInProgressError


class Coordinates(ZfitOrderableDimensional):
    def __init__(self, obs=None, axes=None):
        obs, axes, n_obs = self._check_convert_obs_axes(obs, axes)
        self._obs = obs
        self._axes = axes
        self._n_obs = n_obs

    @staticmethod
    def _check_convert_obs_axes(obs, axes):
        if isinstance(obs, ZfitOrderableDimensional):
            if axes is not None:
                raise OverdefinedError(f"Cannot use {obs}, a"
                                       " ZfitOrderableDimensional as obs with axes not None"
                                       " (currently, please open an issue if desired)")
            coord = obs
            return coord.obs, coord.axes, coord.n_obs
        obs = convert_to_obs_str(obs, container=tuple)
        axes = convert_to_axes(axes, container=tuple)
        if obs is None:
            if axes is None:
                raise CoordinatesUnderdefinedError("Neither obs nor axes specified")
            else:
                if any(not isinstance(ax, int) for ax in axes):
                    raise ValueError(f"Axes have to be int. Currently: {axes}")
            n_obs = len(axes)
        else:
            if any(not isinstance(ob, str) for ob in obs):
                raise ValueError(f"Observables have to be strings. Currently: {obs}")
            n_obs = len(obs)
            if axes is not None:
                if not len(obs) == len(axes):
                    raise CoordinatesIncompatibleError("obs and axes do not have the same length.")
        return obs, axes, n_obs

    @property
    def obs(self):
        return self._obs

    @property
    def axes(self):
        return self._axes

    @property
    def n_obs(self):
        return self._n_obs

    def with_obs(self, obs: Optional[ztyping.ObsTypeInput], allow_superset: bool = False, allow_subset: bool = False):
        if obs is None:  # drop obs, check if there are axes
            if self.axes is None:
                raise AxesIncompatibleError("cannot remove obs (using None) for a Space without axes")
            new_coords = type(self)(obs=obs, axes=self.axes)
        else:
            obs = _convert_obs_to_str(obs)
            if self.obs is None:
                new_coords = type(self)(obs=obs, axes=self.axes)
            else:

                if not frozenset(obs) == frozenset(self.obs):

                    if not allow_superset and frozenset(obs) - frozenset(self.obs):
                        raise ObsIncompatibleError(
                            f"Obs {obs} are a superset of {self.obs}, not allowed according to flag.")

                    if not allow_subset and set(self.obs) - set(obs):
                        raise ObsIncompatibleError(
                            f"Obs {obs} are a subset of {self.obs}, not allowed according to flag.")
                new_indices = self.get_reorder_indices(obs=obs)
                new_obs = self._reorder_obs(indices=new_indices)
                new_axes = self._reorder_axes(indices=new_indices)
                new_coords = type(self)(obs=new_obs, axes=new_axes)
        return new_coords

    def with_axes(self, axes: Optional[ztyping.AxesTypeInput], allow_superset: bool = False,
                  allow_subset: bool = False) -> "zfit.Space":
        """Sort by `axes` and return the new instance. `None` drops the axes.

        Args:
            axes ():
            allow_superset (bool): Allow `axes` to be a superset of the `Spaces` axes

        Returns:
            :py:class:`~zfit.Space`
        """
        if axes is None:  # drop axes
            if self.obs is None:
                raise ObsIncompatibleError("Cannot remove axes (using None) for a Space without obs")
            new_coords = type(self)(obs=self.obs, axes=axes)
        else:
            axes = _convert_axes_to_int(axes)
            if not frozenset(axes) == frozenset(self.axes):
                if not allow_superset and set(axes) - set(self.axes):
                    raise AxesIncompatibleError(
                        f"Axes {axes} are a superset of {self.axes}, not allowed according to flag.")

                if not allow_subset and set(self.axes) - set(axes):
                    raise AxesIncompatibleError(
                        f"Axes {axes} are a subset of {self.axes}, not allowed according to flag.")
            new_indices = self.get_reorder_indices(axes=axes)
            new_obs = self._reorder_obs(indices=new_indices)
            new_axes = self._reorder_axes(indices=new_indices)
            new_coords = type(self)(obs=new_obs, axes=new_axes)
        return new_coords

    def with_autofill_axes(self, overwrite: bool = False) -> "zfit.Space":
        """Return a :py:class:`~zfit.Space` with filled axes corresponding to range(len(n_obs)).

        Args:
            overwrite (bool): If `self.axes` is not None, replace the axes with the autofilled ones.
                If axes is already set, don't do anything if `overwrite` is False.

        Returns:
            :py:class:`~zfit.Space`
        """
        if self.axes and not overwrite:
            raise ValueError("overwrite is not allowed but axes are already set.")
        new_coords = type(self)(obs=self.obs, axes=range(self.n_obs))
        return new_coords

    def _reorder_obs(self, indices: Tuple[int]) -> ztyping.ObsTypeReturn:
        obs = self.obs
        if obs is not None:
            obs = tuple(obs[i] for i in indices)
        return obs

    def _reorder_axes(self, indices: Tuple[int]) -> ztyping.AxesTypeReturn:
        axes = self.axes
        if axes is not None:
            axes = tuple(axes[i] for i in indices)
        return axes

    def get_reorder_indices(self, obs: ztyping.ObsTypeInput = None,
                            axes: ztyping.AxesTypeInput = None) -> Tuple[int]:
        """Indices that would order `self.obs` as `obs` respectively `self.axes` as `axes`.

        Args:
            obs ():
            axes ():

        Returns:

        """
        obs_none = obs is None
        axes_none = axes is None

        obs_is_defined = self.obs is not None and not obs_none
        axes_is_defined = self.axes is not None and not axes_none
        if not (obs_is_defined or axes_is_defined):
            raise ValueError(
                "Neither the `obs` (argument and on instance) nor `axes` (argument and on instance) are defined.")

        if obs_is_defined:
            old, new = self.obs, [o for o in obs if o in self.obs]
        else:
            old, new = self.axes, [a for a in axes if a in self.axes]

        new_indices = _reorder_indices(old=old, new=new)
        return new_indices

    def reorder_x(self, x: Union[tf.Tensor, np.ndarray], *, x_obs: ztyping.ObsTypeInput = None,
                  x_axes: ztyping.AxesTypeInput = None, func_obs: ztyping.ObsTypeInput = None,
                  func_axes: ztyping.AxesTypeInput = None) -> Union[tf.Tensor, np.ndarray]:
        """Reorder x in the last dimension either according to its own obs or assuming a function ordered with func_obs.

        There are two obs or axes around: the one associated with this Coordinate object and the one associated with x.
        If x_obs or x_axes is given, then this is assumed to be the obs resp. the axes of x and x will be reordered
        according to `self.obs` resp. `self.axes`.

        If func_obs resp. func_axes is given, then x is assumed to have `self.obs` resp. `self.axes` and will be
        reordered to align with a function ordered with `func_obs` resp. `func_axes`.

        Switching `func_obs` for `x_obs` resp. `func_axes` for `x_axes` inverts the reordering of x.

        Args:
            x (tensor-like): Tensor to be reordered, last dimension should be n_obs resp. n_axes
            x_obs: Observables associated with x. If both, x_obs and x_axes are given, this has precedency over the
                latter.
            x_axes: Axes associated with x.
            func_obs: Observables associated with a function that x will be given to. Reorders x accordingly and assumes
                self.obs to be the obs of x. If both, `func_obs` and `func_axes` are given, this has precedency over the
                latter.
            func_axes: Axe associated with a function that x will be given to. Reorders x accordingly and assumes
                self.axes to be the axes of x.

        Returns:

        """
        x_reorder = x_obs is not None or x_axes is not None
        func_reorder = func_obs is not None or func_axes is not None
        if not (x_reorder ^ func_reorder):
            raise ValueError("Either specify `x_obs/axes` or `func_obs/axes`, not both.")
        obs_defined = x_obs is not None or func_obs is not None
        axes_defined = x_axes is not None or func_axes is not None
        if obs_defined and self.obs:
            if x_reorder:
                coord_old = x_obs
                coord_new = self.obs
            elif func_reorder:
                coord_new = func_obs
                coord_old = self.obs
            else:
                assert False, 'bug, should never be reached'

        elif axes_defined and self.axes:
            if x_reorder:
                coord_old = x_axes
                coord_new = self.axes
            elif func_reorder:
                coord_new = func_axes
                coord_old = self.axes
            else:
                assert False, 'bug, should never be reached'
        else:
            raise ValueError("Obs and self.obs or axes and self. axes not properly defined. Can only reorder on defined"
                             " coordinates.")

        new_indices = _reorder_indices(old=coord_old, new=coord_new)

        x = z.unstable.gather(x, indices=new_indices, axis=-1)
        return x

    def __eq__(self, other):
        if not isinstance(other, Coordinates):
            return NotImplemented
        obs_equal = False
        axes_equal = False
        if self.obs is not None and other.obs is not None:
            obs_equal = frozenset(self.obs) == frozenset(other.obs)

        if self.axes is not None and other.axes is not None:
            axes_equal = frozenset(self.axes) == frozenset(other.axes)
        equal = obs_equal or axes_equal
        return equal

    def __hash__(self):
        return 42  # always check with equal...  maybe change in future, use dict that checks for different things.

    def __repr__(self):
        return f"<zfit Coordinates obs={self.obs}, axes={self.axes}"


def _convert_axes_to_int(axes):
    if isinstance(axes, ZfitSpace):
        axes = axes.axes
    else:
        axes = convert_to_container(axes, container=tuple)
    return axes


def _convert_obs_to_str(obs):
    if isinstance(obs, ZfitSpace):
        obs = obs.obs
    else:
        obs = convert_to_container(obs, container=tuple)
    return obs


def _reorder_indices(old: Union[List, Tuple], new: Union[List, Tuple]) -> Tuple[int]:
    new_indices = tuple(old.index(o) for o in new)
    return new_indices


def convert_to_axes(axes, container=tuple):
    """Convert `obs` to the list of obs, also if it is a :py:class:`~ZfitSpace`. Return None if axes is None.

    """
    if axes is None:
        return axes
    axes = convert_to_container(value=axes, container=container)
    new_axes = []
    for axis in axes:
        if isinstance(axis, ZfitSpace):
            if len(axis) > 1:
                raise WorkInProgressError("Not implemented, uniqueify?")
            new_axes.extend(axis.obs)
        else:
            new_axes.append(axis)
    return container(new_axes)


def convert_to_obs_str(obs, container=tuple):
    """Convert `obs` to the list of obs, also if it is a :py:class:`~ZfitSpace`. Return None if obs is None.

    """
    if obs is None:
        return obs
    obs = convert_to_container(value=obs, container=container)
    new_obs = []
    for ob in obs:
        if isinstance(ob, ZfitSpace):
            if len(ob) > 1:
                raise WorkInProgressError("Not implemented, uniqueify?")
            new_obs.extend(ob.obs)
        else:
            new_obs.append(ob)
    return container(new_obs)
