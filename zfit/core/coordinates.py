#  Copyright (c) 2023 zfit

from __future__ import annotations

import numpy as np
import tensorflow as tf

from zfit import z
from .interfaces import ZfitData, ZfitDimensional, ZfitOrderableDimensional, ZfitSpace
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import (
    AxesIncompatibleError,
    CoordinatesIncompatibleError,
    CoordinatesUnderdefinedError,
    IntentionAmbiguousError,
    ObsIncompatibleError,
    OverdefinedError,
)


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
                raise OverdefinedError(
                    f"Cannot use {obs}, a"
                    " ZfitOrderableDimensional as obs with axes not None"
                    " (currently, please open an issue if desired)"
                )
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
                    raise CoordinatesIncompatibleError(
                        "obs and axes do not have the same length."
                    )
        if not (obs or axes):
            raise CoordinatesUnderdefinedError(
                f"Neither obs {obs} nor axes {axes} are defined."
            )
        return obs, axes, n_obs

    @property
    def obs(self) -> ztyping.ObsTypeReturn:
        """Return the observables, string identifier for the coordinate system."""
        return self._obs

    @property
    def axes(self) -> ztyping.AxesTypeReturn:
        """Return the axes, integer based identifier(indices) for the coordinate system."""
        return self._axes

    @property
    def n_obs(self) -> int:
        """Return the number of observables, the dimensionality.

        Corresponds to the last dimension.
        """
        return self._n_obs

    def with_obs(
        self,
        obs: ztyping.ObsTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> Coordinates:
        """Create a new instance that has ``obs``; sorted by or set or dropped.

        The behavior is as follows:

         * obs are already set:

           * input obs are None: the observables will be dropped. If no axes are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input obs are not None: the instance will be sorted by the incoming obs. If axes or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned.
             This can be used to take a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the obs will be sorted accordingly as
             if the obs not contained in the instances obs were not in the input obs.
         * obs are not set:

           * if the input obs are None, the same object is returned.
           * if the input obs are not None, they will be set as-is and now correspond to the already
             existing axes in the object.

        Args:
            obs: Observables to sort/associate this instance with
            allow_superset: if False and a strict superset of the own observables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own observables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/observables

        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            ObsIncompatibleError: if ``obs`` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        obs = convert_to_obs_str(obs)
        if obs is None:  # drop obs, check if there are axes
            if self.axes is None:
                raise AxesIncompatibleError(
                    "cannot remove obs (using None) for a Space without axes"
                )
            new_coords = type(self)(obs=obs, axes=self.axes)
        else:
            obs = _convert_obs_to_str(obs)
            if self.obs is None:
                new_coords = type(self)(obs=obs, axes=self.axes)
            else:
                if not set(obs).intersection(self.obs):
                    raise ObsIncompatibleError(
                        f"The requested obs {obs} are not compatible with the current obs "
                        f"{self.obs}"
                    )

                if not frozenset(obs) == frozenset(self.obs):
                    if not allow_superset and frozenset(obs) - frozenset(self.obs):
                        raise ObsIncompatibleError(
                            f"Obs {obs} are a superset of {self.obs}, not allowed according to flag."
                        )

                    if not allow_subset and set(self.obs) - set(obs):
                        raise ObsIncompatibleError(
                            f"Obs {obs} are a subset of {self.obs}, not allowed according to flag."
                        )
                new_indices = self.get_reorder_indices(obs=obs)
                new_obs = self._reorder_obs(indices=new_indices)
                new_axes = self._reorder_axes(indices=new_indices)
                new_coords = type(self)(obs=new_obs, axes=new_axes)
        return new_coords

    def with_axes(
        self,
        axes: ztyping.AxesTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> Coordinates:
        """Create a new instance that has ``axes``; sorted by or set or dropped.

        The behavior is as follows:

         * axes are already set:

           * input axes are None: the axes will be dropped. If no observables are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input axes are not None: the instance will be sorted by the incoming axes. If obs or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned. This can
             be used to retrieve a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the axes will be sorted accordingly as
             if the axes not contained in the instances axes were not present in the input axes.
         * axes are not set:

           * if the input axes are None, the same object is returned.
           * if the input axes are not None, they will be set as-is and now correspond to the already
             existing obs in the object.

        Args:
            axes: Axes to sort/associate this instance with
            allow_superset: if False and a strict superset of the own axeservables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own axeservables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/axes
        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            AxesIncompatibleError: if ``axes`` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        axes = convert_to_axes(axes)
        if axes is None:  # drop axes
            if self.obs is None:
                raise CoordinatesUnderdefinedError(
                    "Cannot remove axes (using None) for a Space without obs"
                )
            new_coords = type(self)(obs=self.obs, axes=axes)
        else:
            axes = _convert_axes_to_int(axes)
            if not self.axes and not len(axes) == len(self.obs):
                raise AxesIncompatibleError(
                    f"Trying to set axes {axes} to object with obs {self.obs}"
                )
            if self.axes is None:
                new_coords = type(self)(obs=self.obs, axes=axes)
            else:
                if not set(axes).intersection(self.axes):
                    raise AxesIncompatibleError(
                        f"The requested axes {axes} are not compatible with the current axes "
                        f"{self.axes}"
                    )
                if not frozenset(axes) == frozenset(self.axes):
                    if not allow_superset and set(axes) - set(self.axes):
                        raise AxesIncompatibleError(
                            f"Axes {axes} are a superset of {self.axes}, not allowed according to flag."
                        )

                    if not allow_subset and set(self.axes) - set(axes):
                        raise AxesIncompatibleError(
                            f"Axes {axes} are a subset of {self.axes}, not allowed according to flag."
                        )
                new_indices = self.get_reorder_indices(axes=axes)
                new_obs = self._reorder_obs(indices=new_indices)
                new_axes = self._reorder_axes(indices=new_indices)
                new_coords = type(self)(obs=new_obs, axes=new_axes)
        return new_coords

    def with_autofill_axes(self, overwrite: bool = False) -> ZfitOrderableDimensional:
        """Overwrite the axes of the current object with axes corresponding to range(len(n_obs)).

        This effectively fills with (0, 1, 2,...) and can be used mostly when an object enters a PDF or
        similar. ``overwrite`` allows to remove the axis first in case there are already some set.

        .. code-block::

            object.obs -> ('x', 'z', 'y')
            object.axes -> None

            object.with_autofill_axes()

            object.obs -> ('x', 'z', 'y')
            object.axes -> (0, 1, 2)


        Args:
            overwrite: If axes are already set, replace the axes with the autofilled ones.
                If axes is already set and ``overwrite`` is False, raise an error.

        Returns:
            The object with the new axes

        Raises:
            AxesIncompatibleError: if the axes are already set and ``overwrite`` is False.
        """
        if self.axes and not overwrite:
            raise AxesIncompatibleError(
                "overwrite is not allowed but axes are already set."
            )
        new_coords = type(self)(obs=self.obs, axes=range(self.n_obs))
        return new_coords

    def _reorder_obs(self, indices: tuple[int]) -> ztyping.ObsTypeReturn:
        obs = self.obs
        if obs is not None:
            obs = tuple(obs[i] for i in indices)
        return obs

    def _reorder_axes(self, indices: tuple[int]) -> ztyping.AxesTypeReturn:
        axes = self.axes
        if axes is not None:
            axes = tuple(axes[i] for i in indices)
        return axes

    def get_reorder_indices(
        self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None
    ) -> tuple[int]:
        """Indices that would order the instances obs as ``obs`` respectively the instances axes as ``axes``.

        Args:
            obs: Observables that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.
            axes: Axes that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.

        Returns:
            New indices that would reorder the instances obs to be obs respectively axes.

        Raises:
            CoordinatesUnderdefinedError: If neither ``obs`` nor ``axes`` is given
        """
        obs_none = obs is None
        axes_none = axes is None

        obs_is_defined = self.obs is not None and not obs_none
        axes_is_defined = self.axes is not None and not axes_none
        if not (obs_is_defined or axes_is_defined):
            raise CoordinatesUnderdefinedError(
                "Neither the `obs` (argument and on instance) nor `axes` (argument and on instance) are defined."
            )

        if obs_is_defined:
            old, new = self.obs, [o for o in obs if o in self.obs]
        else:
            old, new = self.axes, [a for a in axes if a in self.axes]

        new_indices = _reorder_indices(old=old, new=new)
        return new_indices

    def reorder_x(
        self,
        x: tf.Tensor | np.ndarray,
        *,
        x_obs: ztyping.ObsTypeInput = None,
        x_axes: ztyping.AxesTypeInput = None,
        func_obs: ztyping.ObsTypeInput = None,
        func_axes: ztyping.AxesTypeInput = None,
    ) -> ztyping.XTypeReturnNoData:
        """Reorder x in the last dimension either according to its own obs or assuming a function ordered with func_obs.

        There are two obs or axes around: the one associated with this Coordinate object and the one associated with x.
        If x_obs or x_axes is given, then this is assumed to be the obs resp. the axes of x and x will be reordered
        according to ``self.obs`` resp. ``self.axes``.

        If func_obs resp. func_axes is given, then x is assumed to have ``self.obs`` resp. ``self.axes`` and will be
        reordered to align with a function ordered with ``func_obs`` resp. ``func_axes``.

        Switching ``func_obs`` for ``x_obs`` resp. ``func_axes`` for ``x_axes`` inverts the reordering of x.

        Args:
            x: Tensor to be reordered, last dimension should be n_obs resp. n_axes
            x_obs: Observables associated with x. If both, x_obs and x_axes are given, this has precedency over the
                latter.
            x_axes: Axes associated with x.
            func_obs: Observables associated with a function that x will be given to. Reorders x accordingly and assumes
                self.obs to be the obs of x. If both, ``func_obs`` and ``func_axes`` are given, this has precedency over the
                latter.
            func_axes: Axe associated with a function that x will be given to. Reorders x accordingly and assumes
                self.axes to be the axes of x.

        Returns:
            The reordered array-like object
        """

        x_reorder = x_obs is not None or x_axes is not None
        func_reorder = func_obs is not None or func_axes is not None
        if not (x_reorder ^ func_reorder):
            raise ValueError(
                "Either specify `x_obs/axes` or `func_obs/axes`, not both."
            )
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
                assert False, "bug, should never be reached"

        elif axes_defined and self.axes:
            if x_reorder:
                coord_old = x_axes
                coord_new = self.axes
            elif func_reorder:
                coord_new = func_axes
                coord_old = self.axes
            else:
                assert False, "bug, should never be reached"
        else:
            raise ValueError(
                "Obs and self.obs or axes and self. axes not properly defined. Can only reorder on defined"
                " coordinates."
            )

        if isinstance(x, ZfitData) and not (
            coord_old == x.obs if obs_defined else x.axes
        ):
            raise IntentionAmbiguousError(
                "`reorder_x` is supposed to assume that the obs/axes of the given `x` are"
                " either the one from the Space itself or the ones given. x is a ZfitData"
                f" object that has obs/axes themselves {x.obs if obs_defined else x.axes}"
                f" which do not coincied with the assumed obs/axes {coord_old}. Use"
                f" x.value() to get the pure tensor out or rather sort Data accordingly"
                f" (sort_by...)."
            )
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


def _reorder_indices(old: list | tuple, new: list | tuple) -> tuple[int]:
    new_indices = tuple(old.index(o) for o in new)
    return new_indices


def convert_to_axes(axes, container=tuple):
    """Convert `obs` to the list of obs, also if it is a
    :py:class:`~ZfitSpace`. Return None if axes is None.

    Raises
        TypeError: if the axes are not int
    """
    if axes is None:
        return axes
    axes = convert_to_container(value=axes, container=container)

    if isinstance(axes, ZfitDimensional):
        new_axes = axes.axes
    else:
        new_axes = []
        for axis in axes:
            if not isinstance(axis, int):
                raise TypeError(f"Axes have to be int, not {axis} as in {axes}")
            else:
                new_axes.append(axis)
    return container(new_axes)


def convert_to_obs_str(obs, container=tuple):
    """Convert `obs` to the list of obs, also if it is a
    :py:class:`~ZfitSpace`. Return None if obs is None.

    Raises:
        TypeError: if the observable is not a string
    """
    if obs is None:
        return obs
    if isinstance(obs, ZfitDimensional):
        new_obs = obs.obs

    else:
        obs = convert_to_container(value=obs, container=container)
        new_obs = []
        for ob in obs:
            if not isinstance(ob, str):
                raise TypeError(f"Observables have to be string, not {ob} as in {obs}")
            else:
                new_obs.append(ob)
    return container(new_obs)
