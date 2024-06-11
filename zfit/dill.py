#  Copyright (c) 2024 zfit
from __future__ import annotations

import functools as _functools
import io
from typing import Optional

import dill as __dill

# noinspection PyUnresolvedReferences
from dill import *  # noqa: F403

_NONE = object()


class ZfitDillDumpError(Exception):
    pass


class ZfitDillLoadError(Exception):
    pass


def __retry_with_gc(func, kwargs, *, max_retries: int | None = None):
    """Helper function to retry a function with garbage collection if necessary.

    Main intended use is to retry a function that failed due to a graph that cannot be pickled. This can happen
    if the graph is not garbage collected and still in memory. This function will retry the function after
    garbage collection. For the dill.dump and dill.dumps functions. Also catches the ipyopt error and raises a
    more informative error message.

    Args:
        func: Function to retry
        kwargs: Keyword arguments to pass to the function
        max_retries: Maximum number of retries. If None, defaults to 2.
    """
    redo = False
    if max_retries is None:
        max_retries = 2
    max_retries_reached = False
    result_dilled = _NONE
    funcgraph_phrase = "cannot pickle 'FuncGraph' object"
    dictchange_phrase = "dictionary changed size during iteration"
    ipyopt_phrase = "cannot pickle 'ipyopt.Problem' object"
    weakgone_phrase = "weak object has gone away"
    ipyopt_error_msg = (
        f"Cannot pickle {kwargs.get('obj')}, as it contains the ipyopt minimizer, which cannot be pickled/dilled."
        f"This is a known limitation. Use another method to dump the object: if you're dumping a FitResult, call `result.freeze()`"
        f" on it first, or use `zfit.hs3` for human readable serialization."
    )
    try:
        result_dilled = func(**kwargs)
    except TypeError as error:
        if not (redo := funcgraph_phrase in str(error) or weakgone_phrase in str(error)) or (
            max_retries_reached := max_retries <= 0
        ):
            if ipyopt_phrase in str(error):
                from zfit.exception import IpyoptPicklingError

                raise IpyoptPicklingError(ipyopt_error_msg) from error
            if max_retries_reached:
                msg = (
                    f"Max retries reached when dumping {kwargs}, error still occurred. If this happens more than once, please report on github in an issue."
                    f" The following error is not the first one to occur, to debug, please rerun with `max_retries=0` and increase the number`."
                )
                raise RuntimeError(msg) from error
            raise error
    except RuntimeError as error:
        if dictchange_phrase in str(error):
            redo = True
        else:
            raise error
    # need to get out, otherwise "error" has a reference to graph and gc can't collect it
    if redo:
        import gc

        gc.collect()
        result_dilled = __retry_with_gc(func, kwargs, max_retries=max_retries - 1)

    assert result_dilled is not _NONE, "result_dilled should have been set, logic error in zfit/dill.py"
    return result_dilled


@_functools.wraps(__dill.dumps)
def dumps(
    obj,
    protocol=None,
    byref=False,
    fmode=None,
    recurse=None,
    *,
    max_retries: Optional[int | bool] = None,
    verify: Optional[bool] = None,
    **kwds,
):
    """Wrapper around :py:func`dill.dumps` that helps dumping zfit objects as it retries with garbage collection if
    necessary.

    .. note::

        This function can dump any python object, including list/dicts of zfit objects. If two objects are in any way
        related, for example they share parameters, they *have* to be dumped together. Otherwise, the parameters will
        be duplicated when loading, with the same name, which will fail to run the fit.

    Additional argument max_retries: Maximum number of retries if it fails (can occur due to garbage collector required to run first).
        If None, defaults to 2.

    Original docstring:
    {docstring}
    """
    redo = False
    if max_retries is None:
        max_retries = 2
    elif max_retries < 0:
        msg = "max_retries has to be >= 0"
        raise ValueError(msg)
    if verify is None:
        verify = True
    kwargs = dict(obj=obj, protocol=protocol, byref=byref, fmode=fmode, recurse=recurse, **kwds)
    for ntry in range(max_retries + 1):
        out = __retry_with_gc(func=__dill.dumps, kwargs=kwargs, max_retries=max_retries)
        if verify:
            try:
                loads(out, max_retries=max_retries)
            except ZfitDillLoadError as error:
                if ntry == max_retries:
                    msg = "Tried to verify dumps by loading but failed."
                    raise ZfitDillDumpError(msg) from error
                redo = True
                from zfit import run

                run.clear_graph_cache(call_gc=True)
        if not redo:
            break

    return out


dumps.__doc__ = dumps.__doc__.format(docstring=__dill.dumps.__doc__)


@_functools.wraps(__dill.dump)
def dump(
    obj,
    file,
    protocol=None,
    byref=None,
    fmode=None,
    recurse=None,
    *,
    max_retries: Optional[int | bool] = None,
    verify: Optional[bool] = None,
    **kwds,
):
    """Wrapper around :py:func`dill.dump` that helps dumping zfit objects as it retries with garbage collection if
    necessary.

    .. note::

        This function can dump any python object, including list/dicts of zfit objects. If two objects are in any way
        related, for example they share parameters, they *have* to be dumped together. Otherwise, the parameters will
        be duplicated when loading, with the same name, which will fail to run the fit.

    Additional arguments
        - max_retries: Maximum number of retries if it fails (can occur due to garbage collector required to run first).
           If None, defaults to 2.
        - verify: **requires `file` to be in write *and* read mode (i.e. `w+b` instead of `wb` only).**
            This *can* clear the graph cache and should not have an effect on the results but may significantly slow down
            subsequent calls to the same zfit fits, as the graph needs to be recompiled.
            If True, the dump will be verified by loading it again. If it fails, the dump will be redone.
            If it fails after the maximum number of retries, a ZfitDillDumpError will be raised.

    Original docstring:
    {docstring}
    """
    redo = False
    if max_retries is None:
        max_retries = 2
    elif max_retries < 0:
        msg = "max_retries has to be >= 0"
        raise ValueError(msg)
    if verify is None:
        verify = True
    kwargs = dict(obj=obj, file=file, protocol=protocol, byref=byref, fmode=fmode, recurse=recurse, **kwds)
    initial_position = file.tell()
    for ntry in range(max_retries + 1):
        out = __retry_with_gc(func=__dill.dump, kwargs=kwargs, max_retries=max_retries)
        if verify:
            post_out_position = file.tell()
            file.seek(initial_position)
            try:
                load(file=file, max_retries=max_retries)
            except ZfitDillLoadError as error:
                if ntry == max_retries:
                    msg = "Tried to verify dumps by loading but failed."
                    raise ZfitDillDumpError(msg) from error
                redo = True
                from zfit import run

                run.clear_graph_cache(call_gc=True)
                file.seek(initial_position)  # reset to initial position
            else:
                redo = False
                file.seek(post_out_position)  # we're done verifying
        if not redo:
            break
    return out


def __retry_with_graphclear(func, kwargs, max_retries, _file_to_reset=None):
    original_error = None

    # make sure to reset the file to the initial position after an error was raised
    # and the read is re-attempted
    initial_position = None
    if _file_to_reset is not None:
        initial_position = _file_to_reset.tell()
    for i in range(max_retries + 1):
        if _file_to_reset is not None:
            _file_to_reset.seek(initial_position)
        try:
            out = func(**kwargs)
        except io.UnsupportedOperation as error:
            if "read" in str(error):
                msg = (
                    "Tried to verify (use `verify=False` to not verify) dumping by loading but failed (most likely because the file was opened in write mode only. "
                    "Try to open it in write and read mode, for example by changing `wb` to `w+b`.)"
                )
                raise io.UnsupportedOperation(msg) from error
        except Exception as error:
            if original_error is None:
                original_error = error
            if i == max_retries:
                msg = (
                    f"Max retries reached when loading {kwargs}, error still occurred. Original error {original_error}"
                )
                raise ZfitDillLoadError(msg) from error
            from zfit import run

            run.clear_graph_cache(call_gc=True)
        else:
            break
    return out


@_functools.wraps(__dill.loads)
def loads(str, *, max_retries: Optional[int | bool] = None, **kwds):
    """Wrapper around :py:func`dill.loads`that helps loading zfit objects as it retries with graph clearing if
    necessary.

    .. warning ::

        This function *may* clears the cached graph/traced functions. This should not have an effect on the
        results but may significantly slow down subsequent calls to the same zfit fits, as the graph needs to be
        recompiled.

    Additional argument max_retries: Maximum number of retries if it fails (can occur due to garbage collector required to run first).
        If None, defaults to 2.

    Original docstring:
    {docstring}
    """
    if max_retries is None:
        max_retries = 2
    elif max_retries < 0:
        msg = "max_retries has to be >= 0"
        raise ValueError(msg)
    kwargs = dict(str=str, **kwds)
    return __retry_with_graphclear(func=__dill.loads, kwargs=kwargs, max_retries=max_retries)


@_functools.wraps(__dill.load)
def load(file, *, max_retries: Optional[int | bool] = None, **kwds):
    """Wrapper around :py:func`dill.load`that helps loading zfit objects as it retries with graph clearing if necessary.

    .. warning ::

        This function *may* clears the cached graph/traced functions. This should not have an effect on the
        results but may significantly slow down subsequent calls to the same zfit fits, as the graph needs to be
        recompiled.

    Additional argument max_retries: Maximum number of retries if it fails (can occur due to garbage collector required to run first).
        If None, defaults to 2.

    Original docstring:
    {docstring}
    """
    if max_retries is None:
        max_retries = 2
    elif max_retries < 0:
        msg = "max_retries has to be >= 0"
        raise ValueError(msg)
    kwargs = dict(file=file, **kwds)
    return __retry_with_graphclear(func=__dill.load, kwargs=kwargs, max_retries=max_retries, _file_to_reset=file)


dump.__doc__ = dump.__doc__.format(docstring=__dill.dump.__doc__)
