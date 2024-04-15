#  Copyright (c) 2024 zfit
from __future__ import annotations

import functools as _functools
from typing import Optional

import dill as __dill

# noinspection PyUnresolvedReferences
from dill import *  # noqa: F403

_NONE = object()


def _retry_with_gc(func, kwargs, *, max_retries: int | None = None):
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
        result_dilled = _retry_with_gc(func, kwargs, max_retries=max_retries - 1)

    assert result_dilled is not _NONE, "result_dilled should have been set, logic error in zfit/dill.py"
    return result_dilled


@_functools.wraps(__dill.dumps)
def dumps(
    obj, protocol=None, byref=None, fmode=None, recurse=None, *, max_retries: Optional[int | bool] = None, **kwds
):
    """Wrapper around :py:func`dill.dumps` that helps dumping zfit objects as it retries with garbage collection if
    necessary.

    Additional argument max_retries: Maximum number of retries if it fails (can occur due to garbage collector required to run first).
        If None, defaults to 2.

    Original docstring:
    {docstring}
    """
    if max_retries is None:
        max_retries = 2
    kwargs = dict(obj=obj, protocol=protocol, byref=byref, fmode=fmode, recurse=recurse, **kwds)
    return _retry_with_gc(func=__dill.dumps, kwargs=kwargs, max_retries=max_retries)


dumps.__doc__ = dumps.__doc__.format(docstring=__dill.dumps.__doc__)


@_functools.wraps(__dill.dump)
def dump(
    obj, file, protocol=None, byref=None, fmode=None, recurse=None, *, max_retries: Optional[int | bool] = None, **kwds
):
    """Wrapper around :py:func`dill.dump` that helps dumping zfit objects as it retries with garbage collection if
    necessary.

    Additional argument max_retries: Maximum number of retries if it fails (can occur due to garbage collector required to run first).
        If None, defaults to 2.

    Original docstring:
    {docstring}
    """
    if max_retries is None:
        max_retries = 2
    kwargs = dict(obj=obj, file=file, protocol=protocol, byref=byref, fmode=fmode, recurse=recurse, **kwds)
    return _retry_with_gc(func=__dill.dump, kwargs=kwargs, max_retries=max_retries)


dump.__doc__ = dump.__doc__.format(docstring=__dill.dump.__doc__)
