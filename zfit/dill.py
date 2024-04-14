#  Copyright (c) 2024 zfit
from __future__ import annotations

import functools as _functools

import dill as __dill

# noinspection PyUnresolvedReferences
from dill import *  # noqa: F403

_NONE = object()


def _retry_with_gc(func, kwargs):
    redo = False
    result_dilled = _NONE
    phrase = "cannot pickle 'FuncGraph' object"
    ipyopt_phrase = "cannot pickle 'ipyopt.Problem' object"
    ipyopt_error_msg = (
        f"Cannot pickle {kwargs.get('obj')}, as it contains the ipyopt minimizer, which cannot be pickled/dilled."
        f"This is a known limitation. Use another method to dump the object: if you're dumping a FitResult, call `result.freeze()`"
        f" on it first, or use `zfit.hs3` for human readable serialization."
    )
    try:
        result_dilled = func(**kwargs)
    except TypeError as error:
        if not (redo := phrase in str(error)):
            if ipyopt_phrase in str(error):
                from zfit.exception import IpyoptPicklingError

                raise IpyoptPicklingError(ipyopt_error_msg) from error
            raise
    # need to get out, otherwise "error" has a reference to graph and gc can't collect it
    if redo:
        import gc

        gc.collect()
        try:
            result_dilled = func(**kwargs)
        except TypeError as error:
            if phrase in str(error):
                msg = "Error that graph cannot be pickled occurred twice. Please report this issue in zfit on github."
                raise RuntimeError(msg) from error
            if ipyopt_phrase in str(error):
                from zfit.exception import IpyoptPicklingError

                raise IpyoptPicklingError(ipyopt_error_msg) from error
            raise

    assert result_dilled is not _NONE, "result_dilled should have been set, logic error in zfit/dill.py"
    return result_dilled


@_functools.wraps(__dill.dumps)
def dumps(obj, protocol=None, byref=None, fmode=None, recurse=None, **kwds):
    """Wrapper around :py:func`dill.dumps` that helps dumping zfit objects as it retries with garbage collection if
    necessary.

    Original docstring:
    {docstring}
    """
    kwargs = dict(obj=obj, protocol=protocol, byref=byref, fmode=fmode, recurse=recurse, **kwds)
    return _retry_with_gc(func=__dill.dumps, kwargs=kwargs)


dumps.__doc__ = dumps.__doc__.format(docstring=__dill.dumps.__doc__)


@_functools.wraps(__dill.dump)
def dump(obj, file, protocol=None, byref=None, fmode=None, recurse=None, **kwds):
    """Wrapper around :py:func`dill.dump` that helps dumping zfit objects as it retries with garbage collection if
    necessary.

    Original docstring:
    {docstring}
    """
    kwargs = dict(obj=obj, file=file, protocol=protocol, byref=byref, fmode=fmode, recurse=recurse, **kwds)
    return _retry_with_gc(func=__dill.dump, kwargs=kwargs)


dump.__doc__ = dump.__doc__.format(docstring=__dill.dump.__doc__)
