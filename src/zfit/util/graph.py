#  Copyright (c) 2025 zfit

from __future__ import annotations

import typing

from .temporary import TemporarilySet

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


class JIT:
    def _set_all(self, enable: bool = True):
        new_values = dict.fromkeys(self._get_allowed(), enable)

        def getter():
            return self._get_allowed().copy()

        def setter(jit_types):
            self._update_allowed(jit_types)

        return TemporarilySet(getter=getter, setter=setter, value=new_values)

    def _set_default(self):
        from zfit import z  # noqa: PLC0415

        new_values = z.zextension.FunctionWrapperRegistry._DEFAULT_DO_JIT_TYPES.copy()

        for key in self._get_allowed():
            if key not in new_values:
                new_values[key] = new_values[key]  # default dict will explicitly set the default value

        def getter():
            return self._get_allowed().copy()

        def setter(jit_types):
            self._update_allowed(jit_types)

        return TemporarilySet(getter=getter, setter=setter, value=new_values)

    def _update_allowed(self, update_jit):
        from zfit import z  # noqa: PLC0415

        z.zextension.FunctionWrapperRegistry.do_jit_types.update(update_jit)

    def _get_allowed(self):
        from zfit import z  # noqa: PLC0415

        return z.zextension.FunctionWrapperRegistry.do_jit_types

    @property
    def experimental_is_eager(self):
        from ..settings import run  # noqa: PLC0415

        return run.mode["graph"]


jit = JIT()  # singleton
