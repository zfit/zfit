#  Copyright (c) 2022 zfit

from __future__ import annotations

import tensorflow as tf

from zfit.util.temporary import TemporarilySet


def all_parents(op, current_obs=None):
    if current_obs is None:
        current_obs = set()
    ops = {input_.op for input_ in op.inputs if input_.op not in current_obs}
    current_obs = current_obs.union(ops)
    return ops.union(*(all_parents(op, current_obs=current_obs) for op in ops))


def get_dependents_auto(
    tensor: tf.Tensor, candidates: list[tf.Tensor]
) -> list[tf.Tensor]:
    """Return the nodes in `candidates` that `tensor` depends on.

    Args:
        tensor:
        candidates:
    """
    try:
        dependent_ops = all_parents(tensor.op)
    except RuntimeError as error:
        raise ValueError(
            "Tensor too deeply nested, recursion limit exceeded. In the future,"
            "implementation will be different and any dependents can be found."
            "Currently, specify dependents explicitly if needed."
            "Orignal Error: {}".format(error)
        )
    dependent_candidates = [cand for cand in candidates if cand.op in dependent_ops]
    return dependent_candidates


class JIT:
    def _set_all(self, enable: bool = True):
        new_values = {k: enable for k in self._get_allowed()}

        def getter():
            return self._get_allowed().copy()

        def setter(jit_types):
            self._update_allowed(jit_types)

        return TemporarilySet(getter=getter, setter=setter, value=new_values)

    def _set_default(self):
        from zfit import z

        new_values = z.zextension.FunctionWrapperRegistry._DEFAULT_DO_JIT_TYPES.copy()

        for key in self._get_allowed():
            if key not in new_values:
                new_values[key] = new_values[
                    key
                ]  # default dict will explicitly set the default value

        def getter():
            return self._get_allowed().copy()

        def setter(jit_types):
            self._update_allowed(jit_types)

        return TemporarilySet(getter=getter, setter=setter, value=new_values)

    def _update_allowed(self, update_jit):
        from zfit import z

        z.zextension.FunctionWrapperRegistry.do_jit_types.update(update_jit)

    def _get_allowed(self):
        from zfit import z

        return z.zextension.FunctionWrapperRegistry.do_jit_types

    @property
    def experimental_is_eager(self):
        from ..settings import run

        return run.mode["graph"]


jit = JIT()  # singleton
