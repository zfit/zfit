#  Copyright (c) 2019 zfit

from typing import List

import tensorflow as tf


# TODO(Mayou36): make not recursive
def all_parents(op, current_obs=None):
    if current_obs is None:
        current_obs = set()
    ops = set(input_.op for input_ in op.inputs if input_.op not in current_obs)
    current_obs = current_obs.union(ops)
    return ops.union(*(all_parents(op, current_obs=current_obs) for op in ops))


def get_dependents_auto(tensor: tf.Tensor, candidates: List[tf.Tensor]) -> List[tf.Tensor]:
    """Return the nodes in `candidates` that `tensor` depends on.

    Args:
        tensor ():
        candidates ():
    """
    try:
        dependent_ops = all_parents(tensor.op)
    except RuntimeError as error:
        raise ValueError("Tensor too deeply nested, recursion limit exceeded. In the future,"
                         "implementation will be different and any dependents can be found."
                         "Currently, specify dependents explicitly if needed."
                         "Orignal Error: {}".format(error))
    dependent_candidates = [cand for cand in candidates if cand.op in dependent_ops]
    return dependent_candidates
