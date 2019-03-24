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


if __name__ == '__main__':
    a = tf.distributions.Normal(1., 3.).sample() * 5.
    var1 = tf.get_variable('a1', 1.)
    var2 = tf.get_variable('a2', 2.)
    var3 = tf.get_variable('a3', 3.)
    b = tf.constant(3.) + 4 * var1
    c = 5. * b
    d = c + b * var2
    e = c * 3.
    print(get_dependents_auto(e, [b, c, d, var1, var2, var3]))
    print(get_dependents_auto(e, [var1, var2, var3]))
