from typing import List

import tensorflow as tf


def parents(op):
    return set(input.op for input in op.inputs)


# TODO(Mayou36): make not recursive
def all_parents(op, current_obs=None):
    if current_obs is None:
        current_obs = set()
    ops = set(input_.op for input_ in op.inputs if input_.op not in current_obs)
    current_obs = current_obs.union(ops)
    return ops.union(*(all_parents(op, current_obs=current_obs) for op in ops))


def children(op):
    return set(op for out in op.outputs for op in out.consumers())


def get_dependents(tensor: tf.Tensor, candidates: List[tf.Tensor]) -> List[tf.Tensor]:
    """Return the nodes in `candidates` that `tensor` depends on.

    Args:
        tensor ():
        candidates ():
    """

    dependent_ops = all_parents(tensor.op)
    dependent_candidates = [cand for cand in candidates if cand.op in dependent_ops]
    return dependent_candidates


# def print_tf_graph(graph):
#     """Prints tensorflow graph in dictionary form."""
#     for node in graph:
#         for child in graph[node]:
#             print("%s -> %s" % (node.name, child.name))


if __name__ == '__main__':
    a = tf.distributions.Normal(1., 3.).sample() * 5.
    var1 = tf.get_variable('a1', 1.)
    var2 = tf.get_variable('a2', 2.)
    var3 = tf.get_variable('a3', 3.)
    b = tf.constant(3.) + 4 * var1
    c = 5. * b
    d = c + b * var2
    e = c * 3.
    print(get_dependents(e, [b, c, d, var1, var2, var3]))
    print(get_dependents(e, [var1, var2, var3]))
