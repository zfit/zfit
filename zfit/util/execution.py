#  Copyright (c) 2020 zfit

import contextlib
import multiprocessing
import os
import sys
import warnings
from typing import List, Union, Optional

import numpy as np
import tensorflow as tf
from tensorflow_core.python import deprecated

from .container import DotDict, is_container


class RunManager:
    DEFAULT_MODE = {'graph': 'auto',
                    'auto_grad': True}

    def __init__(self, n_cpu='auto'):
        """Handle the resources and runtime specific options. The `run` method is equivalent to `sess.run`"""
        self.MAX_CHUNK_SIZE = sys.maxsize
        self.chunking = DotDict()
        self._cpu = []
        self._n_cpu = None
        self._inter_cpus = None
        self._intra_cpus = None
        self._strict = False
        self.numeric_checks = True
        self._mode = self.DEFAULT_MODE.copy()
        self.set_n_cpu(n_cpu=n_cpu)

        # HACK
        self._enable_parameter_autoconversion = True
        # HACK END

        # set default values
        self.chunking.active = False  # not yet implemented the chunking...
        self.chunking.max_n_points = 1000000

    @property
    def mode(self):
        return self._mode

    @property
    def chunksize(self):
        if self.chunking.active:
            return self.chunking.max_n_points
        else:
            return self.MAX_CHUNK_SIZE

    @property
    def n_cpu(self):
        return len(self._cpu)

    def set_n_cpu(self, n_cpu: Union[str, int] = 'auto', strict: bool = False) -> None:
        """Set the number of cpus to be used by zfit. For more control, use `set_cpus_explicit`.

        Args:
            n_cpu: Number of cpus, will be the number for inter-op parallelism
            strict: If strict, sets intra parallelism to 1
        """
        if n_cpu == 'auto':
            try:
                cpu = sorted(os.sched_getaffinity(0))
            except AttributeError:
                cpu = range(multiprocessing.cpu_count())
                warnings.warn("Not running on Linux. Determining available cpus for thread can fail"
                              "and be overestimated. Workaround (only if too many cpus are used):"
                              "`zfit.run.set_n_cpu(your_cpu_number)`")
        elif isinstance(n_cpu, int):
            cpu = range(n_cpu)
        self._cpu = ['dummy_cpu{}'.format(i) for i in cpu]
        n_cpu = len(cpu)
        if strict ^ self._strict:
            intra = 1 if strict else 2
            inter = n_cpu
            self.set_cpus_explicit(intra=intra, inter=inter)

    def set_cpus_explicit(self, intra: int, inter: int) -> None:
        """Set the number of threads (cpus) used for inter-op and intra-op parallelism

        Args:
            intra: Number of threads used to perform an operation. For larger operations, e.g. large Tensors, this
                is usually beneficial to have >= 2.
            inter: Parallelization on the level of ops. This is beneficial, if many operations can be computed
                independently in parallel.
        """
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            tf.config.threading.set_inter_op_parallelism_threads(inter)
            self._n_cpu = inter + intra
        except RuntimeError as err:
            raise RuntimeError("Cannot set the number of cpus after initialization, has to be at the beginning."
                               f" Original message: {err}")

    @contextlib.contextmanager
    def aquire_cpu(self, max_cpu: int = -1) -> List[str]:
        if isinstance(max_cpu, int):
            if max_cpu < 0:
                max_cpu = max((self.n_cpu + 1 + max_cpu, 0))  # -1 means all
            if max_cpu == 0:
                cpu = []
            else:
                n_cpu = min((max_cpu, self.n_cpu))

                cpu = self._cpu[-n_cpu:]
                self._cpu = self._cpu[:-n_cpu]

            yield cpu
            self._cpu.extend(cpu)

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise RuntimeError("Why kwargs provided?")

        flattened_args = tf.nest.flatten(args)
        evaluated_args = [eval_object(arg) for arg in flattened_args]
        values = tf.nest.pack_sequence_as(args, flat_sequence=evaluated_args)

        was_container = is_container(args[0]) and not isinstance(args[0], np.ndarray, )
        if not was_container and values:
            values = values[0]
        return values

    @staticmethod
    @deprecated(date=None, instructions="Use `set_mode(graph=False)`")
    def experimental_enable_eager(eager: bool = False):
        """DEPRECEATED! Enable eager makes tensorflow run like numpy. Useful for debugging.

        Do NOT directly mix it with Numpy (and if, also enable the numberical gradient).

        This can BREAK in the future.

        """
        from zfit import jit
        jit._set_all(not eager)

    def set_mode(self, graph: Optional[Union[bool, str, dict]] = None, auto_grad: Optional[bool] = None):
        """Set the policy for graph building and the usage of automatic vs numerical gradients.



        Args:
            graph:
            auto_grad:
        """
        jit_mode = graph
        from .. import jit as jit_obj

        if graph is None and auto_grad is None:
            raise ValueError("Both graph and auto_grad are None. Specify at least one.")

        if jit_mode is True:
            jit_obj._set_all(True)
        elif jit_mode is False:
            jit_obj._set_all(False)
        elif jit_mode == 'auto':
            jit_obj._set_default()
        elif isinstance(jit_mode, dict):
            jit_obj._update_allowed(jit_mode)
        elif jit_mode is not None:
            raise ValueError(f"{jit_mode} is not a valid keyword to the `jit` behavior. Use either "
                             f"True, False, 'default' or a dict. You can read more about it in the docs.")
        if jit_mode is not None:
            self._mode['graph'] = graph

        if auto_grad is not None:
            from zfit import settings
            settings.options.numerical_grad = not auto_grad
            self._mode['auto_grad'] = auto_grad

    def set_default_mode(self):
        """Reset the mode to the default of `graph` = 'auto' and `auto_grad` = True."""
        self.set_mode(**self.DEFAULT_MODE)

    def clear_graph_cache(self):
        from zfit.util.cache import clear_graph_cache
        clear_graph_cache()

    def assert_executing_eagerly(self):
        if not tf.executing_eagerly():
            raise RuntimeError("This code is ont supposed to run inside a graph.")

    @property
    @deprecated(None, "Use `not run.mode['graph']")
    def experimental_is_eager(self):
        return not self.mode['graph']

    @deprecated(date=None, instructions="Use clear_graph_caches instead.")
    def experimental_clear_caches(self):
        self.clear_graph_cache()


def eval_object(obj: object) -> object:
    from zfit.core.parameter import BaseComposedParameter
    if isinstance(obj, BaseComposedParameter):  # currently no numpy attribute. Should we add this?
        obj = obj.value()
    if tf.is_tensor(obj):
        return obj.numpy()
    else:
        return obj
