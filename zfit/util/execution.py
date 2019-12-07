#  Copyright (c) 2019 zfit

import contextlib
import copy
import multiprocessing
import os
import sys
from typing import List
import warnings

import numpy as np
import tensorflow as tf



import zfit
from .temporary import TemporarilySet
from .container import DotDict, is_container, convert_to_container


class RunManager:

    def __init__(self, n_cpu='auto'):
        """Handle the resources and runtime specific options. The `run` method is equivalent to `sess.run`"""
        self.MAX_CHUNK_SIZE = sys.maxsize
        self.chunking = DotDict()
        self._cpu = []
        self.numeric_checks = True

        self.set_n_cpu(n_cpu=n_cpu)

        # HACK
        self._enable_parameter_autoconversion = True
        # HACK END

        # set default values
        self.chunking.active = False  # not yet implemented the chunking...
        self.chunking.max_n_points = 1000000

    def auto_initialize(self, variable: tf.Variable):
        pass

    @property
    def chunksize(self):
        if self.chunking.active:
            return self.chunking.max_n_points
        else:
            return self.MAX_CHUNK_SIZE

    @property
    def n_cpu(self):
        return len(self._cpu)

    def set_n_cpu(self, n_cpu='auto'):
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
        raise RuntimeError("Remove and replace by `.numpy()`")
        if kwargs:
            raise RuntimeError("Why kwargs provided? Still under conversion from TF 1.x to 2.x")
        was_container = is_container(args[0]) and not isinstance(args[0], np.ndarray, )
        to_convert = convert_to_container(args[0])
        values = [arg.numpy() for arg in to_convert if isinstance(arg, (tf.Tensor, tf.Variable))]
        if not was_container and values:
            values = values[0]
        return values


