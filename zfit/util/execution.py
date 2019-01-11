import contextlib
import copy
import os
import sys
from typing import List

import tensorflow as tf

from zfit.util.temporary import TemporarilySet
from .container import DotDict


class RunManager:

    def __init__(self, n_cpu='auto'):
        """Handle the resources and runtime specific options. The `run` method is equivalent to `sess.run`"""
        self.MAX_CHUNK_SIZE = sys.maxsize
        self._sess = None
        self._sess_kwargs = {}
        self.chunking = DotDict()
        self._cpu = []

        self.set_n_cpu(n_cpu=n_cpu)

        # set default values
        self.chunking.active = False  # not yet implemented the chunking...
        self.chunking.max_n_points = 100000

    def auto_initialize(self, variable: tf.Variable):
        self(variable.initializer)

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
            cpu = sorted(os.sched_getaffinity(0))
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
        return self.sess.run(*args, **kwargs)

    def create_session(self, *args, **kwargs):
        """Create a new session (or replace the current one). Arguments will overwrite the already set arguments.

        Args:
            *args ():
            **kwargs ():

        Returns:
            :py:class:`tf.Session`
        """
        sess_kwargs = copy.deepcopy(self._sess_kwargs)
        sess_kwargs.update(kwargs)
        self.sess = tf.Session(*args, **sess_kwargs)
        return self.sess

    @property
    def sess(self):
        if self._sess is None:
            self.create_session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value
