import contextlib
import copy
import multiprocessing
import os
import sys
from typing import List
import warnings

import tensorflow as tf

import zfit
from .temporary import TemporarilySet
from .container import DotDict


class RunManager:

    def __init__(self, n_cpu='auto'):
        """Handle the resources and runtime specific options. The `run` method is equivalent to `sess.run`"""
        self.MAX_CHUNK_SIZE = sys.maxsize
        self.sess = None
        self._sess_kwargs = {}
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
        return self.sess.run(*args, **kwargs)

    # def close(self):
    #     """Closes the current session."""
    def reset(self):
        if self._sess is not None:
            self.sess.close()
        tf.reset_default_graph()

    def create_session(self, *args, close_current=True, reset_graph=False, **kwargs):
        """Create a new session (or replace the current one). Arguments will overwrite the already set arguments.

        Args:
            close_current (bool): Closes the current open session before replacement. Has no effect if
                no session was created before.
            reset_graph (bool): Resets the current (default) graph before creating a new :py:class:`tf.Session`.
            *args ():
            **kwargs ():

        Returns:
            :py:class:`tf.Session`
        """
        sess_kwargs = copy.deepcopy(self._sess_kwargs)
        sess_kwargs.update(kwargs)
        if close_current and self._sess is not None:
            self.sess.close()
        if reset_graph:
            tf.reset_default_graph()
            from zfit.core.parameter import ZfitParameterMixin
            ZfitParameterMixin._existing_names = set()  # TODO(Mayou36): better hook for reset?
        self.sess = tf.Session(*args, **sess_kwargs)

        from ..settings import ztypes
        tf.get_variable_scope().set_use_resource(True)
        tf.get_variable_scope().set_dtype(ztypes.float)

        return self.sess

    @property
    def sess(self):
        if self._sess is None:
            self.create_session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value


class SessionHolderMixin:

    def __init__(self, *args, **kwargs):
        """Creates a `self.sess` attribute, a setter `set_sess` (with a fallback to the zfit default session)."""
        super().__init__(*args, **kwargs)
        self._sess = None

    def set_sess(self, sess: tf.Session):
        """Set the session (temporarily) for this instance. If None, the auto-created default is taken.

        Args:
            sess (tf.Session):
        """
        if not isinstance(sess, tf.Session):
            raise TypeError("`sess` has to be a TensorFlow Session but is {}".format(sess))

        def getter():
            return self._sess  # use private attribute! self.sess creates default session

        def setter(value):
            self.sess = value

        return TemporarilySet(value=sess, setter=setter, getter=getter)

    @property
    def sess(self):
        sess = self._sess
        if sess is None:
            sess = zfit.run.sess
        return sess

    @sess.setter
    def sess(self, sess: tf.Session):
        self._sess = sess
