import copy

import tensorflow as tf


class RunManager:

    def __init__(self):
        """Handle the resources and runtime specific options. The `run` method is equivalent to `sess.run`"""
        self._sess = None
        self._sess_kwargs = {}

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
