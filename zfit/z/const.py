#  Copyright (c) 2020 zfit

import tensorflow as _tf
import numpy as _np

from ..settings import ztypes

inf = _tf.constant(_np.inf, dtype=ztypes.float)
pi = _tf.constant(_np.pi, dtype=ztypes.float)
