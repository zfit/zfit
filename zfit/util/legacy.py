#  Copyright (c) 2020 zfit

try:
    from tensorflow.python import deprecated
except ImportError:  # TF < 2.2
    from tensorflow_core.python import deprecated
