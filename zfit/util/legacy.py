#  Copyright (c) 2021 zfit

try:
    from tensorflow.python.util.deprecation import deprecated, deprecated_args
except ImportError:  # TF == 2.2
    from tensorflow.python import deprecated, deprecated_args
