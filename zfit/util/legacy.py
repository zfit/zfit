#  Copyright (c) 2020 zfit

try:
    from tensorflow.python.util.deprecation import deprecated
except ImportError:  # TF == 2.2
    from tensorflow.python import deprecated
