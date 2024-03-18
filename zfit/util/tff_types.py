#  Copyright (c) 2024 zfit
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code was copied directly from tf_quant_finance.math.root_search.brent.
# The repository is deprecated and will be removed in the future.
"""Common data types."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import tensorflow as tf

__all__ = ["BoolTensor", "IntTensor", "RealTensor", "ComplexTensor", "StringTensor"]

tensor_like = (np.ndarray, tf.Tensor, tf.TensorSpec)

# A type that represents a boolean `Tensor`
BoolTensor = TypeVar("BoolTensor", *tensor_like)

# A type that represents int32 or int64 `Tensor`s
IntTensor = TypeVar("IntTensor", *tensor_like)

# A type that represents float or double `Tensor`s
RealTensor = TypeVar("RealTensor", *tensor_like)

# A type that represents a float `Tensor`
FloatTensor = TypeVar("FloatTensor", *tensor_like)

# A type that represents a double `Tensor`
DoubleTensor = TypeVar("DoubleTensor", *tensor_like)

# 'A type that represents complex64 or complex128 `Tensor`s
ComplexTensor = TypeVar("ComplexTensor", *tensor_like)

# A type that represents a string `Tensor`
StringTensor = TypeVar("StringTensor", *tensor_like)
