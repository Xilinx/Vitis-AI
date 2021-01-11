# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for the `tensor_encoding` package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.py_utils import assert_compatible
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.py_utils import merge_dicts
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.py_utils import OrderedEnum
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.py_utils import split_dict_py_tf
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.py_utils import static_or_dynamic_shape

from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.tf_utils import fast_walsh_hadamard_transform
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.tf_utils import pack_into_int
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.tf_utils import random_floats
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.tf_utils import random_floats_cmwc
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.tf_utils import random_signs
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.tf_utils import random_signs_cmwc
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils.tf_utils import unpack_from_int
