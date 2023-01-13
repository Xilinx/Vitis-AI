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
"""Testing utilities for the `tensor_encoding` package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import AdaptiveNormalizeEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import aggregate_state_update_tensors
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import BaseEncodingStageTest
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import get_tensor_with_random_shape
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import is_adaptive_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import PlusOneEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import PlusOneOverNEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import PlusRandomNumEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import RandomAddSubtractOneEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import ReduceMeanEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import SignIntFloatEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import SimpleLinearEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import TestData
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing.test_utils import TimesTwoEncodingStage
