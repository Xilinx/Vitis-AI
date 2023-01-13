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
"""Core parts of the `tensor_encoding` package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.core_encoder import Encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.core_encoder import EncoderComposer

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.encoding_stage import AdaptiveEncodingStageInterface
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.encoding_stage import EncodingStageInterface
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.encoding_stage import StateAggregationMode
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.encoding_stage import tf_style_adaptive_encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.encoding_stage import tf_style_encoding_stage

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.gather_encoder import GatherEncoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core.simple_encoder import SimpleEncoder
