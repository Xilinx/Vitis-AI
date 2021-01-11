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
"""Experimental implementations of encoding stages.

These encoding stages can possibly change without guarantees on backward
compatibility.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.clipping import ClipByNormEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.clipping import ClipByValueEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.kashin import KashinHadamardEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.misc import DifferenceBetweenIntegersEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.misc import SplitBySmallValueEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.quantization import PerChannelPRNGUniformQuantizationEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.quantization import PerChannelUniformQuantizationEncodingStage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research.quantization import PRNGUniformQuantizationEncodingStage
