# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module containing Vitis quantization modules."""

# vitis quantize main
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize

# vitis inspect main
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_inspect

# submodules
from tensorflow_model_optimization.python.core.api.quantization.keras.vitis import layers

# quantize_ops
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_ops

# quantize_scope
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vitis_quantize import quantize_scope

# utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils
