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
"""Module containing quantization code built on Keras abstractions."""
# pylint: disable=g-bad-import-order

# submodules
from tensorflow_model_optimization.python.core.api.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.api.quantization.keras import vitis

# quantize all layers with default quantization implementation.
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_model

# vitis quantize
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize

# quantize some layers with default or custom quantization implementation.
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_model
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_apply

# quantize with custom quantization parameterization or implementation, or
# handle custom Keras layers.
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

# Deserialize quantized model for Keras h5 format.
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope

# pylint: enable=g-bad-import-order
