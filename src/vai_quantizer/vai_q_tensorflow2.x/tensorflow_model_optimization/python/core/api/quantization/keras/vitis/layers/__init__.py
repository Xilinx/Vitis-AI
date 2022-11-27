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
"""Module containing Vitis layers."""

from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_pooling
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_conv_bn

# Layers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.vitis_quantize import *
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.vitis_activation import *
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.vitis_pooling import *
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.vitis_conv_bn import *
