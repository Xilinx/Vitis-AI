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
"""Quantization registry which specifies how layers should be quantized.

Module: tfmot.quantization.keras
"""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class QuantizeRegistry(object):
  """ABC interface which specifies how layers should be quantized.

  The Registry is designed to function as a repository of `QuantizeConfig`s
  linked to layers. The idea is that while applying quantization to the various
  layers within a Keras model, the registry can be used to query which
  `QuantizeConfig` can be used to quantize a specific `layer`. The
  `QuantizeConfig` itself contains information to quantize that specific
  layer.

  We provide a default registry for built-in Keras layers, but implementing this
  interface allows users the ability to write their own custom registries
  specific to their needs. It can also be extended to be used for any Keras
  layer, such as custom Keras layers.
  """

  @abc.abstractmethod
  def get_quantize_config(self, layer):
    """Returns the quantization config for the given layer.

    Args:
      layer: input layer to return quantize config for.

    Returns:
      Returns the QuantizeConfig for the given layer.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
