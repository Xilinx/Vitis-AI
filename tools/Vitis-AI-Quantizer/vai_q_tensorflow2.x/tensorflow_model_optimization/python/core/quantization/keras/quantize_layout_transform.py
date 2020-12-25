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
"""Abstract Base Class for quantization transformations to a keras model.

Keras models need certain transformations for quantization to exactly match the
behavior of the backend they will be implemented on. This is important for
improving model performance.

This interface abstracts that behavior. Different backends can implement their
own version.

Module: tfmot.quantization.keras
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class QuantizeLayoutTransform(object):
  """Apply transformations to the model.

  Transforms the original model to perform better while quantized
  and also match the layout of the target backend.
  """

  @abc.abstractmethod
  def apply(self, model, layer_quantize_map):
    """Transform model to a quantization friendly model.

    Args:
      model: Keras model to be quantized.
      layer_quantize_map: Map containing list of layers to be quantized and
        associated metadata. Keys are layer names which need to be quantized,
        and values are dicts containing relevant metadata. For example,
        any custom `QuantizeConfig` passed with a layer is present.

    Returns:
      New keras model based on `model` which has been
      transformed to match the layout of the target backend.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
