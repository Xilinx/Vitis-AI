# Copyright 2019 Xilinx Inc.
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
"""Vitis 8-bit layout transformation for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer

keras = tf.keras


class Vitis8BitOptimizeLayoutTransform(
    quantize_layout_transform.QuantizeLayoutTransform):
  """Vitis 8Bit model pre-quantization optimize transformations."""

  def apply(self, model, layer_quantize_map, remove_dropout, fold_conv_bn,
            fold_bn, replace_relu6, include_cle, cle_steps):
    """Implement vitis 8-bit transforms.

    All the transforms should not break the float model structure, and
    the output of the transformed model should be consistant with the float 
    model.
    """

    transforms = []
    if remove_dropout:
      transforms.append(vitis_optimize_transforms.RemoveDropout())

    if fold_conv_bn:
      transforms.append(vitis_optimize_transforms.Conv2DBatchNormFold())

    if fold_bn:
      transforms.append(vitis_optimize_transforms.BatchNormFold())

    if replace_relu6:
      transforms.append(vitis_optimize_transforms.ReplaceReLU6WithReLU())

    transformed_model, layer_quantize_map = model_transformer.ModelTransformer(
        model, transforms, None, layer_quantize_map).transform()

    # Cross Layer Equalization
    if include_cle:
      cle_transforms = [
          vitis_optimize_transforms.ConvConvCLE(),
          vitis_optimize_transforms.ConvActConvCLE(),
      ]
      progbar = keras.utils.Progbar(cle_steps)
      print('[INFO] Start CrossLayerEqualization...')
      for i in range(cle_steps):
        progbar.update(i + 1)
        tmp_model, tmp_layer_quantize_map = transformed_model, layer_quantize_map
        transformed_model, layer_quantize_map = model_transformer.ModelTransformer(
            tmp_model, cle_transforms, None,
            tmp_layer_quantize_map).transform()
      print('[INFO] CrossLayerEqualization Done.')
    return transformed_model, layer_quantize_map


class Vitis8BitQuantizeLayoutTransform(
    quantize_layout_transform.QuantizeLayoutTransform):
  """Vitis 8Bit model transformations."""

  def apply(self, model, layer_quantize_map, quantize_registry, mode):
    """Implement vitis 8-bit transforms.

    Currently this means the following.
      1. Pull activations into layers, and apply fuse activations. (TODO)
      2. Modify range in incoming layers for Concat. (TODO)
      3. Fuse Conv2D/DepthwiseConv2D + BN into single layer.

    Args:
      model: Keras model to be quantized.
      layer_quantize_map: Map with keys as layer names, and values as dicts
        containing custom `QuantizeConfig`s which may have been passed with
        layers.
      quantize_registry: QuantizeRegistry object containing the quantize configs
        for each layer.
      mode: String object indicating the mode of the quantized model.

    Returns:
      (Transformed Keras model to better match TensorFlow Lite backend, updated
      layer quantize map.)
    """

    transforms = [
        vitis_8bit_quantize_transforms.InputLayerQuantize(
            quantize_registry.get_input_quantizer(), mode),
        vitis_8bit_quantize_transforms.ConvActivationQuantize(),
        vitis_8bit_quantize_transforms.AddActivationQuantize(),
    ]
    return model_transformer.ModelTransformer(model, transforms,
                                              set(layer_quantize_map.keys()),
                                              layer_quantize_map).transform()
