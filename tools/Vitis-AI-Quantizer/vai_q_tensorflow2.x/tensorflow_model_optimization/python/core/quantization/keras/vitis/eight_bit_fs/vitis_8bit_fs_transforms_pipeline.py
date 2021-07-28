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
"""Vitis 8-bit float scale transformations pipeline for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_equalization_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.eight_bit_fs import vitis_8bit_fs_quantize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils

TransformsPipeline = transforms_pipeline.TransformsPipeline
keras = tf.keras
logger = common_utils.VAILogger


def _apply_availables(model, configs, available_transforms, candidate_layers,
                      layer_metadata):
  transforms = []
  for key in available_transforms:
    if configs.get(key):
      new_trans = available_transforms.get(key)
      if isinstance(new_trans, list):
        transforms.extend(new_trans)
      else:
        transforms.append(new_trans)

  transformed_model, layer_metadata = model_transformer.ModelTransformer(
      model, transforms, candidate_layers,
      layer_metadata).recursive_transform()
  return transformed_model, layer_metadata


class Vitis8BitFSOptimizeTransformsPipeline(TransformsPipeline):
  """Vitis 8bit float scale pre-quantization optimize model transformations."""

  def apply(self, model, candidate_layers, layer_metadata):
    """Implement vitis 8-bit optimize transforms to make it more quantize-friendly.

    All the transforms should not break the float model structure, and
    the output of the transformed model should be consistant with the float 
    model.
    """
    configs = self.get_configs()

    available_transforms = collections.OrderedDict({
        'remove_dropout': vitis_optimize_transforms.RemoveDropout(),
        'separate_conv_act': vitis_optimize_transforms.SeparateConvAct(),
        'replace_relu6': vitis_optimize_transforms.ReplaceReLU6WithReLU(),
        'replace_tf_op': vitis_optimize_transforms.ReplaceTFOpLayer(),
    })

    transformed_model, layer_metadata = _apply_availables(
        model, configs, available_transforms, candidate_layers, layer_metadata)

    # Train with bn is conflict with fold bn params
    model = transformed_model
    if configs['train_with_bn']:
      transforms = [
          vitis_optimize_transforms.FakeConvBNFold(),
      ]
      transformed_model, _ = model_transformer.ModelTransformer(
          model, transforms, None, None).recursive_transform()
    else:
      available_transforms = {
          'fold_conv_bn': vitis_optimize_transforms.Conv2DBatchNormFold(),
          'fold_bn': vitis_optimize_transforms.BatchNormFold(),
      }
      transformed_model, layer_metadata = _apply_availables(
          model, configs, available_transforms, candidate_layers,
          layer_metadata)

    # Cross Layer Equalization
    if configs['include_cle']:
      forced_cle = configs['forced_cle']
      balance_method = configs['balance_method']
      weight_threshold = configs['weight_threshold']
      cle_transforms = [
          vitis_equalization_transforms.ConvConvCLE(forced_cle, balance_method,
                                                    weight_threshold),
          vitis_equalization_transforms.ConvActConvCLE(forced_cle,
                                                       balance_method,
                                                       weight_threshold),
          vitis_equalization_transforms.ConvReLUConvCLE(forced_cle,
                                                        balance_method,
                                                        weight_threshold),
          vitis_equalization_transforms.ConvReLUPadConvCLE(
              forced_cle, balance_method, weight_threshold)
      ]

      cle_steps = configs['cle_steps']
      progbar = keras.utils.Progbar(cle_steps)
      logger.info('Start CrossLayerEqualization...')
      for i in range(cle_steps):
        progbar.update(i + 1)
        tmp_model = transformed_model
        transformed_model, layer_metadata = model_transformer.ModelTransformer(
            tmp_model, cle_transforms, candidate_layers,
            layer_metadata).recursive_transform()
      logger.info('CrossLayerEqualization Done.')

    if logger.debug_enabled():
      model_utils.save_model(transformed_model, 'optimized_model.h5',
                             './debug/')
    return transformed_model, layer_metadata


class Vitis8BitFSQuantizeTransformsPipeline(TransformsPipeline):
  """Vitis 8Bit float scale model quantize transformations pipeline."""

  def apply(self, model, candidate_layers, layer_metadata, quantize_registry,
            mode):
    """Implement vitis 8-bit quantize transforms.

    Args:
      model: Keras model to be quantized.
      quantize_registry: QuantizeRegistry object containing the quantize configs
        for each layer.
      mode: String object indicating the mode of the quantized model.

    Returns:
      (Quantized Keras model.)
    """
    configs = self.get_configs()

    available_pre_annotate_transforms = collections.OrderedDict({})

    pre_annotated_model, layer_metadata = _apply_availables(
        model, configs, available_pre_annotate_transforms, candidate_layers,
        layer_metadata)

    if logger.debug_enabled():
      model_utils.save_model(pre_annotated_model, 'pre_annotated_model.h5',
                             './debug/')

    available_annotate_transforms = collections.OrderedDict({
        'conv_bn_activation_annotate':
            vitis_8bit_fs_quantize_transforms.ConvBNActivationAnnotate(),
        'conv_activation_annotate':
            vitis_8bit_fs_quantize_transforms.ConvActivationAnnotate(),
        'add_activation_annotate':
            vitis_8bit_fs_quantize_transforms.AddActivationAnnotate(),
    })
    annotated_model, layer_metadata = _apply_availables(
        pre_annotated_model, configs, available_annotate_transforms,
        candidate_layers, layer_metadata)

    if logger.debug_enabled():
      model_utils.save_model(annotated_model, 'annotated_model.h5', './debug/')

    freeze_bn_delay = configs['freeze_bn_delay']
    if freeze_bn_delay < 0:
      freeze_bn_delay = None
    quantize_transforms = [
        vitis_8bit_fs_quantize_transforms.InputLayerQuantize(
            quantize_registry, mode),
        vitis_8bit_fs_quantize_transforms.ConvBNQuantize(
            quantize_registry, mode, freeze_bn_delay),
        vitis_8bit_fs_quantize_transforms.LayersQuantize(
            quantize_registry, mode),
        vitis_8bit_fs_quantize_transforms.LayersInputQuantize(
            quantize_registry, mode),
    ]
    quantized_model, layer_metadata = model_transformer.ModelTransformer(
        annotated_model, quantize_transforms, candidate_layers,
        layer_metadata).recursive_transform()
    return quantized_model, layer_metadata
