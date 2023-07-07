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
"""Vitis float scale transformations pipeline for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_equalization_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_fast_finetune
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_bias_correction
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fs import vitis_fs_quantize_transforms
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


class VitisFSOptimizeTransformsPipeline(TransformsPipeline):
  """Vitis float scale pre-quantization optimize model transformations."""

  def apply(self, model, candidate_layers, layer_metadata):
    """Implement vitis 8-bit optimize transforms to make it more quantize-friendly.

    All the transforms should not break the float model structure, and
    the output of the transformed model should be consistant with the float 
    model.
    """
    configs = self.get_configs()

    available_transforms = collections.OrderedDict({
        'remove_dropout':
            vitis_optimize_transforms.RemoveDropout(),
        'separate_conv_act':
            vitis_optimize_transforms.SeparateConvAct(),
        'convert_relu6_to_relu':
            vitis_optimize_transforms.ConvertReLU6ToReLU(),
        'convert_tf_op_to_keras':
            vitis_optimize_transforms.ConvertTFOpToKeras(),
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
          'fold_conv_bn': vitis_optimize_transforms.ConvBNFold(),
          'convert_bn_to_dwconv': vitis_optimize_transforms.ConvertBNToDWConv(),
      }
      transformed_model, layer_metadata = _apply_availables(
          model, configs, available_transforms, candidate_layers,
          layer_metadata)

    # Cross Layer Equalization
    if configs['include_cle']:
      cle_to_relu6 = configs['cle_to_relu6']
      cle_balance_method = configs['cle_balance_method']
      cle_weight_threshold = configs['cle_weight_threshold']
      cle_transforms = [
          vitis_equalization_transforms.ConvConvCLE(cle_to_relu6,
                                                    cle_balance_method,
                                                    cle_weight_threshold),
          vitis_equalization_transforms.ConvActConvCLE(cle_to_relu6,
                                                       cle_balance_method,
                                                       cle_weight_threshold),
          vitis_equalization_transforms.ConvReLUConvCLE(cle_to_relu6,
                                                        cle_balance_method,
                                                        cle_weight_threshold),
          vitis_equalization_transforms.ConvReLUPadConvCLE(
              cle_to_relu6, cle_balance_method, cle_weight_threshold)
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


class VitisFSQuantizeTransformsPipeline(TransformsPipeline):
  """Vitis float scale model quantize transformations pipeline."""

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
        'no_quant_in_conv_bn_act':
            vitis_fs_quantize_transforms.NoQuantInConvBNAct(),
        'no_quant_in_conv_act':
            vitis_fs_quantize_transforms.NoQuantInConvAct(
                pre_annotated_model, quantize_registry),
        'no_quant_in_add_act':
            vitis_fs_quantize_transforms.NoQuantInAddAct(
                pre_annotated_model, quantize_registry),
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
        vitis_fs_quantize_transforms.InputLayerQuantize(quantize_registry,
                                                        mode),
        vitis_fs_quantize_transforms.ConvBNQuantize(quantize_registry, mode,
                                                    freeze_bn_delay),
        vitis_fs_quantize_transforms.LayersQuantize(annotated_model,
                                                    quantize_registry, mode),
    ]
    quantized_model, layer_metadata = model_transformer.ModelTransformer(
        annotated_model, quantize_transforms, candidate_layers,
        layer_metadata).recursive_transform()

    input_quantize_transforms = [
        vitis_fs_quantize_transforms.LayersInputQuantize(
            quantized_model, quantize_registry, mode)
    ]
    quantized_model, layer_metadata = model_transformer.ModelTransformer(
        quantized_model, input_quantize_transforms, candidate_layers,
        layer_metadata).recursive_transform()
    return quantized_model, layer_metadata


class VitisFSRefineTransformsPipeline(TransformsPipeline):
  """Vitis float scale model quantize transformations pipeline."""

  def apply(self, quantized_model, candidate_layers, layer_metadata,
            optimized_model, dataset, batch_size, steps, add_shape_info,
            input_shape):
    """Implement vitis float scale refine transforms.

    Args:
      qunantized_model: the quantized model to be refined.
      optimized_model: the optimized float model used in fast finetune to generate fake label.
      dataset: the dataset used in fast finetune.
      batch_size: the batch size of dataset used in fast finetune.
      steps: the steps of dataste used in fast finetune.
      add_shape_info: bool, whether to add shape information to the refined model. Must be set True
        for models with custom layers.
      input_shape: the shape of the model inputs, if not set, the default shape in the model inputs
        will be used.

    Returns:
      (Refined quantized model.)
    """
    refined_model = quantized_model
    configs = self.get_configs()

    # Fast finetune
    include_fast_ft = configs['include_fast_ft']
    fast_ft_epochs = configs['fast_ft_epochs']
    if include_fast_ft:
      logger.info("Start Fast Finetuning...")
      vitis_fast_finetune.fast_finetune(refined_model, optimized_model, dataset,
                                        batch_size, steps, fast_ft_epochs)
      logger.info("Fast Finetuning Done.")

    #  # Bias correction
    #  include_bias_corr = configs['quantize_pipeline_config'][
    #      'include_bias_corr']
    #  if include_bias_corr:
    #    logger.info("Start Bias Correction...")
    #    vitis_bias_correction.bias_correction(self._qcbev_model,
    #                                          self._optimized_model,
    #                                          calib_dataset, calib_batch_size,
    #                                          calib_steps)
    #    logger.info("Bias Correction Done.")

    if add_shape_info:
      logger.info("Start Getting Shape Information...")
      shape_info = model_utils.get_shape(
          model=refined_model, calib_dataset=dataset, input_shape=input_shape)
      if logger.debug_enabled():
        model_utils.save_shape_info(shape_info, './debug/')
        model_utils.save_model(refined_model, 'calibrated_model_add_shape.h5',
                               './debug/')
      logger.info("Getting Shape Information Done.")

    return refined_model, layer_metadata


class VitisFSFinalizeTransformsPipeline(TransformsPipeline):
  """Vitis float scale model finalize transformations pipeline."""

  def apply(self, refined_model, candidate_layers, layer_metadata):
    """Implement vitis float scale refine transforms.

    Args:
      refined_model: the refined model to be finalized.

    Returns:
      (Finalized quantized model.)
    """
    finalized_model = refined_model
    configs = self.get_configs()

    # Convert model format
    if configs['output_format'] == '':
      logger.debug(
          "No `output_format` found, skip model format conversion and output.")
      return finalized_model, layer_metadata

    formats = {'h5': '.h5', 'tf': '', 'onnx': '.onnx'}
    if configs['output_format'] not in formats:
      logger.error(
          "Invalid output_format: {}, supported output_format are: {}".format(
              configs['output_format'], list(formats.keys())))

    finalized_model_name = 'quantized_model'
    if configs['output_format'] == 'onnx':
      onnx_opset_version = configs['onnx_opset_version']
      model_utils.convert_to_onnx(finalized_model, configs['output_dir'],
                                  finalized_model_name, onnx_opset_version)
    else:
      filepath = os.path.join(
          configs['output_dir'],
          'quantized_model' + formats[configs['output_format']])
      finalized_model.save(filepath, save_format=configs['output_format'])

    return finalized_model, layer_metadata
