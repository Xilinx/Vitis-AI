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
"""Quantization API functions for tf.keras models."""

from __future__ import absolute_import

import os
import copy
import collections
import sys

import tensorflow as tf
import numpy as np

try:
  import target_factory
except:
  target_factory = None

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_config as quantize_config_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_custom_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_ops
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_pooling
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy import vitis_quantize_strategy_factory
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fs import vitis_fs_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fsx import vitis_fsx_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import vai_utf_parser
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import subclass_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common.entropy_percentile import calibrator_numpy

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import subclass_replacement

logger = common_utils.VAILogger
keras = tf.keras


def create_optimize_model(model, candidate_layers, layer_metadata,
                          quantize_strategy):
  """Optimize a `tf.keras` model before quantization.

  Args:
    model: the float model to be optimized.

  Returns:
    (Optimized float model.)
  """
  if model is None:
    logger.error('`model` cannot be None')

  if not isinstance(model, keras.Model):
    logger.error('[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
                 '`model` can only be a `tf.keras.Model` instance.'
                 'You passed an instance of type: {input}.'.format(
                     input=model.__class__.__name__))

  configs = quantize_strategy.get_optimize_pipeline().get_configs()

  optimized_model = model

  return optimized_model, layer_metadata


def create_refine_model(quantized_model, candidate_layers, layer_metadata,
                        quantize_strategy, optimized_model, dataset, batch_size,
                        steps, add_shape_info, input_shape):
  """Refine a quantize calibrated model

  Will do post-quantize adjustments and perform some finetuning algorithms.

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
  if quantized_model is None:
    logger.error('`quantized_model` cannot be None')

  if not isinstance(quantized_model, keras.Model):
    logger.error('[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
                 '`quantized_model` can only be a `tf.keras.Model` instance. '
                 'You passed an instance of type: {input}.'.format(
                     input=quantized_model.__class__.__name__))

  configs = quantize_strategy.get_refine_pipeline().get_configs()

  # Clear predict function cache (generated by calibration) for the quantized model,
  # otherwise there is accuracy differences when users evaluate it without compilation.
  quantized_model.predict_function = None

  refined_model = quantized_model

  if 'convert_to_fs_quantize_strategy' in configs and \
    configs['convert_to_fs_quantize_strategy']:
    converter = subclass_utils.SubclassConverter()
    refined_model = converter.work(refined_model, inputs=dataset,
                                   conversion='pof2s_to_fs')

  return refined_model, layer_metadata


def create_finalize_model(refined_model, candidate_layers, layer_metadata,
                          quantize_strategy, dataset=None):
  """Finalize a quantize refined model.

  Will do model format conversions.

    Args:
      refined_model: the refined model to be finalized.

    Returns:
      (finalized quantized model.)
    """
  if refined_model is None:
    logger.error('`refined_model` cannot be None')

  if not isinstance(refined_model, keras.Model):
    logger.error('`refined_model` can only be a `tf.keras.Model` instance. '
                 'You passed an instance of type: {input}.'.format(
                     input=refined_model.__class__.__name__))

  configs = quantize_strategy.get_finalize_pipeline().get_configs()

  finalized_model = refined_model

  if configs['output_format'] != '':
    subclass_utils.save_subclass_model(finalized_model, configs,
            dataset=dataset)
  
  return finalized_model, layer_metadata


def create_quantize_model(model, candidate_layers, layer_metadata,
                          quantize_strategy, mode, target,
                          dataset=None, batch_size=1, steps=1, specific_layers=None):
  """Quantize a `tf.keras` model with the default quantization implementation.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    model: tf.keras model to be quantized. It can have pre-trained
      weights.
    quantize_strategy: QuantizeStrategy constaining the configurations.

  Returns:
    Returns a new `tf.keras` model prepared for quantization.
  """
  if model is None:
    logger.error('`model` cannot be None')

  if not model.built:
    logger.error('`model` must be a built model. '
                 'been built yet. Please call `model.build(input_shape)` '
                 'before quantizing your model.')

  AVAILABLE_MODES = ['QCB', 'QAT', 'ANALYSE', 'QCBEV']
  if mode not in AVAILABLE_MODES:
    logger.error('Mode `{}` is not valid, available modes are:{}.'.format(
        mode, AVAILABLE_MODES))

  remove_dropout = False if mode == 'QAT' else True

  configs = quantize_strategy.get_quantize_pipeline().get_configs()

  # Only some build-in models are supported, such as huggingface BERT
  replacer = subclass_replacement.SubclassReplacer()
  replaced_model = replacer.work(model, dataset)

  # Quantize sublayers of subclass which is named with "Quant" prefix
  quantizer = subclass_replacement.SublayerWrapper(
      quantize_strategy.get_quantize_registry(), mode)
  quantized_model = quantizer.work(replaced_model, dataset,
          remove_dropout=remove_dropout)

  return quantized_model, layer_metadata
