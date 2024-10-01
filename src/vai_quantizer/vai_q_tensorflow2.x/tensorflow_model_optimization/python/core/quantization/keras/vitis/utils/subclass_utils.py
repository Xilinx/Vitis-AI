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
"""Subclass Utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import copy
import collections
import pprint

from tensorflow_model_optimization.python.core.keras import compat as tf_compat
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_custom_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import subclass_replacement
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import replacements
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_refine_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.tqt import vitis_tqt_refine_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger
keras = tf.keras


def remove_subclass_dropout(model):
  """Remove dropout layer by replacing it with a identity layer
     for subclassed model in a in-place manner
  """

  #model_utils.show_sub_layers_tree(model, caption_str='remove dropout before')
  subclass_replacement.SublayerWrapper(quantize_registry=None).postprocess(
                                       model, remove_dropout=True)
  #model_utils.show_sub_layers_tree(model, caption_str='remove dropout after')

  return model


def convert_strategy_tqt_to_pof2s(layer, use_fixneuron_quant=False):
  """Helper function to convert strategy from TQTQuantizer to Pof2SQuantizer.
  The layer should be a quantized layer (not a keras layer).
  """
  layer_config = layer.get_config()

  new_layer = layer

  if isinstance(layer, vitis_quantize_layer.VitisQuantize):
    quantizer = layer_config['quantizer']
    if quantizer['class_name'] == 'Vitis>TQTQuantizer':
      # VitisQuantize's fixneuron quantize mode is always for activation
      use_fixneuron_quant_mode = 1 if use_fixneuron_quant else 0
      tqt_quantizer = deserialize_keras_object(quantizer)
      pof2s_quantizer = tqt_quantizer.convert_to_pof2s_quantizer(
              use_fixneuron_quant_mode)
      quantizer.update(serialize_keras_object(pof2s_quantizer))

      new_layer = vitis_quantize_layer.VitisQuantize.from_config(layer_config)

  elif isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
    quantize_config = layer_config['quantize_config']
    if not quantize_config['class_name'] == 'Vitis>NoQuantizeConfig':
      config = quantize_config['config']

      for k, v in config.items():
        # generate fixneuron quantize mode, 1 for activation, 2 for weights
        use_fixneuron_quant_mode = 0
        if k == 'activation_quantizers' or k == 'output_quantizers':
          if use_fixneuron_quant : use_fixneuron_quant_mode = 1
        elif k == 'weight_quantizers' or k == 'bias_quantizers':
          if use_fixneuron_quant : use_fixneuron_quant_mode = 2
        else:
          continue

        # skip blank quantizer
        if v == []:
          continue

        quantizers = config[k]
        for quantizer in quantizers:
          if quantizer['quantizer_type'] != 'TQTQuantizer':
            continue
          tqt_quantizer = vitis_quantize_configs._make_quantizer(
                                          quantizer['quantizer_type'],
                                          quantizer['quantizer_params'])
          pof2s_quantizer = tqt_quantizer.convert_to_pof2s_quantizer(
                  use_fixneuron_quant_mode)
          quantizer.update({
              'quantizer_type': 'Pof2SQuantizer',
              'quantizer_params': pof2s_quantizer.get_config()
          })

      new_layer = vitis_quantize_wrapper.QuantizeWrapper.from_config(layer_config)

  if new_layer is layer:
    return None

  # Build the new layer
  if layer.built and not (layer._build_input_shape is None):
    new_layer.build(layer._build_input_shape)

  # Copy weights
  weight_value_dict = {}
  for weight_tensor, weight_numpy in zip(layer.weights, layer.get_weights()):
    k = weight_tensor.name
    v = weight_numpy

    # Note that the variable "log_th" should be converted to "pos"
    if k.endswith('log_th:0'):
      k = k.replace('log_th:0', 'pos:0')
      v = 7 - np.ceil(v)

    weight_value_dict[k] = v

  weights = []

  for weight_tensor, weight_numpy in zip(new_layer.weights, new_layer.get_weights()):
    for k, v in weight_value_dict.items():
      if k.split('/')[-1] == weight_tensor.name.split('/')[-1]:
        weights.append(v)
        break

  if weights == []:
    logger.warning("tqt_to_pof2s gets empty weights during set layer weights, " \
                   "layer name: {}, built: {}".format(layer.name, layer.built))
  else:
    new_layer.set_weights(weights)

  return new_layer

def convert_strategy_pof2s_to_fs(layer, use_framework_quant=True):
  """Helper function to convert strategy from Pof2SQuantizer to FSQuantizer.
  The layer should be a quantized layer (not a keras layer).
  """
  layer_config = layer.get_config()

  new_layer = layer

  if isinstance(layer, vitis_quantize_layer.VitisQuantize):
    quantizer = layer_config['quantizer']
    if quantizer['class_name'] == 'Vitis>Pof2SQuantizer':
      pof2s_quantizer = deserialize_keras_object(quantizer)
      fs_quantizer = pof2s_quantizer.convert_to_fs_quantizer(use_framework_quant)
      quantizer.update(serialize_keras_object(fs_quantizer))

      new_layer = vitis_quantize_layer.VitisQuantize.from_config(layer_config)

  elif isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
    quantize_config = layer_config['quantize_config']
    if not quantize_config['class_name'] == 'Vitis>NoQuantizeConfig':
      config = quantize_config['config']
      quantizers = config['weight_quantizers'] + config[
          'bias_quantizers'] + config['activation_quantizers'] + config[
              'output_quantizers']
      for quantizer in quantizers:
        if quantizer['quantizer_type'] == 'Pof2SQuantizer':
          pof2s_quantizer = vitis_quantize_configs._make_quantizer(
                                          quantizer['quantizer_type'],
                                          quantizer['quantizer_params'])
          fs_quantizer = pof2s_quantizer.convert_to_fs_quantizer(use_framework_quant)
          quantizer.update({
              'quantizer_type': 'FSQuantizer',
              'quantizer_params': fs_quantizer.get_config()
          })

      new_layer = vitis_quantize_wrapper.QuantizeWrapper.from_config(layer_config)

  if new_layer is layer:
    return None

  # Build the new layer
  if layer.built and not (layer._build_input_shape is None):
    new_layer.build(layer._build_input_shape)

  # Copy weights
  weight_value_dict = {}

  for weight_tensor, weight_numpy in zip(layer.weights, layer.get_weights()):
    weight_value_dict[weight_tensor.name] = weight_numpy

  weight_names = list(weight_value_dict.keys())
  for name in weight_names:
    # Note that map the pof2s parameters to fs
    if name.endswith('_pos:0'):
      min_key = name[:-6] + '_min:0'
      max_key = name[:-6] + '_max:0'
      weight_value_dict[min_key] = -128. / np.power(2, weight_value_dict[name])
      weight_value_dict[max_key] = 127. / np.power(2, weight_value_dict[name])
      weight_value_dict.pop(name)  # fs strategy does not need this variable

  weights = []

  for weight_tensor, weight_numpy in zip(new_layer.weights, new_layer.get_weights()):
    have_copied = False

    for k, v in weight_value_dict.items():
      if k.split('/')[-1] == weight_tensor.name.split('/')[-1]:
        weights.append(v)
        have_copied = True
        break

    if have_copied == False:
      weights.append(weight_numpy)  # fs strategy has calib_hist:0 and calib_bin_edges:0

  if weights == []:
    logger.warning("pof2s_to_fs gets empty weights during set layer weights, " \
                   "layer name: {}, built: {}".format(layer.name, layer.built))
  else:
    new_layer.set_weights(weights)

  return new_layer

class SubclassConverter(replacements.Replacement):
  """Do some conversions for subclassed model.
  Not the same as functional model supports graph transform, for a subclass model,
  any modification on layers can be achieved by replacement only.
  """

  def __init__(self, use_fixneuron_quant=0, use_framework_quant=True):
    """These parameters are used for this converter."""
    self.use_fixneuron_quant = use_fixneuron_quant
    self.use_framework_quant = use_framework_quant
    self.allowed_conversions = ['tqt_to_pof2s', 'pof2s_to_fs']

  def _conversion_tqt_to_pof2s(self, layer, parent=None):
    """Conversion of tqt quantize stragety to pof2s quantize strategy."""
    if parent is not None:
      pass

    if not model_utils.is_quantize_layer(layer):
      return None

    return convert_strategy_tqt_to_pof2s(layer,
            use_fixneuron_quant=self.params)

  def _conversion_pof2s_to_fs(self, layer, parent=None):
    """Conversion of pof2s quantize stragety to fs quantize strategy."""
    if parent is not None:
      pass

    if not model_utils.is_quantize_layer(layer):
      return None

    return convert_strategy_pof2s_to_fs(layer,
            use_framework_quant=self.params)

  def work(self, model, inputs, conversion):
    """Subclass replacer gets to work."""
    if not model_utils.is_subclass_model(model):
      logger.error('Invalid model {}, allowed a subclassed model only.'.format(
                    model.name))
    if not conversion in self.allowed_conversions:
      logger.error('Invalid conversion {}, allowed conversions are: {}.'.format(
                    conversion, self.allowed_conversions))

    if conversion=='tqt_to_pof2s':
      self.worker = self._conversion_tqt_to_pof2s
      self.params = self.use_fixneuron_quant
    elif conversion=='pof2s_to_fs':
      self.worker = self._conversion_pof2s_to_fs
      self.params = self.use_framework_quant
    else:
      logger.error('Not implemented conversion {}'.format(conversion))

    self._traverse_sub_layers(model)
    model_utils.show_sub_layers_tree(model, caption_str='convert_strategy')

    if inputs is not None:
      # Predict once because we replaced sublayers.
      model.predict(inputs, batch_size=1, steps=1)
      # Compile the model to clear model.predict_function cache.
      optimizer = 'adam' if (model.optimizer is None) else model.optimizer
      model.compile(optimizer=optimizer, loss=None)

    return model


def convert_quantize_strategy(model, conversion='tqt_to_pof2s',
        use_fixneuron_quant=0, use_framework_quant=True):
  """Convert quantize strategy for subclassed model"""

  converter = SubclassConverter(use_fixneuron_quant=use_fixneuron_quant,
          use_framework_quant=use_framework_quant)

  return converter.work(model, None, conversion)


def extract_spec_from_dataset(dataset):
  """Extract spec from dataset."""
  if dataset is None:
    logger.warning("No dataset provide.")
    return None

  def _data_shape_and_dtype(data):
    if isinstance(data, np.ndarray):
      return data.shape[1:], data.dtype
    elif isinstance(data, tf.Tensor):
      return tuple(data.shape)[1:], data.dtype
    else:
      return None, None

  dataspec = []

  if isinstance(dataset, dict):
    for name, data in dataset.items():
      shape, dtype = _data_shape_and_dtype(data)
      if shape is None or dtype is None:
        logger.warning("Extracting data spec from dataset for subclass failed.")
        return []
      else:
        dataspec.append({'shape': shape, 'dtype' : dtype, 'name' : name})

  #elif isinstance(dataset, (list, tuple)):
  #  for data in dataset:
  #    shape, dtype = _data_shape_and_dtype(data)
  #    if shape is None or dtype is None:
  #      logger.warning("Extracting data spec from dataset for subclass failed.")
  #      return []
  #    else:
  #      dataspec.append({'shape': shape, 'dtype' : dtype})

  else:
    logger.warning("Extracting data spec from dataset for subclass failed.")
    return []

  return dataspec


def save_subclass_model(model, configs, dataset=None):
  """Save quantized subclass model to specified formats"""

  formats = {'tf': '', 'pb': '.pb', 'onnx': '.onnx'}
  if configs['output_format'] not in formats:
    logger.error(
      "Invalid output_format: {}, supported output_format are: {}".format(
      configs['output_format'], list(formats.keys())))

  model_name = 'quantized_model'
  model_path = os.path.join(configs['output_dir'],
      model_name + formats[configs['output_format']])

  if configs['output_format'] == 'onnx':
    onnx_opset_version = configs['onnx_opset_version']

    model_utils.convert_to_onnx(model, configs['output_dir'],
                                model_name, onnx_opset_version)
  elif configs['output_format'] == 'pb':
    from .pb_utils import graph_from_keras, graph_from_keras_subclass

    if dataset is None:
      # This may be float subclass model
      graph_def = graph_from_keras(model,
        model.input_names, model.output_names, constfold=True)
    else:
      dataspec = extract_spec_from_dataset(dataset)
      if len(dataspec) <= 0:
        # This may be quantized subclass model with single inputs
        graph_def = graph_from_keras(model,
          model.input_names, model.output_names, constfold=False)
      else:
        # This may be quantized subclass model with multiple inputs
        graph_def = graph_from_keras_subclass(model, dataspec, constfold=False)

    # Convert 'TRAIN' phase to 'EVAL' phase for FixNeuron
    for node in graph_def.node:
      if node.op == "FixNeuron":
        if node.attr["phase"].i == 2:
          node.attr["phase"].i = 1

    with tf.io.gfile.GFile(model_path, mode='wb') as f:
      f.write(graph_def.SerializeToString())
  else:
    model.save(model_path, save_format=configs['output_format'])
