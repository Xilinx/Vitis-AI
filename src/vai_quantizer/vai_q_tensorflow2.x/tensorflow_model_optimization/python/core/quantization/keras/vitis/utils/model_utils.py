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
"""Model Utilities."""

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
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_custom_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_refine_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.tqt import vitis_tqt_refine_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger
keras = tf.keras

if model_transformer._is_tf_version_above(2, 11):
  Optimizer = keras.optimizers.legacy.Optimizer
else:
  Optimizer = keras.optimizers.Optimizer


def sublayers(subclass):
  """Fetch sublayers of subclass model or subclass layer"""
  # If we use subclass.submodules, it will raise a TypeError
  # when the model had been compiled. Refer to this issue:
  # https://github.com/keras-team/keras/issues/15183
  return list(subclass._flatten_layers(include_self=False))


def is_subclass_layer(layer):
  """Check if Subclass Layer."""
  if not isinstance(layer, keras.layers.Layer):
    return False

  if isinstance(layer, keras.layers.Wrapper):
    # Exclude wrappers
    return False
  elif (not isinstance(layer, keras.Model)) and len(sublayers(layer)):
    # Should exclude submodels
    return True

  return False


def is_subclass_model(model):
  """Check if Subclass Model."""
  if not isinstance(model, keras.Model):
    return False

  if not isinstance(model, keras.Sequential) \
     and not model._is_graph_network:  # pylint: disable=protected-access
    # Keras official judging method
    return True
  '''
  else:
    # When the model contains subclass layer or nested subclass model,
    # it will be classified into a subclass model
    layers = sublayers(model)
    for layer in layers:
      if is_subclass_layer(layer) or is_subclass_model(layer):
        return True
  '''

  return False


def is_tf_graphdef(model):
  """Check if tensorflow graph def"""
  if isinstance(model, tf.compat.v1.GraphDef):
    return True

  return False


def have_nested_submodel(model):
  """Check having nested submodel or not"""
  if not isinstance(model, keras.Model):
    return False

  for layer in model.layers:
    if isinstance(layer, keras.Model):
      return True

  return False


def create_input_tensor(shape):
  """create the input_shape tensor with tensor or inputlayer format"""
  if shape.count(None) or shape.count(-1):
    return keras.layers.Input(shape=shape[1:])
  else:
    return tf.convert_to_tensor(np.random.random(shape))


def remove_layer(model, class_name, name='.*'):
  """Remove given layer from the model."""
  transforms = [vitis_optimize_transforms.RemoveLayer(class_name, name)]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def convert_quantize_strategy(model, conversion='tqt_to_pof2s',
        use_fixneuron_quant=0, use_framework_quant=True):
  allowed_conversions = ['tqt_to_pof2s', 'pof2s_to_fs']
  if not conversion in allowed_conversions:
    logger.error('Invalid conversion {}, allowed conversions are: {}.'.format(
        conversion, allowed_conversions))

  if conversion == 'tqt_to_pof2s':
    transforms = [
        vitis_tqt_refine_transforms.ConvertTQTToPof2SQuantizeStrategy(
            use_fixneuron_quant=use_fixneuron_quant)
    ]
    transformed_model, _ = model_transformer.ModelTransformer(
        model, transforms, None, None).recursive_transform()
    return transformed_model
  elif conversion == 'pof2s_to_fs':
    transforms = [
        vitis_pof2s_refine_transforms.ConvertPof2SToFSQuantizeStrategy(
            use_framework_quant=use_framework_quant)
    ]
    transformed_model, _ = model_transformer.ModelTransformer(
        model, transforms, None, None).recursive_transform()
    return transformed_model

  return model


def insert_fix_neuron(model):
  transforms = [
    vitis_pof2s_refine_transforms.ConvertPof2SToPof2SQuantizeStrategyWithFixNeuron()]
  transformed_model, _ = model_transformer.ModelTransformer(
        model, transforms, None, None).recursive_transform()
  return transformed_model


def separate_conv_act(model):
  transforms = [vitis_optimize_transforms.SeparateConvAct()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def separable_conv(model):
  transforms = [vitis_optimize_transforms.SeparableConv()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def fake_conv_bn_fold(model):
  transforms = [vitis_optimize_transforms.FakeConvBNFold()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def real_conv_bn_fold(model):
  transforms = [vitis_optimize_transforms.RealConvBNFold()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def conv_bn_quantize_fold(model):
  transforms = [vitis_pof2s_quantize_transforms.ConvBNQuantizeFold()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def replace_hard_sigmoid(model):
  transforms = [vitis_pof2s_quantize_transforms.ReplaceHardSigmoid()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def get_quantize_info(model):
  """Get the quantize info of the model"""
  layers = sublayers(model) if is_subclass_model(model) else model.layers

  quantize_info = collections.OrderedDict()
  for layer in layers:
    if is_subclass_model(model) and not layer.built:
      continue  # Skip those layers defined in init() but not used in call()

    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      quantize_info[layer.layer.name] = layer.get_quantize_info()
    elif isinstance(layer, vitis_quantize_layer.VitisQuantize):
      quantize_info[layer.name] = layer.get_quantize_info()
    elif isinstance(layer, vitis_custom_wrapper.CustomOpWrapper) and isinstance(
        layer.layer, vitis_quantize_wrapper.QuantizeWrapper):
      quantize_info[layer.layer.layer.name] = layer.layer.get_quantize_info()
  return quantize_info


def set_quantize_info(model, new_quantize_info):
  """Set the quantize info of the model"""
  layers = sublayers(model) if is_subclass_model(model) else model.layers

  for layer in layers:
    if is_subclass_model(model) and not layer.built:
      continue  # Skip those layers defined in init() but not used in call()

    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper
                 ) and layer.layer.name in new_quantize_info:
      layer.set_quantize_info(new_quantize_info[layer.layer.name])
    elif isinstance(
        layer,
        vitis_quantize_layer.VitisQuantize) and layer.name in new_quantize_info:
      layer.set_quantize_info(new_quantize_info[layer.name])
  return


def save_quantize_info(quantize_info, output_dir='./'):
  """Save the quantize info to the disk."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  filename = os.path.join(output_dir, 'quantize_info.txt')
  with open(filename, 'w') as f:
    idx = 0
    for k, v in quantize_info.items():
      f.write(str(idx) + ' ' + k + '\n')
      formatted = pprint.pformat(v)
      for line in formatted.splitlines():
        f.write('  ' + line + '\n')
      idx += 1
  logger.debug(filename + ' saved.')
  return


def save_shape_info(shape_info, output_dir='./'):
  """Save the shape info to the disk."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  filename = os.path.join(output_dir, 'shape_info.txt')
  with open(filename, 'w') as f:
    for k, v in shape_info.items():
      f.write("{} : {}\n".format(k, v))
  logger.debug(filename + ' saved.')
  return


class SSOptimizer(Optimizer):
  # shape saving optimizer
  def __init__(self, shape_info={}, name="SSOptimizer", **kwargs):
    super().__init__(name, **kwargs)
    self._set_hyper("shape_info", shape_info)
    # for k, v in shape_info.items():
    #   self._set_hyper(k, v))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    return None

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    return None

  def get_config(self):
    config = super(SSOptimizer, self).get_config()
    config.update({
        "shape_info": self._serialize_hyperparameter("shape_info"),
    })
    return config


def get_layer(model, layer_name):
  """ get the layer from mode by name in a recursive manner"""

  def _get_layer_recursively(model, layer_name):
    target_layer = None

    for layer in model.layers:
      if layer.name == layer_name:
        target_layer = layer
        break
      elif isinstance(layer, keras.Model):
        submodel = layer
        target_layer = _get_layer_recursively(submodel, layer_name)
        if target_layer is not None:
          break

    return target_layer

  layer = None

  if layer_name is not None:
    layer = _get_layer_recursively(model, layer_name)

  if layer is None:
    logger.error("No such layer: {}".format(layer_name))
    return model.get_layer(layer_name)

  return layer


def get_shape(model, calib_dataset=None, input_shape=None):
  """ get the shape of the layer output tensor and save it into layer.weight"""
  logger.info("Getting model layer shape information")
  if not hasattr(model, "shape_info"):
    input_data = None
    if input_shape is not None:
      if input_shape.count(-1) == 0 and input_shape.count(None) == 0:
        input_data = np.random.random(input_shape)
      else:
        if calib_dataset is not None:
          input_data = calib_dataset
        else:
          logger.error(
              'Please assign calib_dataset when input_shape has None or -1, input_shape is:({}) .'
              .format(input_shape))
    elif calib_dataset is not None:
      input_data = calib_dataset
    else:
      logger.error(
          'Please assign calib_dataset or input_shape to do shape inference.')

    shape_info = {}
    for layer in model.layers:
      output_shape = None
      try:
        output_shape = layer.output_shape
      except AttributeError:
        pass
      except RuntimeError:  # output_shape unknown in Eager mode.
        pass
      if output_shape:
        if isinstance(output_shape, list):
          output_shape = list(output_shape[0])
        elif isinstance(output_shape, tuple):
          output_shape = list(output_shape)
        else:
          logger.warning('unknown output shape')
          output_shape = output_shape
        output_shape[0] = 1
        shape_info[layer.name] = np.array(output_shape)

    prog_bar = tf.keras.utils.Progbar(len(model.layers))
    outputs = []
    for layer in model.layers:
      if layer.name in shape_info:
        continue
      tmp_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
      layer_output_shape = np.array(
          tmp_model.predict(input_data, batch_size=1, steps=1).shape)
      layer_output_shape[0] = 1
      shape_info[layer.name] = layer_output_shape
      prog_bar.add(1)
    model.shape_info = shape_info

  # save shape info into optimizer weights
  if not model.optimizer or \
          not isinstance(model.optimizer, SSOptimizer):
    model.optimizer = SSOptimizer(shape_info=model.shape_info)

  org_weights = model.optimizer.weights
  for layer_name, shape_array in model.shape_info.items():
    has_add_weight = False
    for w in org_weights:
      if w.name == layer_name + ":0":
        has_add_weight = True
        break
    if not has_add_weight:
      w = model.optimizer.add_weight(
          name=layer_name,
          dtype=tf.int32,
          shape=np.shape(shape_array),
          trainable=False)
      model.optimizer._weights.append(w)

  dst_weights = []
  for w in model.optimizer.weights:
    name = w.name.split(":")[0]
    shape_info = model.shape_info[name]
    for i in range(len(shape_info)):
      if shape_info[i] == None:
        shape_info[i] = -1
    dst_weights.append(shape_info)
  model.optimizer.set_weights(dst_weights)
  return model.shape_info


def save_model(model, filename, output_dir='./'):
  """Save the model to the disk."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  filename = os.path.join(output_dir, filename)
  model.save(filename)
  logger.debug(filename + ' saved.')
  return


def is_layer_wrapper(layer):
  """Check if QuantizeWrapper."""
  return isinstance(layer,
                    vitis_quantize_wrapper.QuantizeWrapper)


def keras_to_graphdef(model):
  """Convert keras model to graph def"""
  from .pb_utils import graph_from_keras

  return graph_from_keras(model, model.input_names,
          model.output_names, constfold=True)


def save_func_model(model, configs):
  """Save quantized functional model to specified formats"""

  formats = {'h5': '.h5', 'tf': '', 'pb': '.pb', 'onnx': '.onnx'}
  if configs['output_format'] not in formats:
    logger.error(
      "Invalid output_format: {}, supported output_format are: {}".format(
      configs['output_format'], list(formats.keys())))

  model_name = 'quantized_model'
  model_path = os.path.join(configs['output_dir'],
      model_name + formats[configs['output_format']])

  if configs['output_format'] == 'onnx':
    onnx_opset_version = configs['onnx_opset_version']

    convert_to_onnx(model, configs['output_dir'],
                    model_name, onnx_opset_version)
  elif configs['output_format'] == 'pb':
    from .pb_utils import graph_from_keras

    graph_def = graph_from_keras(model,
      model.input_names, model.output_names, constfold=False)

    # Convert 'TRAIN' phase to 'EVAL' phase for FixNeuron
    for node in graph_def.node:
      if node.op == "FixNeuron":
        if node.attr["phase"].i == 2:
          node.attr["phase"].i = 1

    with tf.io.gfile.GFile(model_path, mode='wb') as f:
      f.write(graph_def.SerializeToString())
  else:
    model.save(model_path, save_format=configs['output_format'])


def is_quantize_layer(layer):
  """Check if QuantizeWrapper or VitisQuantize Layer."""
  return isinstance(layer,
                    vitis_quantize_wrapper.QuantizeWrapper) or isinstance(
                        layer, vitis_quantize_layer.VitisQuantize)


def set_layer_mode(model, mode, layer_names=None):
  """Set the mode of QuantizeWrapper and VitisQuantize Layer."""
  layers = sublayers(model) if is_subclass_model(model) else model.layers

  for layer in layers:
    if is_subclass_model(model) and not layer.built:
      continue  # Skip those layers defined in init() but not used in call()

    if is_quantize_layer(layer):
      if not layer_names or layer.name in layer_names:
        layer.mode = mode
      elif isinstance(layer, vitis_custom_wrapper.CustomOpWrapper):
        if is_quantize_layer(layer.layer):
          if not layer_names or layer.layer.name in layer_names:
            layer.layer.mode = mode
  return


def get_quantize_layers(model):
  """Get quantize layers within the model."""
  layers = sublayers(model) if is_subclass_model(model) else model.layers

  quantize_layers = {}

  for layer in layers:
    if is_subclass_model(model) and not layer.built:
      continue # Skip those layers defined in init() but not used in call()

    if is_quantize_layer(layer):
      quantize_layers[layer.name] = layer

  return quantize_layers


def clone_model_with_weights(model_to_clone):
  """Clone keras model with weights."""
  if is_subclass_model(model_to_clone):
    logger.warning('clone subclass model is not supported')
    return model_to_clone

  cloned_model = keras.models.clone_model(model_to_clone)
  cloned_model.set_weights(model_to_clone.get_weights())
  return cloned_model


def convert_to_onnx(tf_model, output_dir, quantized_model_name,
                    onnx_opset_version):
  """Convert tf.keras model to onnx model."""
  #saved_model_filepath = os.path.join(output_dir, quantized_model_name)
  #tf_model.save(saved_model_filepath, save_format='tf')

  import tf2onnx
  filepath = os.path.join(output_dir, quantized_model_name + '.onnx')
  model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
      tf_model,
      input_signature=None,
      opset=onnx_opset_version,
      custom_ops=None,
      custom_op_handlers=None,
      custom_rewriter=None,
      inputs_as_nchw=None,
      extra_opset=None,
      shape_override=None,
      target=None,
      large_model=False,
      output_path=filepath)


def dump_model_weights(model, dump_float, output_dir):
  """Dump model weights."""
  # Get weight quantize info
  w_q_map = {}
  for layer in model.layers:
    if isinstance(layer, vitis_custom_wrapper.CustomOpWrapper):
      for w in layer.weights:
        w_q_map[w.name.rstrip(":0")] = None
    if is_quantize_layer(layer):
      if isinstance(layer, vitis_quantize_layer.VitisQuantize):
        continue
      layer_quantize_info = layer.get_quantize_info()
      for name, value in layer_quantize_info.items():
        if value.get('type') == 'weight':
          w_name = "quant_" + name.rstrip(':0')
          #for scale exclude flost-scale
          if 'quant_pos_var' in value['info']:
            w_q_map[w_name] = 2**value['info']['quant_pos_var']
          #for float scale
          elif 'scale' in value['info']:
            w_q_map[w_name] = value['info']['scale']

  logger.info("Dumping weights/biases...")
  dump_folder = os.path.join(output_dir, "dump_results_weights")
  if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)

  index = 0
  for w in model.weights:
    w_name = w.name.rstrip(':0')
    if w_name not in w_q_map:
      continue

    index = index + 1
    filename = os.path.join(dump_folder, w_name.replace('/', '_'))
    logger.info("Dumping ({}/{}): {}".format(index, len(w_q_map), w_name))

    res = w.numpy()
    res_flatten = res.flatten()
    if dump_float:
      res_flatten.tofile(filename + '_float.bin')
      np.savetxt(filename + "_float.txt", res_flatten, fmt="%s", delimiter=",")

    if w_name in w_q_map and w_q_map[w_name] is not None:
      res = np.round(res * w_q_map[w_name])
      res = res.clip(-128, 127)
      res_flatten = res.flatten()
      res_flatten.astype(np.int8).tofile(filename + ".bin")
      np.savetxt(
          filename + ".txt",
          res_flatten.astype(np.int8),
          fmt="%s",
          delimiter=",")


def dump_model_activations(model, dataset, dump_float, output_dir):
  """Dump model activation."""
  # Get activation quantize info
  a_q_map = {}
  for layer in model.layers:
    if is_quantize_layer(layer):
      layer_quantize_info = layer.get_quantize_info()
      if isinstance(layer, vitis_quantize_layer.VitisQuantize):
        if 'quant_pos_var' in layer_quantize_info['info']:
          # for pof2s
          a_q_map[layer.name] = 2**layer_quantize_info['info']['quant_pos_var']
        elif 'scale' in layer_quantize_info['info']:
          #for float scale
          a_q_map[layer.name] = layer_quantize_info['info']['scale']
      else:
        for name, value in layer_quantize_info.items():
          if value.get('type') in [
              'output', 'pre_activation', 'post_activation'
          ]:
            if 'quant_pos_var' in value['info']:
              a_q_map[layer.name] = 2**value['info']['quant_pos_var']
            elif 'scale' in value['info']:
              a_q_map[layer.name] = value['info']['scale']

  # TODO: Support layers of nested models
  if dump_float:
    quant_layers = [
        layer for layer in model.layers if not isinstance(layer, keras.Model)
    ]
  else:
    quant_layers = [
        layer for layer in model.layers
        if is_quantize_layer(layer) and not isinstance(layer, keras.Model)
    ]

  model = keras.Model(
      inputs=model.inputs, outputs=[layer.output for layer in quant_layers])

  logger.info("Dumping activations...")
  # TODO: Support dump for multi-batches
  dump_folder = os.path.join(output_dir, "dump_results_0")
  if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)

  results = model.predict(dataset, steps=1)

  index = 0
  for layer, res in zip(quant_layers, results):
    index = index + 1
    a_name = layer.name
    filename = os.path.join(dump_folder, a_name.replace('/', '_'))
    logger.info("Dumping ({}/{}): {}".format(index, len(quant_layers), a_name))
    res = res.flatten()

    if dump_float:
      res.tofile(filename + '_float.bin')
      np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")

    if a_name in a_q_map:
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and any(
          act._should_pre_quantize()
          for act in layer._quantize_activations) and not dump_float:
        res.tofile(filename + '_float.bin')
        np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")
      else:
        res = res * a_q_map[a_name]
        res.astype(np.int8).tofile(filename + ".bin")
        np.savetxt(
            filename + ".txt", res.astype(np.int8), fmt="%s", delimiter=",")


def check_in_model(model, layer):
  layers = [l.name for l in model.layers]
  return layer in layers


def check_near_dropout(model, ignore_layers=[]):
  for layer in model.layers:
    if isinstance(layer, keras.layers.Dropout):
      layer_in_name = layer.inbound_nodes[0].inbound_layers.name
      if layer_in_name in ignore_layers:
        return True


def get_candidate_layers(model, in_layers=[], out_layers=[], ignore_layers=[]):
  """
  Get candidate layers from model according to input, output layers and ignore layers.

  parameters:
    in_layers: specify start layers
    out_layers: specify end layers
    ignore_layers: specify which layers will be ignored
  return:
    * if all parameters are empty then do nothing and return None
    * if input and output layers are empty, and ignore_layers are not in model then
      return None
    * the default value input output layers are model.inputs and model.outputs
  """
  logger.debug("Getting candidate layers...")
  if model is None:
    logger.error('`model` cannot be None')

  if not isinstance(model, keras.Model):
    logger.error('`model` can only be a `tf.keras.Model` instance. '
                 'You passed an instance of type: {input}.'.format(
                     input=model.__class__.__name__))

  if not isinstance(model, keras.Sequential) \
      and not model._is_graph_network:  # pylint: disable=protected-access
    logger.error('`model` can only either be a tf.keras Sequential or '
                 'Functional model.')

  def _remove_ignore_layers(candidate_layers, ignore_layers):
    is_remove = False
    if candidate_layers is None:
      return None, is_remove

    ret = set()
    for l in candidate_layers:
      if l in ignore_layers:
        is_remove = True
      else:
        ret.add(l)
    return ret, is_remove

  def _get_layer_input(layer):
    curr_in_layers = []
    keras_layer = model.get_layer(layer)
    for node in keras_layer.inbound_nodes:
      inbound_layers = node.inbound_layers
      if not isinstance(inbound_layers, list):
        curr_in_layers.append(inbound_layers.name)
      else:
        for inb in inbound_layers:
          curr_in_layers.append(inb.name)
    return curr_in_layers

  def _get_layers_from_in_out(model, in_layers, out_layers):
    no_visit_layer = set()
    candidate_layers = set()
    model_inputs = [l.name for l in model.inputs]
    for layer in out_layers:
      no_visit_layer.add(layer)
    while(no_visit_layer):
      layer = no_visit_layer.pop()
      candidate_layers.add(layer)
      curr_in_layers = _get_layer_input(layer) if layer not in (in_layers or model_inputs) else None
      for in_layer in curr_in_layers:
        if in_layer in (in_layers or model_inputs):
          candidate_layers.add(in_layer)
        if in_layer not in candidate_layers:
          no_visit_layer.add(in_layer)
    return candidate_layers

  if not (in_layers or out_layers or ignore_layers):
    return None

  if not (in_layers or out_layers):
    candidate_layers = [l.name for l in model.layers]
    candidate_layers, is_remove = _remove_ignore_layers(candidate_layers,
                                                        ignore_layers)
    if is_remove:
      return candidate_layers
    else:
      return None

  # Check if in and out layers in model
  filted_in_layers = []
  for l in in_layers:
    if not check_in_model(model, l):
      logger.warning(
          "Found specified in_layer `{}` is not in model, please check it."
          .format(l))
    else:
      filted_in_layers.append(l)

  filted_out_layers = []
  for l in out_layers:
    if not check_in_model(model, l):
      logger.warning(
          "Found specified out_layer `{}` is not in model, please check it."
          .format(l))
    else:
      filted_out_layers.append(l)

  if not filted_in_layers:
    in_layers = model.input_names
    logger.warning("There is no valid in_layers, using default "
                   "model.input_names: {}".format(in_layers))
  if not filted_out_layers:
    out_layers = model.output_names
    logger.warning("There is no valid out_layers, using default "
                   "model.output_names: {}".format(out_layers))

  candidate_layers = _get_layers_from_in_out(model, in_layers, out_layers)
  candidate_layers, _ = _remove_ignore_layers(candidate_layers, ignore_layers)
  return candidate_layers


def modify_input_shape(model, new_input_shape, calib_dataset=None):
  """Modify the model's input shape, return modified model."""
  new_inputs = None
  new_num = 1
  if isinstance(new_input_shape, list):
    if isinstance(new_input_shape[0], list):
      new_inputs = [create_input_tensor(shape) for shape in new_input_shape]
      new_num = len(new_input_shape)
    else:
      new_inputs = create_input_tensor(new_input_shape)
  elif isinstance(new_input_shape, tuple):
    new_inputs = create_input_tensor(new_input_shape)
  elif isinstance(new_input_shape, dict):
    new_inputs = {}
    for name, shape in new_input_shape.items():
      dict_inputs = create_input_tensor(shape)
      new_inputs[name] = dict_inputs
    new_num = len(new_input_shape)
  else:
    logger.error('Invalid input shape {}.'.format(new_input_shape))

  if new_num != len(model.inputs):
    logger.error('Model {} expects {} inputs, but got {} input_shape.'.format(
        model.name, len(model.inputs), new_num))
  fixed_shape_model = keras.models.clone_model(model, input_tensors=new_inputs)
  fixed_shape_model.set_weights(model.get_weights())
  #check fixed shape  model
  #for cnt in range(new_num):
  #  shape_list = new_inputs[cnt].shape.as_list()
  #  if shape_list.count(None) or shape_list.count(-1):
  #    if calib_dataset is not None:
  #      new_inputs = calib_dataset
  #      break
  #  else:
  #    continue
  #fixed_shape_model.predict(new_inputs, batch_size=1, steps=1)
  return fixed_shape_model, new_inputs


def adjust_quantize_info(model,
                         quantize_info,
                         adjust_vitis_sigmoid,
                         adjust_shift_cut,
                         adjust_shift_bias,
                         adjust_shift_read,
                         adjust_shift_write,
                         adjust_shift_swish,
                         align_concat=False,
                         align_pool=False):
  return vitis_pof2s_refine_transforms.adjust_quantize_info(
      model, quantize_info, adjust_vitis_sigmoid, adjust_shift_cut,
      adjust_shift_bias, adjust_shift_read, adjust_shift_write,
      adjust_shift_swish, align_concat, align_pool)


def get_sub_layers_dict(subclass_layer):
  """Get sublayers defined in subclass layer's (or model's) init function"""
  sub_layers_dict = {}

  layer_attr = vars(subclass_layer)  # this is a dict
  sub_layers = list(subclass_layer._flatten_layers(
                    recursive=False, include_self=False))

  for k, v in layer_attr.items():
    for layer in sub_layers:
      if v is layer:
        sub_layers_dict[k] = v

      # Sublayers maybe in a list(ListWrapper) or a tuple
      elif (not k.startswith('_') and isinstance(v, (list, tuple))):
        for l in v:
          if not (l is layer):
            continue

          # This attribute will be a list
          if k not in sub_layers_dict:
            sub_layers_dict[k] = [layer]
          elif isinstance(sub_layers_dict[k], list):
            sub_layers_dict[k].append(layer)

  return sub_layers_dict


def set_sub_layer_weights(subclass_layer, quantizable_subclass_layer):
  """Set quantizable sublayers weights from original sublayers.
    Note that both the two input layers should have been built."""

  # Clone the original sublayer's weights to the new sublayer
  sub_layers_dict = get_sub_layers_dict(subclass_layer)

  for name, layer in sub_layers_dict.items():
    if (not isinstance(layer, list)) and (not is_subclass_layer(layer)):
      if not layer.built:
        logger.warning('Not built sublayer. Subclass {}, sublayer {}'.format(
                                  subclass_layer.name, layer.name))
        continue
      #elif layer._build_input_shape is None and len(layer.get_weights()) == 0:
      #  logger.warning('No weight sublayer. Subclass {}, sublayer {}'.format(
      #                            subclass_layer.name, layer.name))
      #  continue

    qattr = getattr(quantizable_subclass_layer, name, None)

    if isinstance(layer, list):
      if not isinstance(qattr, (list, tuple)):
        logger.warning('Not matched sublayers. subclass {}, attribute {}'.format(
                                    subclass_layer.name, name))
        continue

      for i, l in enumerate(layer):
        if is_subclass_layer(l):
          set_sub_layer_weights(l, qattr[i])
        elif isinstance(qattr[i], keras.layers.Layer):
          qattr[i].set_weights(l.get_weights())
        else:
          logger.warning('Not matched sublayers. subclass {}, attribute {}, #{} sublayer {}'.format(
                                      subclass_layer.name, name, i, l.name))
    else:
      if not isinstance(qattr, keras.layers.Layer):
        logger.warning('Not matched sublayer. subclass {}, attribute {}'.format(
                                    subclass_layer.name, name))
        continue

      if is_subclass_layer(layer):
        set_sub_layer_weights(layer, qattr)
      elif isinstance(qattr, keras.layers.Layer):
        qattr.set_weights(layer.get_weights())
      else:
        logger.warning('Not matched sublayer. subclass {}, attribute {}, sublayer {}'.format(
                                    subclass_layer.name, name, layer.name))

  # Initialize the new variables from the original variables
  layer_attrs_dict = vars(subclass_layer)

  for name, attr in layer_attrs_dict.items():
    if name.startswith('_') or not isinstance(attr, tf.Variable):
      continue

    qattr = getattr(quantizable_subclass_layer, name, None)

    if qattr is None:
      logger.warning('Not found variable. subclass {}, attribute {}'.format(
                                  subclass_layer.name, name))

      continue
    elif not isinstance(qattr, tf.Variable):
      logger.warning('Not matched variable. subclass {}, attribute {}'.format(
                                  subclass_layer.name, name))

      continue
    else:
      logger.info('Init from variable. subclass {}, attribute {}'.format(
                                  subclass_layer.name, name))

      setattr(quantizable_subclass_layer, name,
              tf_compat.assign(qattr, tf.keras.backend.get_value(attr)))

  #return  # Following is only suitable for embedding layer

  # Deal with the new added sublayers of quantizable subclass
  sub_layers_dict = get_sub_layers_dict(quantizable_subclass_layer)

  for name, layer in sub_layers_dict.items():
    if isinstance(layer, list):  # Do not support list here
      continue

    layer_type = layer.__class__.__name__

    attr = getattr(subclass_layer, name, None)

    if attr is None:
      logger.info('This is new sublayer. quantizable subclass {}, attribute {}, type {}'.format(
                                  quantizable_subclass_layer.name, name, layer_type))

      continue
    elif is_subclass_layer(attr) and attr.__class__.__name__ == layer_type:
      logger.info('Reserved original subclass layer. quantizable subclass {}, attribute {}, type {}'.format(
                                  quantizable_subclass_layer.name, name, layer_type))

      setattr(quantizable_subclass_layer, name, attr)
    elif isinstance(attr, tf.Variable):
      logger.info('Get weights from variables. quantizable subclass {}, attribute {}, type {}'.format(
                                  quantizable_subclass_layer.name, name, layer_type))

      layer_initial = False

      if (len(layer.get_weights()) == 1):
        # Extract from tf.Variables
        '''
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.Session() as sess:
          layer.set_weights([sess.run(attr)])
          layer_initial = True
        '''

        # Get from layer.weights
        for weight in subclass_layer.get_weights():
          weight_shape = weight.shape if hasattr(weight, "shape") else ()
          if attr.shape.is_compatible_with(weight_shape):
            layer.set_weights([weight])
            layer_initial = True
            break

      if layer_initial == False:
        logger.warning('Not found correspond weights. quantizable subclass {}, attribute {}'.format(
                                  quantizable_subclass_layer.name, name))


def show_sub_layers_tree(subclass_model, show_leaf=False, caption_str=''):
  """Show subclass model's sublayers hierarchical tree"""

  def _gen_indent_str(indent):
    """Add indent to show hierarchy"""
    space_str = ''
    for i in range(indent):
      space_str += '   '
    return space_str

  def _going_to_deeper(layer):
    """Goto deeper layer or not"""
    if show_leaf:
      return len(sublayers(layer)) > 0
    else:
      return is_subclass_layer(layer)

  def _show_sub_layers(subclass, indent=0):
    """Print out hierarchical tree"""
    sub_layers_dict = get_sub_layers_dict(subclass)

    for name, layer in sub_layers_dict.items():
      if isinstance(layer, list):
        for l in layer:
          print(_gen_indent_str(indent) + '|--' + l.name +
                        '(' + l.__class__.__name__ + ')')

          if _going_to_deeper(l):
            _show_sub_layers(l, indent+1)
      else:
        print(_gen_indent_str(indent) + '|--' + layer.name +
                      '(' + layer.__class__.__name__ + ')')

        if _going_to_deeper(layer):
          _show_sub_layers(layer, indent+1)

  if logger.debug_enabled():
    print(subclass_model.name+'('+subclass_model.__class__.__name__+')')
    _show_sub_layers(subclass_model)
    print("+++++++++++++++++++{}+++++++++++++++++++".format(caption_str))

def specific_layers_with_layer_config(quantized_model, specific_layers):
  int_data_type_bit_width = {'int8': 8, 'int16': 16, 'int32': 32}
  quant_prefix_len = len("quant_")
  if specific_layers and len(specific_layers):
    for layer in quantized_model.layers:
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper
                   ) and layer.layer.name in specific_layers.keys():
        layer_quantizer_list = [
            'activation_quantizers', 'bias_quantizers', 'weight_quantizers',
            'output_quantizers'
        ]
        for layer_quantizer_key in layer_quantizer_list:
          if hasattr(layer.quantize_config, layer_quantizer_key):
            layer_quantizer_value = getattr(layer.quantize_config,
                                            layer_quantizer_key)
            for i_quan in layer_quantizer_value:
              int_data_type = specific_layers[layer.layer.name]
              i_quan['quantizer_params']['bit_width'] = int_data_type_bit_width[
                  int_data_type]
      elif isinstance(
          layer, vitis_quantize_layer.VitisQuantize
      ) and layer.name[quant_prefix_len:] in specific_layers.keys():
        int_data_type = specific_layers[layer.name[quant_prefix_len:]]
        layer.quantizer.bit_width = int_data_type_bit_width[int_data_type]
  return quantized_model

def specific_layers_check_datatype(quantized_model, specific_layers):
  int_data_type_bit_width = {'int8': 8, 'int16': 16, 'int32': 32}
  quant_prefix_len = len("quant_")
  if len(specific_layers):
    for layer in quantized_model.layers:
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper
                   ) and layer.layer.name in specific_layers.keys():
        layer_quantizer_list = [
            'activation_quantizers', 'bias_quantizers', 'weight_quantizers',
            'output_quantizers'
        ]
        for layer_quantizer_key in layer_quantizer_list:
          if hasattr(layer.quantize_config, layer_quantizer_key):
            layer_quantizer_value = getattr(layer.quantize_config,
                                            layer_quantizer_key)
            for i_quan in layer_quantizer_value:
              int_data_type = specific_layers[layer.layer.name]
              i_quan['quantizer_params']['bit_width'] = int_data_type_bit_width[
                  int_data_type]
      elif isinstance(
          layer, vitis_quantize_layer.VitisQuantize
      ) and layer.name[quant_prefix_len:] in specific_layers.keys():
        int_data_type = specific_layers[layer.name[quant_prefix_len:]]
        layer.quantizer.bit_width = int_data_type_bit_width[int_data_type]
  return quantized_model


