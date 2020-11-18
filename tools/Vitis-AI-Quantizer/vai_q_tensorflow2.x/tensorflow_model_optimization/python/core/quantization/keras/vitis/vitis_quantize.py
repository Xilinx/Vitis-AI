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

import os
import copy
import math
import collections

import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_config as quantize_config_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize_layer

keras = tf.keras


def quantize_scope(*args):
  """Scope which can be used to deserialize quantized Keras models and layers.

  Under `quantize_scope`, Keras methods such as `tf.keras.load_model` and
  `tf.keras.models.model_from_config` will be able to deserialize Keras models
  and layers which contain quantization classes such as `QuantizeConfig`
  and `Quantizer`.

  Example:

  ```python
  tf.keras.models.save_model(quantized_model, keras_file)

  with quantize_scope():
    loaded_model = tf.keras.models.load_model(keras_file)

  # If your quantized model uses custom objects such as a specific `Quantizer`,
  # you can pass them to quantize_scope to deserialize your model.
  with quantize_scope({'FixedRangeQuantizer', FixedRangeQuantizer}
    loaded_model = tf.keras.models.load_model(keras_file)
  ```

  For further understanding, see `tf.keras.utils.custom_object_scope`.

  Args:
    *args: Variable length list of dictionaries of `{name, class}` pairs to add
      to the scope created by this method.

  Returns:
    Object of type `CustomObjectScope` with quantization objects included.
  """
  quantization_objects = {
      'QuantizeAnnotate':
          quantize_annotate_mod.QuantizeAnnotate,
      'QuantizeAwareActivation':
          vitis_quantize_aware_activation.QuantizeAwareActivation,
      'NoQuantizeActivation':
          vitis_quantize_aware_activation.NoQuantizeActivation,
      'QuantizeWrapper':
          vitis_quantize_wrapper.QuantizeWrapper,
      'QuantizeLayer':
          vitis_quantize_layer.QuantizeLayer,
  }
  quantization_objects.update(vitis_quantizers._types_dict())  # pylint: disable=protected-access
  quantization_objects.update(vitis_8bit_quantize_configs._types_dict())  # pylint: disable=protected-access

  return tf.keras.utils.custom_object_scope(*(args + (quantization_objects,)))


def get_quantize_info(model):
  """Get the quantize info of the model"""
  quantize_info = {}
  for layer in model.layers:
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      quantize_info[layer.layer.name] = layer.get_quantize_info()
    elif isinstance(layer, vitis_quantize_layer.QuantizeLayer):
      quantize_info[layer.name] = layer.get_quantize_info()
  return quantize_info


def set_quantize_info(model, new_quantize_info):
  """Set the quantize info of the model"""
  for layer in model.layers:
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper
                 ) and layer.layer.name in new_quantize_info:
      layer.set_quantize_info(new_quantize_info[layer.layer.name])
    elif isinstance(
        layer,
        vitis_quantize_layer.QuantizeLayer) and layer.name in new_quantize_info:
      layer.set_quantize_info(new_quantize_info[layer.name])
  return


def is_quantize_layer(layer):
  """Check if QuantizeWrapper or QuantizeLayer."""
  return isinstance(layer,
                    vitis_quantize_wrapper.QuantizeWrapper) or isinstance(
                        layer, vitis_quantize_layer.QuantizeLayer)


def set_layer_mode(model, mode, layer_names=None):
  """Set the mode of QuantizeWrapper and QuantizeLayer."""
  for layer in model.layers:
    if is_quantize_layer(layer):
      if not layer_names or layer.name in layer_names:
        layer.mode = mode


def clone_model_with_weights(model_to_clone):
  """Clone keras model with weights."""
  with quantize_scope():
    cloned_model = keras.models.clone_model(model_to_clone)
    cloned_model.set_weights(model_to_clone.get_weights())
  return cloned_model


def dump_model_weights(model, output_dir):
  """Dump model weights."""
  # Get weight quantize info
  w_q_map = {}
  for layer in model.layers:
    if is_quantize_layer(layer):
      if isinstance(layer, vitis_quantize_layer.QuantizeLayer):
        continue
      layer_quantize_info = layer.get_quantize_info()
      for name, value in layer_quantize_info.items():
        if value.get('type') == 'weight':
          w_name = name.rstrip(':0')
          w_q_map[w_name] = value['info']['quant_pos_var']

  print("[INFO] Dumping weights/biases...")
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
    print("[INFO] Dumping ({}/{}): {}".format(index, len(w_q_map), w_name))

    res = w.numpy()
    res = res.flatten()
    if w_name in w_q_map:
      res = np.round(res * 2**w_q_map[w_name])
      res = res.clip(-128, 127)
      res.astype(np.int8).tofile(filename + ".bin")
      np.savetxt(
          filename + ".txt", res.astype(np.int8), fmt="%s", delimiter=",")


def dump_model_activations(model, output_dir, dataset):
  """Dump model activation."""
  # Get activation quantize info
  a_q_map = {}
  for layer in model.layers:
    if is_quantize_layer(layer):
      layer_quantize_info = layer.get_quantize_info()
      if isinstance(layer, vitis_quantize_layer.QuantizeLayer):
        a_q_map[layer.name] = layer_quantize_info['info']['quant_pos_var']
      else:
        for name, value in layer_quantize_info.items():
          if value.get('type') in [
              'output', 'pre_activation', 'post_activation'
          ]:
            a_q_map[layer.name] = value['info']['quant_pos_var']

  quant_layers = [layer for layer in model.layers if is_quantize_layer(layer)]
  model = keras.Model(
      inputs=model.inputs, outputs=[layer.output for layer in quant_layers])

  print("[INFO] Dumping activations...")
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
    print("[INFO] Dumping ({}/{}): {}".format(index, len(quant_layers), a_name))
    res = res.flatten()
    if a_name in a_q_map:
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and any(
          act._should_pre_quantize() for act in layer._quantize_activations):
        res.tofile(filename + '_float.bin')
        np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")
      else:
        res = res * 2**a_q_map[a_name]
        res.astype(np.int8).tofile(filename + ".bin")
        np.savetxt(
            filename + ".txt", res.astype(np.int8), fmt="%s", delimiter=",")


class CollectQuantizeInfoCallback(keras.callbacks.Callback):
  """Callback to collect the quantize info of each batch."""

  def __init__(self):
    super(CollectQuantizeInfoCallback, self).__init__()
    self._quantize_info = collections.OrderedDict()

  def on_predict_batch_end(self, batch, logs=None):
    self._quantize_info[batch] = get_quantize_info(self.model)

  @property
  def quantize_info(self):
    return self._quantize_info

  def get_last_quantize_info(self):
    return next(reversed(self._quantize_info.values()))

  def get_most_common_quantize_info(self):
    pos_map = {}
    for batch_quantize_info in self._quantize_info.values():
      for layer, q_info in batch_quantize_info.items():
        if q_info.get('type') == 'input':
          if layer not in pos_map:
            pos_map[layer] = {'input': []}
          pos = q_info['info']['quant_pos_var']
          pos_map[layer]['input'].append(pos)
        else:
          if layer not in pos_map:
            pos_map[layer] = {}
          for k, v in q_info.items():
            if not v:
              continue
            if k not in pos_map[layer]:
              pos_map[layer][k] = []
            pos = v['info']['quant_pos_var']
            pos_map[layer][k].append(pos)

    mc_pos_map = {}
    for layer, q_info in pos_map.items():
      mc_pos_map[layer] = {}
      for k, v in q_info.items():
        mc_pos_map[layer][k] = max(v, key=v.count)

    _, mc_quantize_info = self._quantize_info.popitem()
    for layer, q_info in mc_quantize_info.items():
      if q_info.get('type') == 'input':
        if q_info['info']['quant_pos_var'] != mc_pos_map[layer]['input']:
          q_info['info']['quant_pos_var'] = mc_pos_map[layer]['input']
      else:
        for k, v in q_info.items():
          if not v:
            continue
          if v['info']['quant_pos_var'] != mc_pos_map[layer][k]:
            v['info']['quant_pos_var'] = mc_pos_map[layer][k]

    return mc_quantize_info


class VitisQuantizer(object):
  """Vitis Quantizer main APIs"""

  def __init__(self, float_model, custom_quantize_strategy=None):
    """Init VitisQuantizer."""
    self._float_model = float_model
    self._qat_model = None
    self._qcb_model = None
    self._qcbev_model = None
    self._analyse_model = None
    self._optimized_model = None

    self._quantize_registry = vitis_8bit_quantize_registry.QuantizeRegistry()

    # Custom quantizer strategy
    self._custom_quantize_strategy = custom_quantize_strategy
    if custom_quantize_strategy:
      self._quantize_registry.update_quantize_strategy(custom_quantize_strategy)

  def _create_qat_model(self):
    """Create quantize-aware training model."""
    self._qat_model = create_quantize_model(
        self._optimized_model, self._quantize_registry, mode='QAT')

  def _run_model_with_collector(self, model, dataset):
    """Run model with quantize info collector."""
    collector = CollectQuantizeInfoCallback()
    model.predict(
        dataset, batch_size=50, verbose=1, steps=None, callbacks=[collector])
    return collector

  def _create_optimized_model(self, remove_dropout, fold_conv_bn, fold_bn,
                              replace_relu6, include_cle, cle_steps):
    """Create optimized model."""
    self._optimized_model, _ = create_optimize_model(self._float_model, {},
                                                     remove_dropout,
                                                     fold_conv_bn, fold_bn,
                                                     replace_relu6, include_cle,
                                                     cle_steps)

  def _create_analysed_model(self, dataset):
    """Create analysed model."""
    self._analysed_model = create_quantize_model(
        self._float_model, self._quantize_registry, mode='ANALYSE')

    print("[INFO] Start Model Analyse...")
    collector = self._run_model_with_collector(self._analysed_model, dataset)
    print("[INFO] Model Analyse Done.")
    #  model_info = collector.get_last_quantize_info()
    model_info = collector.get_most_common_quantize_info()
    return model_info

  def _freeze_quantize_info(self, quantize_info):
    """Freeze the quantize info into the quantize evaluate model."""
    if not self._qcb_model:
      raise ValueError('No qcb_model found.')

    #  Create quantize calibration evaluation model
    self._qcbev_model = clone_model_with_weights(self._qcb_model)
    set_layer_mode(self._qcbev_model, 'QCBEV')

    # Freeze the quantize info into the model
    set_quantize_info(self._qcbev_model, quantize_info)

  def _adjust_quantize_info(self,
                            quantize_info,
                            adjust_shift_cut=True,
                            adjust_shift_bias=True):
    """Adjust the quantize info to meet the compiler constraints."""

    def _get_pos(layer, quantize_info, key):
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        q_info = quantize_info[layer.layer.name]
        for k, v in q_info.items():
          if key == 'w' and v['type'] == 'weight' and k.endswith('kernel:0'):
            return v['info']['quant_pos_var']
          elif key == 'b' and v['type'] == 'weight' and k.endswith('bias:0'):
            return v['info']['quant_pos_var']
          elif key == 'o':
            if v.get('type') in ['post_activation', 'pre_activation', 'output']:
              return v['info']['quant_pos_var']
      elif isinstance(layer, vitis_quantize_layer.QuantizeLayer):
        if key == 'o':
          q_info = quantize_info[layer.name]
          return q_info['info']['quant_pos_var']
      else:
        return None

    def _set_pos(layer, quantize_info, key, new_pos):
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        q_info = quantize_info[layer.layer.name]
        for k, v in q_info.items():
          if key == 'w' and v['type'] == 'weight' and k.endswith('kernel:0'):
            v['info']['quant_pos_var'] = new_pos
          elif key == 'b' and v['type'] == 'weight' and k.endswith('bias:0'):
            v['info']['quant_pos_var'] = new_pos
          elif key == 'o' and v['type'] in [
              'post_activation', 'pre_activation', 'output'
          ]:
            v['info']['quant_pos_var'] = new_pos
      elif isinstance(layer, vitis_quantize_layer.QuantizeLayer):
        if key == 'o':
          q_info = quantize_info[layer.name]
          q_info['info']['quant_pos_var'] = new_pos

    def _adjust_shift_cut(layer, adjusted_quantize_info, ip, wp, bp, op):
      min_sc = 0
      max_sc = 16
      sc = wp + ip - op

      new_sc = None
      if sc < min_sc:
        new_sc = min_sc
      elif sc > max_sc:
        new_sc = max_sc

      if new_sc:
        new_wp = min_sc + op - ip
        _set_pos(layer, adjusted_quantize_info, 'w', new_wp)
        print('[INFO] Shift cut of layer {} is {}. It exceed range [{}, {}]. '
              'Modify wpos from {} to {}.'.format(layer.name, int(sc),
                                                  int(min_sc), int(max_sc),
                                                  int(wp), int(new_wp)))

    def _adjust_shift_bias(layer, adjusted_quantize_info, ip, wp, bp, op):
      sc = wp + ip - op
      min_sb = min(0, -(24 - (8 + sc)))
      max_sb = 16
      sb = wp + ip - bp

      new_sb = None
      if sb < min_sb:
        new_sb = min_sb
      elif sb > max_sb:
        new_sb = max_sb

      if new_sb:
        new_bp = wp + ip - new_sb
        _set_pos(layer, adjusted_quantize_info, 'b', new_bp)
        print('[INFO] Shift bias of layer {} is {}. It exceed range [{}, {}]. '
              'Modify bpos from {} to {}.'.format(layer.name, int(sb),
                                                  int(min_sb), int(max_sb),
                                                  int(bp), int(new_bp)))

    adjusted_quantize_info = copy.deepcopy(quantize_info)

    for i in range(len(self._qcbev_model.layers)):
      if i == 0:
        continue

      layer = self._qcbev_model.layers[i]
      if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        continue

      if not (isinstance(layer.layer, keras.layers.Conv2D) or
              isinstance(layer.layer, keras.layers.DepthwiseConv2D) or
              isinstance(layer.layer, keras.layers.Conv2DTranspose) or
              isinstance(layer.layer, keras.layers.Dense)):
        continue

      wp = _get_pos(layer, quantize_info, 'w')
      bp = _get_pos(layer, quantize_info, 'b')

      if isinstance(layer.layer.activation.activation,
                    vitis_quantize_aware_activation.NoQuantizeActivation):
        post_layer = layer.outbound_nodes[0].outbound_layer
        op = _get_pos(post_layer, quantize_info, 'o')
      else:
        op = _get_pos(layer, quantize_info, 'o')

      pre_layer = layer.inbound_nodes[0].inbound_layers
      ip = _get_pos(pre_layer, quantize_info, 'o')
      if all([ip, wp, bp, op]):
        if adjust_shift_cut:
          _adjust_shift_cut(layer, adjusted_quantize_info, ip, wp, bp, op)
        if adjust_shift_bias:
          _adjust_shift_bias(layer, adjusted_quantize_info, ip, wp, bp, op)
      else:
        print('[Warning] Skip quantize pos adjustment for layer {}, '
              'its quantize pos is [i={}, w={}, b={}, o={}]'.format(
                  layer.name, ip, wp, bp, op))
    return adjusted_quantize_info

  def _calibrate_without_loss(self, calib_dataset):
    """Calibrate model without loss, only with unlabeled dataset."""
    # Create quantize calibration model
    self._qcb_model = create_quantize_model(
        self._optimized_model, self._quantize_registry, mode='QCB')

    print("[INFO] Start Quantize Calibration...")
    collector = self._run_model_with_collector(self._qcb_model, calib_dataset)
    print("[INFO] Quantize Calibration Done.")

    print("[INFO] Start Generating Quantized Model...")
    #  Create quantize calibration evaluation model
    self._qcbev_model = clone_model_with_weights(self._qcb_model)
    set_layer_mode(self._qcbev_model, 'QCBEV')

    # Freeze the quantize info into the model, now using most_common_quantize_info
    #  last_quantize_info = collector.get_last_quantize_info()
    common_quantize_info = collector.get_most_common_quantize_info()
    adjusted_quantize_info = self._adjust_quantize_info(
        common_quantize_info, adjust_shift_cut=True, adjust_shift_bias=True)
    self._freeze_quantize_info(adjusted_quantize_info)
    print("[INFO] Generating Quantized Model Done.")

  def _calibrate_with_loss(self, loss, metrics, calib_dataset, eval_dataset,
                           verbose):
    """Calibrate model with loss and metrics to get better accuracy, need eval_dataset."""
    self._calibrate_without_loss(calib_dataset)
    init_quantize_info = get_quantize_info(self._qcbev_model)

    quantize_layers = {}
    for layer in self._qcb_model.layers:
      if is_quantize_layer(layer):
        quantize_layers[layer.name] = layer

    def _recompile(model):
      """Helper function to re-compile the model."""
      # Must reset metrics to get accurate results
      for m in metrics:
        if not isinstance(m, str):
          m.reset_states()
      model.compile(loss=loss, metrics=metrics)

    def _evaluate(model):
      """Helper function to evaluate model to get loss and accuracy."""
      _recompile(model)
      if isinstance(eval_dataset, tuple):
        eval_images, eval_labels = eval_dataset
        return model.evaluate(
            eval_images, eval_labels, verbose=verbose, return_dict=True)
      else:
        return model.evaluate(eval_dataset, verbose=verbose, return_dict=True)

    def _print_results(results, title=''):
      """Helper function to print evaluation results."""
      pstr = '[' + title + ']: ' if title else ''
      for k, v in results.items():
        pstr += '\t{}: {}'.format(k, v)
      print(pstr)

    # Get float results
    set_layer_mode(self._qcb_model, 'ANALYSE')
    float_results = _evaluate(self._qcb_model)
    _print_results(float_results, 'float_results')

    # Get simple quantize calibrated results
    init_results = _evaluate(self._qcbev_model)
    _print_results(init_results, 'init_results')

    # Do quantize pos searching
    print("[INFO] Start Quantize Position Searching...")
    set_layer_mode(self._qcb_model, 'QCBEV')
    best_results = init_results
    best_quantize_info = copy.deepcopy(init_quantize_info)
    count = 0
    for name, layer in quantize_layers.items():
      count += 1
      print('[INFO] ({}/{})Processing layer: {}'.format(count,
                                                        len(quantize_layers),
                                                        name))

      def _search_optimal_pos(init_quantize_info,
                              init_results,
                              layer_name,
                              quantizer_name,
                              delta=[-1, 1, 2]):
        new_best_results = init_results
        new_best_quantize_info = copy.deepcopy(init_quantize_info)

        tmp_quantize_info = copy.deepcopy(init_quantize_info)
        layer_info = tmp_quantize_info[layer_name]
        if quantizer_name == 'NoQuantizeActivation':
          return new_best_quantize_info, new_best_results
        elif quantizer_name == 'input':
          q_info = layer_info['info']
        else:
          q_info = layer_info[quantizer_name]['info']
        q_pos = q_info['quant_pos_var']

        for dt in delta:
          if verbose:
            print('[INFO]  Try change {}.{}: {} -> {}'.format(
                layer_name, quantizer_name, q_pos, q_pos + dt))
          q_info['quant_pos_var'] = q_pos + dt
          set_quantize_info(self._qcb_model, tmp_quantize_info)
          q_results = _evaluate(self._qcb_model)
          if q_results['loss'] < new_best_results['loss']:
            new_best_results = q_results
            new_best_quantize_info = copy.deepcopy(tmp_quantize_info)
            _print_results(new_best_results, 'Update Best Results')
        return new_best_quantize_info, new_best_results

      # Quantize Layer
      if isinstance(layer, vitis_quantize_layer.QuantizeLayer):
        best_quantize_info, best_results = _search_optimal_pos(
            init_quantize_info=best_quantize_info,
            init_results=best_results,
            layer_name=layer.name,
            quantizer_name='input')
      # Quantize Wrappers
      elif isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        layer_info = best_quantize_info[layer.layer.name]
        for quantizer_name, q_info in layer_info.items():
          best_quantize_info, best_results = _search_optimal_pos(
              init_quantize_info=best_quantize_info,
              init_results=best_results,
              layer_name=layer.layer.name,
              quantizer_name=quantizer_name)

    print("[INFO] Done Quantize Position Searching.")
    _print_results(best_results, 'Final Best Results')

    # Freeze the quantize info into the model, now using last_quantize_info
    self._freeze_quantize_info(best_quantize_info)

  def optimize_model(self,
                     remove_dropout=True,
                     fold_conv_bn=True,
                     fold_bn=True,
                     replace_relu6=True,
                     include_cle=True,
                     cle_steps=10):
    """Get optimized model."""
    if not self._optimized_model:
      self._create_optimized_model(remove_dropout, fold_conv_bn, fold_bn,
                                   replace_relu6, include_cle, cle_steps)
    return self._optimized_model

  def get_analysed_model(self, dataset):
    """Get analysed model."""
    if not self._analyse_model:
      model_info = self._create_analysed_model(dataset)
    return self._analysed_model, model_info

  def get_qat_model(self,
                    remove_dropout=True,
                    fold_conv_bn=True,
                    fold_bn=True,
                    replace_relu6=True,
                    include_cle=True,
                    cle_steps=10):
    """Get quantize-aware training model."""
    self.optimize_model(remove_dropout, fold_conv_bn, fold_bn, replace_relu6,
                        include_cle, cle_steps)
    if not self._qat_model:
      self._create_qat_model()
    return self._qat_model

  def quantize_model(
      self,
      loss=None,
      metrics=None,
      calib_dataset=None,
      eval_dataset=None,
      verbose=0,
      remove_dropout=True,
      fold_conv_bn=True,
      fold_bn=True,
      replace_relu6=True,
      include_cle=True,
      cle_steps=10,
  ):
    """Interface of quantize calibration."""
    if not 'calib_dataset':
      raise ValueError(
          'Need to assign `calib_dataset` for when calling quantize_model().')

    if loss and not eval_dataset:
      raise ValueError(
          'Need to assign `eval_dataset` for when calling quantize_model(loss=loss_fn).'
      )

    self.optimize_model(remove_dropout, fold_conv_bn, fold_bn, replace_relu6,
                        include_cle, cle_steps)

    if not self._qcb_model or not self._qcbev_model:
      if loss:
        self._calibrate_with_loss(loss, metrics, calib_dataset, eval_dataset,
                                  verbose)
      else:
        self._calibrate_without_loss(calib_dataset)
    return self._qcbev_model

  @staticmethod
  def dump_model(model,
                 dataset=None,
                 output_dir='./dump_results',
                 weights_only=False):
    """Dump golden results of quantized model."""
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    if not weights_only and dataset is None:
      raise ValueError('`dataset` is needed to dump with activation.')

    print("INFO: Start Dumping...")
    dump_model_weights(model, output_dir)
    if not weights_only:
      dump_model_activations(model, output_dir, dataset)


def create_optimize_model(model, layer_quantize_map, remove_dropout,
                          fold_conv_bn, fold_bn, replace_relu6, include_cle,
                          cle_steps):
  """Optimize a `tf.keras` model before quantization, such as bn folding, activation folding."""
  optimize_transform = \
    vitis_8bit_quantize_layout_transform.Vitis8BitOptimizeLayoutTransform()
  optimized_model, layer_quantize_map = optimize_transform.apply(
      model, layer_quantize_map, remove_dropout, fold_conv_bn, fold_bn,
      replace_relu6, include_cle, cle_steps)
  #  optimized_model.save('optimized.h5')
  return optimized_model, layer_quantize_map


def create_quantize_model(to_quantize, quantize_registry, mode):
  """Quantize a `tf.keras` model with the default quantization implementation.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  For more information, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training

  Quantize a model:

  ```python
  # Quantize sequential model
  model = create_quantize_model(
      keras.Sequential([
          layers.Dense(10, activation='relu', input_shape=(100,)),
          layers.Dense(2, activation='sigmoid')
      ]))

  # Quantize functional model
  in = tf.keras.Input((3,))
  out = tf.keras.Dense(2)(in)
  model = tf.keras.Model(in, out)

  quantized_model = create_quantize_model(model)
  ```

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    to_quantize: tf.keras model to be quantized. It can have pre-trained
      weights.

  Returns:
    Returns a new `tf.keras` model prepared for quantization.
  """
  if to_quantize is None:
    raise ValueError('`to_quantize` cannot be None')

  if not isinstance(to_quantize, keras.Model):
    raise ValueError(
        '`to_quantize` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers.'
        'You passed an instance of type: {input}.'.format(
            input=to_quantize.__class__.__name__))

  if not isinstance(to_quantize, keras.Sequential) \
      and not to_quantize._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_quantize` can only either be a tf.keras Sequential or '
        'Functional model.')

  AVAILABLE_MODES = ['QCB', 'QAT', 'ANALYSE', 'QCBEV']
  if mode not in AVAILABLE_MODES:
    raise ValueError('Mode `{}` is not valid, available modes are:{}.'.format(
        mode, AVAILABLE_MODES))

  annotated_model = quantize_annotate_model(to_quantize)
  return quantize_apply(annotated_model, quantize_registry, mode)


def quantize_annotate_model(to_annotate):
  """Annotate a `tf.keras` model to be quantized.

  This function does not actually quantize the model. It merely specifies
  that the model needs to be quantized. `quantize_apply` can then be used
  to quantize the model.

  This function is intended to be used in conjunction with the
  `quantize_annotate_layer` API. Otherwise, it is simpler to use
  `create_quantize_model`.

  Annotate a model while overriding the default behavior for a layer:

  ```python
  quantize_config = MyDenseQuantizeConfig()

  model = quantize_annotate_model(
    keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(
          layers.Dense(2, activation='sigmoid'),
          quantize_config=quantize_config)
    ]))

  # The first Dense layer gets quantized with the default behavior,
  # but the second layer uses `MyDenseQuantizeConfig` for quantization.
  quantized_model = quantize_apply(model)
  ```

  Note that this function removes the optimizer from the original model.

  Args:
    to_annotate: `tf.keras` model which needs to be quantized.

  Returns:
    New tf.keras model with each layer in the model wrapped with
    `QuantizeAnnotate`. The new model preserves weights from the original
    model.
  """
  if to_annotate is None:
    raise ValueError('`to_annotate` cannot be None')

  if not isinstance(to_annotate, keras.Model):
    raise ValueError(
        '`to_annotate` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers. '
        'You passed an instance of type: {input}.'.format(
            input=to_annotate.__class__.__name__))

  if not isinstance(to_annotate, keras.Sequential) \
      and not to_annotate._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_annotate` can only either be a tf.keras Sequential or '
        'Functional model.')

  def _add_quant_wrapper(layer):
    """Add annotation wrapper."""
    # Already annotated layer. No need to wrap.
    if isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
      return layer

    if isinstance(layer, tf.keras.Model):
      raise ValueError(
          'Quantizing a tf.keras Model inside another tf.keras Model is not supported.'
      )

    return quantize_annotate_mod.QuantizeAnnotate(layer)

  return keras.models.clone_model(
      to_annotate, input_tensors=None, clone_function=_add_quant_wrapper)


def quantize_annotate_layer(to_annotate, quantize_config=None):
  """Annotate a `tf.keras` layer to be quantized.

  This function does not actually quantize the layer. It is merely used to
  specify that the layer should be quantized. The layer then gets quantized
  accordingly when `quantize_apply` is used.

  This method should be used when the user wants to quantize only certain
  layers of the model, or change the default behavior of how a layer is
  quantized.

  Annotate a layer:

  ```python
  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'))
  ])

  # Only the second Dense layer is quantized.
  quantized_model = quantize_apply(model)
  ```

  Args:
    to_annotate: `tf.keras` layer which needs to be quantized.
    quantize_config: optional `QuantizeConfig` which controls how the layer is
      quantized. In its absence, the default behavior for the layer is used.

  Returns:
    `tf.keras` layer wrapped with `QuantizeAnnotate`.
  """
  if to_annotate is None:
    raise ValueError('`to_annotate` cannot be None')

  # Check against keras.Model since it is an instance of keras.layers.Layer.
  if not isinstance(to_annotate, keras.layers.Layer) or isinstance(
      to_annotate, keras.Model):
    raise ValueError(
        '`to_annotate` can only be a `tf.keras.layers.Layer` instance. '
        'You passed an instance of type: {input}.'.format(
            input=to_annotate.__class__.__name__))

  if quantize_config is not None and not isinstance(
      quantize_config, quantize_config_mod.QuantizeConfig):
    raise ValueError(
        '`quantize_config` can only be a `tfmot.quantization.keras.QuantizeConfig` instance.'
        'You passed an instance of type: {input}.'.format(
            input=quantize_config.__class__.__name__))

  return quantize_annotate_mod.QuantizeAnnotate(
      layer=to_annotate, quantize_config=quantize_config)


def quantize_apply(model, quantize_registry, mode):
  """Quantize a `tf.keras` model that has been annotated for quantization.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  For more information, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training
  TODO(tfmot): Link blog once launched.

  This function takes a `tf.keras` model in which the desired layers for
  quantization have already been annotated. See `quantize_annotate_model`
  and `quantize_annotate_layer`.

  Quantize model.
  ```python
  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(layers.Dense(2, activation='sigmoid'))
  ])

  # Only the second Dense layer is quantized.
  quantized_model = quantize_apply(model)
  ```

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    model: A `tf.keras` Sequential or Functional model which has been annotated
      with `quantize_annotate`. It can have pre-trained weights.

  Returns:
    Returns a new `tf.keras` model in which the annotated layers have been
    prepared for quantization.
  """
  if model is None:
    raise ValueError('`model` cannot be None')

  if not isinstance(model, keras.Model):
    raise ValueError('`model` can only be a `tf.keras.Model` instance.'
                     'You passed an instance of type: {input}.'.format(
                         input=model.__class__.__name__))

  if not isinstance(model, keras.Sequential) \
      and not model._is_graph_network:  # pylint: disable=protected-access
    raise ValueError('`model` can only either be a tf.keras Sequential or '
                     'Functional model.')

  if not model.built:
    raise ValueError('`model` must be a built model. '
                     'been built yet. Please call `model.build(input_shape)` '
                     'before quantizing your model.')

  def _extract_original_model(model_to_unwrap):
    """Extracts original model by removing wrappers."""
    layer_quantize_map = {}

    def _unwrap(layer):
      if not isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
        return layer

      annotate_wrapper = layer
      layer_quantize_map[annotate_wrapper.layer.name] = {
          'quantize_config': annotate_wrapper.quantize_config
      }
      return annotate_wrapper.layer

    unwrapped_model = keras.models.clone_model(
        model_to_unwrap, input_tensors=None, clone_function=_unwrap)

    return unwrapped_model, layer_quantize_map

  def _make_quantize_fn(mode):

    def quantize_fn(layer):  # pylint: disable=missing-docstring
      if layer.name not in layer_quantize_map:
        return layer

      quantize_config = layer_quantize_map[layer.name].get('quantize_config')
      if not quantize_config and quantize_registry.supports(layer):
        quantize_config = quantize_registry.get_quantize_config(layer)

      if not quantize_config:
        warning_msg = (
            '[WARNING] Layer {}:{} is not quantized. You can quantize this '
            'layer by passing a custom quantize strategy to the quantizer. '
            'For example of quantize strategy, please see the [Vitis AI User Document]'
            '(https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html#documentation)'
        )
        print(
            warning_msg.format(layer.name, layer.__class__,
                               quantize_registry.__class__))
        return layer

      # `QuantizeWrapper` does not copy any additional layer params from
      # `QuantizeAnnotate`. This should generally be fine, but occasionally
      # `QuantizeAnnotate` wrapper may contain `batch_input_shape` like params.
      # TODO(pulkitb): Ensure this does not affect model cloning.
      return vitis_quantize_wrapper.QuantizeWrapper(layer, quantize_config,
                                                    mode)

    return quantize_fn

  # 1. Create a copy of the model with the same weights. This ensures
  # modifications don't affect the original model, or its weights.
  try:
    model_copy = clone_model_with_weights(model)
  except ValueError:
    raise ValueError(
        'Unable to clone model. This generally happens if you used custom Keras layers or objects '
        'in your model. Please specify them via `quantize_scope` for your calls to `create_quantize_model`'
    )

  # 2. Remove QuantizeAnnotate wrappers from the layers in the model. This
  # extracts the original model structure (easier to transform), and
  # stores relevant quantization information in a map.
  unwrapped_model, layer_quantize_map = _extract_original_model(model_copy)
  # Model cloning excludes input layers. Add input layers into the map
  # since they need to be matched for patterns as well.
  # pylint: disable=protected-access
  for input_layer in unwrapped_model._input_layers:
    for outbound_node in input_layer._outbound_nodes:
      if outbound_node.outbound_layer.name in layer_quantize_map:
        layer_quantize_map[input_layer.name] = {}
  # pylint: enable=protected-access

  # 3. Apply the graph transformations required to match model passes on
  # target device/dialect.
  quantize_transform = \
    vitis_8bit_quantize_layout_transform.Vitis8BitQuantizeLayoutTransform()
  # layer_quantize_map gets modified by the transformations.
  transformed_model, layer_quantize_map = quantize_transform.apply(
      unwrapped_model, layer_quantize_map, quantize_registry, mode)

  # 4. Actually quantize all the relevant layers in the model. This is done by
  # wrapping the layers with QuantizeWrapper, and passing the associated
  # `QuantizeConfig`.
  return keras.models.clone_model(
      transformed_model,
      input_tensors=None,
      clone_function=_make_quantize_fn(mode))
