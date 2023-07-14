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

bfloat16 = tf.bfloat16.as_numpy_dtype

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
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.bfp import vitis_bfp_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.tqt import vitis_tqt_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fs import vitis_fs_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.fsx import vitis_fsx_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.gpu import vitis_gpu_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import vai_utf_parser
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import subclass_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common.entropy_percentile import calibrator_numpy

from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize_func
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize_subclass

logger = common_utils.VAILogger
keras = tf.keras

Usage = (
    '\nVitisQuantizer introduce a new feature to quantize model for specific DPU `target` '
    'since Vitis-AI 3.0. Without `target` we can only give some general, target-independent '
    'quantize results. Please assign `target` to get more accurate partition and quantization '
    'results. This feature in only available for default pof2s quantize strategy due to DPU '
    'limitation. See user guide for more information. This feature is experimental, please '
    'contact us if you meet with bugs or problems.'
    '\n'
    '\nUsage:'
    '\n[1] Quantize a model with target:'
    '\n    VitisQuantizer(my_model, target=my_target).quantize_model(calib_dataset)'
    '\n'
    '\n[Note] `my_target` is the target DPU to deploy this model, it can be a name(e.g. "DPUCZDX8G_ISA1_B4096"), '
    'a arch json(e.g. "./U50/arch.json") or a fingerprint.\n')


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
      'QuantizeAwareActivation':
          vitis_quantize_aware_activation.QuantizeAwareActivation,
      'NoQuantizeActivation':
          vitis_quantize_aware_activation.NoQuantizeActivation,
      'QuantizeWrapper':
          vitis_quantize_wrapper.QuantizeWrapper,
      'CustomOpWrapper':
          vitis_custom_wrapper.CustomOpWrapper,
  }
  quantization_objects.update(vitis_quantizers._types_dict())
  quantization_objects.update(vitis_quantize_configs._types_dict())
  quantization_objects.update(vitis_quantize_layer._types_dict())
  quantization_objects.update(vitis_activation._types_dict())
  quantization_objects.update(vitis_pooling._types_dict())

  return tf.keras.utils.custom_object_scope(*(args + (quantization_objects,)))


class CollectQuantizeInfoCallback(keras.callbacks.Callback):
  """Callback to collect the quantize info of each batch."""

  def __init__(self):
    super(CollectQuantizeInfoCallback, self).__init__()
    self._quantize_info = collections.OrderedDict()

  def on_predict_batch_end(self, batch, logs=None):
    self._quantize_info[batch] = model_utils.get_quantize_info(self.model)

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
        if isinstance(v[0], np.ndarray):
          mc_pos_map[layer][k] = v[0]
        else:
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
          if isinstance(mc_pos_map[layer][k], np.ndarray):
            continue
          if v['info']['quant_pos_var'] != mc_pos_map[layer][k]:
            v['info']['quant_pos_var'] = mc_pos_map[layer][k]

    return mc_quantize_info

  def get_quantizer_q_min_max(self, quantizer_params):
    bit_width = quantizer_params['bit_width']
    unsigned = quantizer_params['unsigned']
    narrow_range = False if unsigned else quantizer_params['narrow_range']
    if unsigned:
      bound = float(2**bit_width)
      q_min, q_max = 0., bound - 1
    else:
      bound = float(2**(bit_width - 1))
      q_min, q_max = -bound, bound - 1
      if narrow_range:
        q_min += 1
    return q_min, q_max

  def get_scale_min_max(self, method, quantize_information, q_min, q_max,
                        quantizer_params):
    if vitis_quantize_ops.QuantizeMethod.MIN_KL == vitis_quantize_ops.QuantizeMethod(
        method):
      batch_max = calibrator_numpy.numpy_kl_div(
          quantize_information['info']['calib_hist'],
          quantize_information['info']['calib_bin_edges'])
      batch_min = -batch_max
      scale = vitis_quantize_ops.get_scale(batch_min, batch_max, q_min, q_max)
      quantize_information['info']['min_var'] = batch_min
      quantize_information['info']['max_var'] = batch_max
      quantize_information['info']['scale'] = scale
    elif vitis_quantize_ops.QuantizeMethod.PERCENTILE == vitis_quantize_ops.QuantizeMethod(
        method):
      percentile = quantizer_params['method_percentile']
      batch_max = calibrator_numpy.numpy_percentile(
          percentile, quantize_information['info']['calib_hist'],
          quantize_information['info']['calib_bin_edges'])
      batch_min = -batch_max
      scale = vitis_quantize_ops.get_scale(batch_min, batch_max, q_min, q_max)
      quantize_information['info']['min_var'] = batch_min
      quantize_information['info']['max_var'] = batch_max
      quantize_information['info']['scale'] = scale
    elif quantize_information.get('type') in ['weight']:
      scale = vitis_quantize_ops.get_scale(
          quantize_information['info']['min_var'],
          quantize_information['info']['max_var'], q_min, q_max)
      quantize_information['info']['scale'] = scale
    return quantize_information

  def get_entropy_percentile_amax(self, model):
    quantize_info = collections.OrderedDict()
    progress_total = len(model.layers)
    progbar = keras.utils.Progbar(progress_total)
    for progress, layer in enumerate(model.layers):
      progbar.update(progress + 1)
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        layer_quantize_info = layer.get_quantize_info()
        for name, quantize_information in layer_quantize_info.items():
          method = 0
          if isinstance(quantize_information,
                        dict) and (quantize_information.get('type')
                                   in ['post_activation', 'pre_activation']):
            quantizer_params = layer.get_config()['quantize_config']['config'][
                'activation_quantizers'][0]['quantizer_params']
            method = quantizer_params['method']
            q_min, q_max = self.get_quantizer_q_min_max(quantizer_params)
          elif isinstance(quantize_information,
                          dict) and (quantize_information.get('type')
                                     in ['output']):

            quantizer_params = layer.get_config()['quantize_config']['config'][
                'output_quantizers'][0]['quantizer_params']
            method = quantizer_params['method']
            q_min, q_max = self.get_quantizer_q_min_max(quantizer_params)
          elif isinstance(quantize_information,
                          dict) and (quantize_information.get('type')
                                     in ['weight']):
            quantizer_params = layer.get_config()['quantize_config']['config'][
                'weight_quantizers'][0]['quantizer_params']
            method = quantizer_params['method']
            q_min, q_max = self.get_quantizer_q_min_max(quantizer_params)
          layer_quantize_info[name] = self.get_scale_min_max(
              method, quantize_information, q_min, q_max, quantizer_params)
        quantize_info[layer.layer.name] = layer_quantize_info
      elif isinstance(layer, vitis_quantize_layer.VitisQuantize):
        layer_quantize_info = layer.get_quantize_info()
        method = 0
        if layer.get_quantize_info()['type'] in {'input'}:
          method = layer.get_config()['quantizer']['config']['method']
          q_min, q_max = self.get_quantizer_q_min_max(
              layer.get_config()['quantizer']['config'])
        layer_quantize_info = self.get_scale_min_max(
            method, layer_quantize_info, q_min, q_max,
            layer.get_config()['quantizer']['config'])
        quantize_info[layer.name] = layer_quantize_info
    quantize_map = copy.deepcopy(quantize_info)
    return quantize_map


class VitisQuantizer(object):
  """Vitis Quantizer main APIs"""

  def __init__(self,
               float_model,
               model_format=None,
               quantize_strategy='pof2s',
               custom_quantize_strategy=None,
               custom_objects={},
               target=None,
               target_type=None,
               **kwargs):
    """Init VitisQuantizer.

    Args:
      float_model: tfkeras.Model object, the float model to be quantized.
      model_format: string, specified format of the model used in the quantization.
        Available choices are ['func', 'subclass', 'pb']. Default to None.
      quantize_strategy: string, the quantize strategy type . Available choices are
        ['fs', 'pof2s', 'tqt']. Default to 'pof2s'.
      custom_quantize_strategy: string, file path of custom quantize strategy json
        file. Default to None.
      custom_objects: dict, mapping names(strings) to custom classes or functions.
        Default to {}.
      target: string, file path of the arch.json. Default to None.
      target_type: string, type of target, choices are ['json', 'fingerprint', 'name'].

    Return:
      The created VitisQuantizer instance.
    """
    self._float_model = float_model
    self._qat_model = None
    self._qcb_model = None
    self._qcbev_model = None
    self._analyse_model = None
    self._optimized_model = None
    self._layer_metadata = None
    self._candidate_layers = None
    self._specific_layers = {}
    self._check_near_dropout = None

    # Check float model and determine quantizing format
    if float_model is None:
      logger.error('`float_model` cannot be None')

    if model_format is None or \
       model_format not in ['func', 'subclass', 'pb']:
      self._model_format = self._get_model_format(float_model)
    else:
      self._model_format = model_format

    logger.info('Using {} format quantizer'.format(self._model_format))

    if self._model_format == 'func':
      self._quantizer = vitis_quantize_func
    elif self._model_format == 'subclass':
      self._quantizer = vitis_quantize_subclass
    else:
      if isinstance(float_model, keras.Model):
        self._float_model = model_utils.keras_to_graphdef(float_model)
      elif not isinstance(float_model, tf.compat.v1.GraphDef):
        logger.error('`float_model` must be GraphDef for this quantizer')
      from tensorflow_model_optimization.python.core.quantization.keras.vitis import (
              vitis_quantize_pb)
      self._quantizer = vitis_quantize_pb.PBQuantizer(self._float_model)
      return

    # Custom objects
    self.custom_objects = custom_objects
    self._check_custom_objects()
    self.custom_layer_type = [
        l.__class__.__name__
        for l in float_model.layers
        if l.__class__.__name__ in custom_objects
    ]
    self._custom_object_scope = tf.keras.utils.custom_object_scope(
        custom_objects)

    # Built-in quantize strategy
    self._quantize_strategy = vitis_quantize_strategy_factory.get_quantize_strategy(
        quantize_strategy)
    if self.custom_layer_type:
      self._parse_configs({}, {"custom_layer_type": self.custom_layer_type})

    # Custom quantize strategy
    if custom_quantize_strategy:
      if isinstance(custom_quantize_strategy, str):
        custom_quantize_strategy = common_utils.load_json(
            custom_quantize_strategy)
      self._quantize_strategy.update(custom_quantize_strategy)

    # Apply target constraints
    self._target = None
    if target:
      if target_factory is None:
        logger.error(
            'Please install `target_factory` package to quantize with targets.')

      if not target_type:
        if isinstance(target, str):
          if target.endswith('.json'):
            target_type = 'json'
          elif target.startswith('DPU'):
            target_type = 'name'
          else:
            target_type = 'fingerprint'
        else:
          logger.error(
              "'target_type' supports 'json', 'fingerprint' or 'name'. \n{}"
              .format(Usage))

      if target_type == 'json':
        self._target = common_utils.load_json(target)['target']
      elif target_type == 'name':
        self._target = target
      elif target_type == 'fingerprint':
        self._target = target
      else:
        logger.error(
            "'target_type' supports 'json', 'fingerprint' or 'name'. \n{}"
            .format(Usage))

      if target_type in ['json', 'name']:
        if not target_factory.is_registered_target(self._target):
          logger.error(
              '[Quantizer_TF2_Invalid_Target][Invalid Target] {} not in target_factory.'
              .format(self._target))
      elif target_type in ['fingerprint']:
        try:
          target_factory.get_target_by_fingerprint(eval(self._target))
        except:
          logger.error(
              '[Quantizer_TF2_Invalid_Target][Invalid Target] {} not in target_factory.'
              .format(self._target))

    #  if not self._target and type(
    #      self._quantize_strategy
    #  ) == vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy:
    #    logger.warning('Quantizing without specific `target`. \n{}'.format(Usage))

  def _create_qat_model(self, dataset):
    """Create quantize-aware training model."""
    if not self._optimized_model:
      logger.error('Should call `optimize_model()` before `_create_qat_model`.')
    self._qat_model, self._layer_metadata = self._quantizer.create_quantize_model(
        self._optimized_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        mode='QAT',
        target=self._target,
        dataset=dataset)

  def _run_model_with_collector(self, model, dataset, batch_size, steps):
    """Run model with quantize info collector."""
    collector = CollectQuantizeInfoCallback()
    model.predict(
        dataset,
        batch_size=batch_size,
        verbose=1,
        steps=steps,
        callbacks=[collector])
    return collector

  def _create_optimized_model(self):
    """Create optimized model."""
    self._optimized_model, self._layer_metadata = self._quantizer.create_optimize_model(
        self._float_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy)

  def _create_analysed_model(self, dataset):
    """Create analysed model."""
    self._analysed_model, self._layer_metadata = self._quantizer.create_quantize_model(
        self._float_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        mode='ANALYSE',
        target=self._target,
        dataset=dataset)

    logger.info("Start Model Analyse...")
    collector = self._run_model_with_collector(self._analysed_model, dataset,
                                               batch_size, steps)
    logger.info("Model Analyse Done.")
    #  model_info = collector.get_last_quantize_info()
    model_info = collector.get_most_common_quantize_info()
    return model_info

  def _create_refined_model(self, dataset, batch_size, steps, add_shape_info,
                            input_shape):
    """Refine the quantize calibrated model, do post-quantize adjustments and perform
    some finetuning algorithms."""

    logger.info("Start Post-Quant Model Refinement...")
    self._qcbev_model, self._layer_metadata = self._quantizer.create_refine_model(
        quantized_model=self._qcbev_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        optimized_model=self._optimized_model,
        dataset=dataset,
        batch_size=batch_size,
        steps=steps,
        add_shape_info=add_shape_info,
        input_shape=input_shape)
    refined_quantize_info = model_utils.get_quantize_info(self._qcbev_model)
    self._freeze_quantize_info(refined_quantize_info)
    logger.info("Post-Quant Model Refninement Done.")

  def _create_finalized_model(self, dataset):
    """Finalize the refined model, convert model format and save model."""

    logger.info("Start Model Finalization...")
    self._qcbev_model, self._layer_metadata = self._quantizer.create_finalize_model(
        refined_model=self._qcbev_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        dataset=dataset)
    logger.info("Model Finalization Done.")

  def _freeze_quantize_info(self, quantize_info):
    """Freeze the quantize info into the quantize calibrate and evaluate model."""
    if not self._qcb_model:
      logger.error('No qcb_model found.')

    if not self._qcbev_model:
      logger.error('No qcbev_model found.')

    # Freeze the quantize info into the quantized model
    model_utils.set_quantize_info(self._qcb_model, quantize_info)
    model_utils.set_quantize_info(self._qcbev_model, quantize_info)

  def _calibrate_without_loss(self, calib_dataset, calib_batch_size,
                              calib_steps):
    """Calibrate model without loss, only with unlabeled dataset."""
    # Create quantize calibration model
    if not self._optimized_model:
      logger.error(
          'Should call `optimize_model()` before `_calibrate_without_loss`.')
    self._qcb_model, self._layer_metadata = self._quantizer.create_quantize_model(
        self._optimized_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        mode='QCB',
        target=self._target,
        dataset=calib_dataset,
        batch_size=calib_batch_size,
        steps=calib_steps,
        specific_layers=self._specific_layers)

    if calib_dataset is not None:
      logger.info("Start Quantize Calibration...")
      collector = self._run_model_with_collector(self._qcb_model, calib_dataset,
                                                 calib_batch_size, calib_steps)

    #  Create quantize calibration evaluation model
    self._qcbev_model = model_utils.clone_model_with_weights(self._qcb_model)
    model_utils.set_layer_mode(self._qcbev_model, 'QCBEV')

    if type(self._quantize_strategy
           ) == vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy:
      # Freeze the quantize info into the model, now using most_common_quantize_info
      #  last_quantize_info = collector.get_last_quantize_info()
      common_quantize_info = collector.get_most_common_quantize_info()
      self._freeze_quantize_info(common_quantize_info)
    elif type(self._quantize_strategy) in [
        vitis_fs_quantize_strategy.VitisFSQuantizeStrategy,
        vitis_fsx_quantize_strategy.VitisFSXQuantizeStrategy,
        vitis_gpu_quantize_strategy.VitisGPUQuantizeStrategy
    ]:
      # Freeze the quantize info into the model, now using most_common_quantize_info
      #  last_quantize_info = collector.get_last_quantize_info()
      common_quantize_info = collector.get_entropy_percentile_amax(
          self._qcb_model)
      self._freeze_quantize_info(common_quantize_info)

    logger.info("Quantize Calibration Done.")

  def _calibrate_with_loss(self, loss, metrics, calib_dataset, eval_dataset,
                           verbose):
    """Calibrate model with loss and metrics to get better accuracy, need eval_dataset."""
    self._calibrate_without_loss(calib_dataset, calib_batch_size, calib_steps)
    init_quantize_info = model_utils.get_quantize_info(self._qcbev_model)
    quantize_layers = model_utils.get_quantize_layers(self._qcb_model)

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
    model_utils.set_layer_mode(self._qcb_model, 'ANALYSE')
    float_results = _evaluate(self._qcb_model)
    _print_results(float_results, 'float_results')

    # Get simple quantize calibrated results
    init_results = _evaluate(self._qcbev_model)
    _print_results(init_results, 'init_results')

    # Do quantize pos searching
    logger.info("Start Quantize Position Searching...")
    model_utils.set_layer_mode(self._qcb_model, 'QCBEV')
    best_results = init_results
    best_quantize_info = copy.deepcopy(init_quantize_info)
    count = 0
    for name, layer in quantize_layers.items():
      count += 1
      logger.info('({}/{})Processing layer: {}'.format(count,
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
            logger.info('Try change {}.{}: {} -> {}'.format(
                layer_name, quantizer_name, q_pos, q_pos + dt))
          q_info['quant_pos_var'] = q_pos + dt
          model_utils.set_quantize_info(self._qcb_model, tmp_quantize_info)
          q_results = _evaluate(self._qcb_model)
          if q_results['loss'] < new_best_results['loss']:
            new_best_results = q_results
            new_best_quantize_info = copy.deepcopy(tmp_quantize_info)
            _print_results(new_best_results, 'Update Best Results')
        return new_best_quantize_info, new_best_results

      # Quantize Layer
      if isinstance(layer, vitis_quantize_layer.VitisQuantize):
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

    logger.info("Quantize Position Searching Done.")
    _print_results(best_results, 'Final Best Results')

    # Freeze the quantize info into the model, now using last_quantize_info
    self._freeze_quantize_info(best_quantize_info)

  def _parse_configs(self, configs, kwargs):
    """Parse configs from arguments and update the quantize strategy."""
    old_ver_key_map = {
        'fold_bn': 'convert_bn_to_dwconv',
        'replace_relu6': 'convert_relu6_to_relu',
        'convert_tf_op': 'convert_tf_op_to_keras',
        'forced_cle': 'cle_to_relu6',
        'balance_method': 'cle_balance_method',
        'weight_threshold': 'cle_weight_threshold',
        'replace_sigmoid': 'convert_sigmoid_to_hard_sigmoid',
        'replace_hard_sigmoid': 'convert_hard_sigmoid_to_dpu_version',
        'replace_average_pooling2d': 'convert_average_pooling2d_to_dpu_version',
        'replace_leaky_relu': 'convert_leaky_relu_to_dpu_version'
    }

    if not isinstance(configs, dict):
      logger.error('Configs should be a Dict.')
    configs.update(kwargs)
    if configs:
      # Update old version configs
      old_configs = copy.deepcopy(configs)
      for k, v in old_configs.items():
        if k in old_ver_key_map:
          new_key = old_ver_key_map[k]
          configs.pop(k)
          configs[new_key] = old_configs[k]
      self._quantize_strategy.update(configs)

  def _find_unregistered_layer(self):
    custom_layer_type = set()
    model_config = self._float_model.get_config()
    tf_version = tf.__version__.split('.')
    if int(tf_version[0]) == 2 and int(tf_version[1]) >= 6:
      import keras.layers.serialization as serialization
    else:
      import tensorflow.python.keras.layers.serialization as serialization
    serialization.populate_deserializable_objects()
    for layer in self._float_model.layers:
      class_name = layer.__class__.__name__
      cls = tf.keras.utils.get_registered_object(
          class_name, {}, module_objects=serialization.LOCAL.ALL_OBJECTS)
      if cls is None:
        custom_layer_type.add(layer.__class__.__name__)
    return custom_layer_type

  def _check_custom_objects(self):
    custom_layer_type = self._find_unregistered_layer()
    for t in custom_layer_type:
      if t not in self.custom_objects:
        logger.warning("Un-registerd layer type {} is not supplied " \
              "by init args 'custom_objects'".format(t))

  def _get_model_format(self, model):
    if isinstance(model, keras.Model):
      if model_utils.is_subclass_model(model):
        return 'subclass'
      else:
        if model_utils.have_nested_submodel(model):
          logger.warning("This function model {} has nested submodel, " \
                 "try to quantize it by pb format.".format(
                 model.__class__.__name__))
        return 'func'
    elif model_utils.is_tf_graphdef(model):
      return 'pb'
    else:
      logger.error('[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
                   '`float_model` can only be a `tf.keras.Model` instance or '
                   '`tf.compat.v1.GraphDef` instance.')

  # Public Interfaces
  def optimize_model(self, configs={}, **kwargs):
    """Get optimized model.

    Available configs:
       * remove_dropout=True
       * fold_conv_bn=True
       * fold_bn=True
       * replace_relu6=False
       * include_cle=True
       * cle_steps=5
    """
    if self._model_format == 'pb':
      logger.warning('This function does not support pb format model.')
      return None

    # Configure the quantize strategy
    self._parse_configs(configs, kwargs)

    with self._custom_object_scope:
      logger.debug('Optimize Configurations:')
      self._quantize_strategy.get_optimize_pipeline().print_configs()

      self._create_optimized_model()
      if self._check_near_dropout:
        self._optimized_model = model_utils.remove_layer(
            self._optimized_model, 'Dropout')
    return self._optimized_model

  def convert_model(self, target_dtype="float16", **kwargs):
    """Convert model data type to target type.

    args:
       target_dtype: string value, specify target data type, available choices
       ["float32", "float16", "float64"]
    """
    if self._model_format == 'pb':
      self._quantizer.quantize_model(target_dtype=target_dtype, **kwargs)
      return None
    elif self._model_format == 'subclass':
      logger.warning('This function does not support subclass model.')
      return None

    ## need fold BN firstly as BN not support float16
    available_type = ["float32", "float16", "float64", "bfloat16"]
    if target_dtype not in available_type:
      logger.error('Unknown target data type: {}, expecte one of supported ' \
             'data type {}.'.format(target_dtype, available_type))
    self.optimize_model(remove_dropout=False, include_cle=False)

    model_config = self._optimized_model.get_config()
    for l in model_config["layers"]:
      l["config"]["dtype"] = target_dtype
    converted_model = keras.Model.from_config(
        model_config, custom_objects=self.custom_objects)
    weights = self._optimized_model.get_weights()
    weights = [w.astype(target_dtype) for w in weights]
    converted_model.set_weights(weights)
    logger.info("Convert model data type to {}".format(target_dtype))
    return converted_model

  def convert_model_from_candidate(self, target_dtype="float16"):
    """Convert mode__l data type to target type.

    args:
       target_dtype: string value, specify target data type, available choices
       ["float32", "float16", "float64"]
    """
    ## need fold BN firstly as BN not support float16
    available_type = ["float32", "float16", "float64", "bfloat16"]
    if target_dtype not in available_type:
      logger.error('Unknown target data type: {}, expecte one of supported ' \
             'data type {}.'.format(target_dtype, available_type))
    self.optimize_model(remove_dropout=False, include_cle=False)

    model_config = self._optimized_model.get_config()
    for l in model_config["layers"]:
      if l['name'] in self._candidate_layers:
        l["config"]["dtype"] = target_dtype
    converted_model = keras.Model.from_config(
        model_config, custom_objects=self.custom_objects)
    for new_layer, old_layer in zip(converted_model.layers,
                                    self._optimized_model.layers):
      if new_layer.name in self._candidate_layers:
        new_weights = [w.astype(target_dtype) for w in old_layer.get_weights()]
        new_layer.set_weights(new_weights)
      else:
        new_layer.set_weights(old_layer.get_weights())
    logger.info(
        "Convert model data type to {} from in_out layer".format(target_dtype))
    return converted_model

  def set_model_with_layer_config(self, layer_config={}):
    """Convert model layer data type with layer_config.

    args:
       layer_config: dict value, specify layer data type, available choices
       {"A": "float16", "B": "float32", "C": "int16"}
    """
    available_type = [
        "float16", "float32", "float64", "bfloat16", "int8", "int16", "int32"
    ]
    specific_layers = {}
    for layer, layer_datatype in layer_config.items():
      if layer_datatype not in available_type:
        logger.error('Unknown layer {} data type: {}, expecte one of supported ' \
             'data type {}.'.format(layer, layer_datatype, available_type))
      else:
        specific_layers[layer] = layer_datatype
    self.optimize_model()

    model_config = self._optimized_model.get_config()
    for l in model_config["layers"]:
      # the dtype is not int
      if l["config"]['name'] in layer_config.keys() and layer_config[
          l["config"]['name']] not in ["int8", "int16", "int32"]:
        l["config"]["dtype"] = layer_config[l["config"]['name']]
    converted_model = keras.Model.from_config(
        model_config, custom_objects=self.custom_objects)

    for new_layer, old_layer in zip(converted_model.layers,
                                    self._optimized_model.layers):
      old_layer_weights = old_layer.get_weights()
      if old_layer.name in layer_config.keys():
        change_type_weights = [
            w.astype(new_layer.dtype) for w in old_layer_weights
        ]
        new_layer.set_weights(change_type_weights)
        logger.info("Convert model layer {} data type to {}".format(
            new_layer.name, new_layer.dtype))
      else:
        new_layer.set_weights(old_layer_weights)
    self._optimized_model = converted_model
    return specific_layers

  def get_analysed_model(self, dataset):
    """Get analysed model."""
    if self._model_format == 'pb':
      logger.warning('This function does not support pb format model.')
      return None, None

    if not self._analyse_model:
      with self._custom_object_scope:
        model_info = self._create_analysed_model(dataset)
    return self._analysed_model, model_info

  def get_qat_model(self,
                    init_quant=False,
                    calib_dataset=None,
                    calib_batch_size=None,
                    calib_steps=None,
                    configs={},
                    **kwargs):
    """Get quantize-aware training model.

    Available configs:
       * input_bit=8
       * weight_bit=8
       * activation_bit=8
       * remove_dropout=True
       * fold_conv_bn=True
       * fold_bn=True
       * replace_relu6=False
       * include_cle=True
       * cle_steps=5
       * forced_cle=False
       * include_fast_ft=False
       * fast_ft_epochs=10
    """
    if self._model_format == 'pb':
      logger.warning('This function does not support pb format model.')
      return None

    with self._custom_object_scope:
      self._parse_configs(configs, kwargs)
      configs = self._quantize_strategy.get_configs()

      if not self._target and type(
          self._quantize_strategy
      ) == vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy:
        configs['quantize_pipeline_config']['quantize_with_xcompiler'] = False
        logger.info('Quantizing without specific `target`.')

      # Handle user-defined partition
      if not self._candidate_layers:
        input_layers = configs["quantize_registry_config"][
            'user_quantize_config']['input_layers']
        output_layers = configs["quantize_registry_config"][
            'user_quantize_config']['output_layers']
        ignore_layers = configs["quantize_registry_config"][
            'user_quantize_config']['ignore_layers']
        if input_layers or output_layers or ignore_layers:
          input_quantize_config = configs["quantize_registry_config"][
              'input_quantize_config']
          input_quantize_config["input_layers"] = input_layers
          self._quantize_strategy.update(
              {"input_quantize_config": input_quantize_config})
          self._candidate_layers = model_utils.get_candidate_layers(
              self._float_model, input_layers, output_layers, ignore_layers)
          if configs["optimize_pipeline_config"]["remove_dropout"]:
            self._check_near_dropout = model_utils.check_near_dropout(
                self._float_model, ignore_layers)

      self.optimize_model()

      logger.debug('Quantize Pipeline Configurations:')
      self._quantize_strategy.get_quantize_registry().print_configs()
      self._quantize_strategy.get_quantize_pipeline().print_configs()

      logger.info('Start Generation of Quantize-aware Training Model.')
      if not self._qat_model:
        self._create_qat_model(calib_dataset)

      # Do post training quantization to initialize the quantize-aware training model
      if init_quant:
        logger.info('Start Initialization with Quantize Calibration...')

        new_kwargs = {}

        for key, value in kwargs.items():
          new_kwargs[key] = value

        if isinstance(self._quantize_strategy,
                      vitis_tqt_quantize_strategy.VitisTQTQuantizeStrategy):
          new_kwargs['convert_to_pof2s_quantize_strategy'] = False
        elif isinstance(
            self._quantize_strategy,
            vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy):
          new_kwargs['convert_to_fs_quantize_strategy'] = False

        self.quantize_model(
            loss=None,
            metrics=None,
            calib_dataset=calib_dataset,
            calib_batch_size=calib_batch_size,
            calib_steps=calib_steps,
            eval_dataset=None,
            verbose=0,
            add_shape_info=False,
            **new_kwargs)

        init_weights = self._qcbev_model.get_weights()
        self._qat_model.set_weights(init_weights)

        logger.info('Initialization with Quantize Calibration Done.')

      logger.info('Generation of Quantize-aware Training Model Done.')
    return self._qat_model

  def quantize_model(self,
                     loss=None,
                     metrics=None,
                     calib_dataset=None,
                     calib_batch_size=None,
                     calib_steps=None,
                     eval_dataset=None,
                     verbose=0,
                     add_shape_info=False,
                     input_shape=None,
                     configs={},
                     **kwargs):
    """Interface of Post-Training Quantization.

    Args:
      float_model: tf.keras.Model instance, the float model to be quantized.
      configs: dict(string, value), the user config of quantize strategy, will override the
        default built-in quantize strategy. It accecpt all valid configurations listed in
        the quantize strategy json file. Common used configurations are listed below.
        For full list of configurations, see ./quantize_strategy/fs/vitis_fs_quantize_strategy.json
        for reference.
      **kwargs: dict(string, value), same use as configs option, but maybe extended for other
        arguments.

    Commonly used user configs are:
      input_bit: int, the bit_width of all input, default to 8.
      weight_bit: int, the bit_width of all weights, default to 8.
      bias_bit: int: the bit_width of all biases, default to 8.
      activation_bit: int, the bit_width of all activation, default to 8.
      weight_symmetry: bool, whether to do symmetry quantization for all weights, default
        to True.
      bias_symmetry: bool, whether to do symmetry quantization for all biases, default to
        True.
      activation_symmetry: bool, whether to do symmetry quantization for all activation,
        default to True.
      weight_per_channel: bool, whether to do per_channel quantization or per_tensor
        quantization for all weights, default to True.
      bias_per_channel: bool, whether to do per_channel quantization or per_tensor
        quantization for all biases, default to True.
      activation_per_channel: bool, whether to do per_channel quantization or per_tensor
        quantization for all activation, default to True.
      weight_round_mode: int, the round mode of quantization for all weights. 0 for H
        all weights, default to True.
      fold_conv_bn: bool, whether to fold the batchnorm layers into previous Conv2D,
        DepthwiseConv2D, TransposeConv2D and Dense layers.
      convert_bn_to_dwconv: bool, whether to convert the standalone batchnorm layer
        into DepthwiseConv2D layers.
      convert_sigmoid: A bool object, whether to replace the Activation(activation='sigmoid')
        layers into hard sigmoid layers and do quantization. If not, the sigmoid layers
        will be left unquantized and put on CPU.
      convert_relu6_to_relu: bool, whether to replace the Relu6 layers with Relu layers.
      include_cle: bool, whether to do Cross Layer Equalization before quantization.
      cle_steps: int, the iteration steps to do Cross Layer Equalization.
      forced_cle: bool, whether to do forced cle for relu6 layers.
      include_fast_ft: bool, wether to do fast finetuning or not. Fast finetuning
        adjust the weights layer by layer with calibration dataset and may get better
        accuracy for some models. It will take much longer time than normal PTQ
        (still shorter than QAT as calib_dataset is much smaller than train dataset)
        and is disabled by default to save time, and can be turned on to try to improve
        the performance if you see accuracy issues.
      fast_ft_epochs: int, the iteration epochs to do fast finetuning for each layer.
      output_format: string, indicates what format to save the quantized model. Options
        are: '' for skip saving, h5' for saving .h5 file, 'tf' for saving saved_model
        file, 'onnx' for saving .onnx file.
      output_dir: string, indicates the directory to save the quantized model in,
        defaulted to './quantize_results'.

    Return:
      A tf.keras.Model instance, the quantized model.
    """
    if self._model_format == 'pb':
      return self._quantizer.quantize_model(input_shape=input_shape,calib_steps=calib_steps, **kwargs)

    # Configure the quantize strategy
    self._parse_configs(configs, kwargs)
    configs = self._quantize_strategy.get_configs()

    if not self._target and type(
        self._quantize_strategy
    ) == vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy:
      configs['quantize_pipeline_config']['quantize_with_xcompiler'] = False
      logger.info('Quantizing without specific `target`.')

    all_specific_layers = {}
    if "layer_config" in configs["quantize_registry_config"][
        'user_quantize_config']:
      layer_config = configs["quantize_registry_config"][
          'user_quantize_config']["layer_config"]
      if layer_config != {} and self._model_format == 'func':
        all_specific_layers = self.set_model_with_layer_config(layer_config)

    if loss and not eval_dataset:
      logger.error(
          'Need to assign `eval_dataset` for when calling quantize_model(loss=loss_fn).'
      )

    add_shape_info = (bool(self.custom_objects) or add_shape_info)
    if input_shape is not None and isinstance(input_shape, dict):
      if self._model_format == 'subclass':
        logger.info(
            "This subclassed model with input_shape {}".format(input_shape))
      elif len(input_shape) == 1:
        for k, v in input_shape.items():
          if len(v) != len(self._float_model.input_shape):
            logger.error(
                "[Quantizer_TF2_Invalid_Input_Shape][Invalid input shape]"
                "The input_shape {} ndim does not match the model input_shape {} ndim"
                .format(input_shape, self._float_model.input_shape))
      elif len(input_shape) != len(self._float_model.input_shape):
        logger.error(
            "[Quantizer_TF2_Invalid_Input_Shape][Invalid input shape]"
            "The input_shape {} ndim does not match the model input_shape {} ndim"
            .format(input_shape, self._float_model.input_shape))

    if input_shape is not None and (isinstance(input_shape, list) or
                                    isinstance(input_shape, tuple)):
      if self._model_format == 'subclass':
        logger.info(
            "This subclassed model with input_shape {}".format(input_shape))
      elif len(input_shape) != len(self._float_model.input_shape):
        logger.error(
            "[Quantizer_TF2_Invalid_Input_Shape][Invalid input shape]"
            "The input_shape {} ndim does not match the model input_shape {} ndim"
            .format(input_shape, self._float_model.input_shape))
    # Handle user-defined partition
    if not self._candidate_layers or all_specific_layers:
      input_layers = configs["quantize_registry_config"][
          'user_quantize_config']['input_layers']
      output_layers = configs["quantize_registry_config"][
          'user_quantize_config']['output_layers']
      ignore_layers = configs["quantize_registry_config"][
          'user_quantize_config']['ignore_layers']
      for layer, layer_datatype in all_specific_layers.items():
        if layer_datatype not in ["int8", "int16", "int32"]:
          ignore_layers.append(layer)
        else:
          self._specific_layers[layer] = layer_datatype
      if input_layers or output_layers or ignore_layers:
        input_quantize_config = configs["quantize_registry_config"][
            'input_quantize_config']
        input_quantize_config["input_layers"] = input_layers
        self._quantize_strategy.update(
            {"input_quantize_config": input_quantize_config})
        self._candidate_layers = model_utils.get_candidate_layers(
            self._float_model, input_layers, output_layers, ignore_layers)
        if configs["optimize_pipeline_config"]["remove_dropout"]:
          self._check_near_dropout = model_utils.check_near_dropout(
              self._float_model, ignore_layers)
    # Disable tf.logging warnings during quantization
    log_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')
    if "convert_datatype" in configs["quantize_pipeline_config"] and len(
        configs["quantize_pipeline_config"]["convert_datatype"]) != 0:
      if not input_layers:
        input_layers = self._float_model.input_names
      if not output_layers:
        output_layers = self._float_model.output_names
      self._candidate_layers = model_utils.get_candidate_layers(
          self._float_model, input_layers, output_layers, ignore_layers)
      self._optimized_model = self.convert_model_from_candidate(
          target_dtype=configs["quantize_pipeline_config"]["convert_datatype"])
      self._qcbev_model = self._optimized_model

    else:
      if not isinstance(self._quantize_strategy, vitis_bfp_quantize_strategy.VitisBFPQuantizeStrategy) and calib_dataset is None:
        logger.error(
            '[Quantizer_TF2_Invalid_Calib_Dataset][Invalid calibration dataset]'
            'Need to assign `calib_dataset` for when calling quantize_model().')

      with self._custom_object_scope:
        # Optimize model before quantization
        if not self._optimized_model:
          self.optimize_model()

        # Quantize model
        logger.debug('Quantize Pipeline Configurations:')
        self._quantize_strategy.get_quantize_registry().print_configs()
        self._quantize_strategy.get_quantize_pipeline().print_configs()

        if loss:
          self._calibrate_with_loss(loss, metrics, calib_dataset,
                                    calib_batch_size, calib_steps, eval_dataset,
                                    verbose)
        else:
          self._calibrate_without_loss(calib_dataset, calib_batch_size,
                                       calib_steps)

        if logger.debug_enabled():
          model_utils.save_model(self._qcbev_model, 'calibrated_model.h5',
                                 './debug/')
          quantize_info = model_utils.get_quantize_info(self._qcbev_model)
          model_utils.save_quantize_info(quantize_info, './debug/')

        # Refine model
        self._create_refined_model(
            dataset=calib_dataset,
            batch_size=calib_batch_size,
            steps=calib_steps,
            add_shape_info=add_shape_info,
            input_shape=input_shape)

    # Finalize model
    self._create_finalized_model(dataset=calib_dataset)

    logger.info("Quantization Finished.")
    if len(all_specific_layers):
      bfloat16_supported_op_list = self._quantize_strategy._bfloat16_op_list
      for layer in self._qcbev_model.layers:
        if not isinstance(
            layer, vitis_quantize_wrapper.QuantizeWrapper) and not isinstance(
                layer, vitis_quantize_layer.VitisQuantize):
          if layer.__class__.__name__ not in bfloat16_supported_op_list[
              'bfloat16_op_supported'] and layer.dtype == 'bfloat16':
            logger.warning("The model layer: {} dtype: {} is not supported " \
          "by \"keras.models.load_model() API\" .".format(layer, layer.dtype))

    tf.get_logger().setLevel(log_level)
    return self._qcbev_model

  @staticmethod
  def get_deploy_model(model,
                       convert_to_pof2s_quantize_strategy=True,
                       convert_to_fs_quantize_strategy=False,
                       use_fixneuron_quant=0,
                       use_framework_quant=True,
                       output_format='',
                       onnx_opset_version=13,
                       output_dir='./quantize_results/',
                       add_shape_info=False,
                       input_shape=None,
                       dataset=None):
    """Convert the QAT model to the deploy model which is compatible with the compiler
    and meet the DPU hardware constraints. """

    if not isinstance(model, keras.Model):
      logger.warning('This function only supports keras model.')
      return None

    configs = {'output_format': output_format}
    configs['onnx_opset_version'] = onnx_opset_version
    configs['output_dir'] = output_dir

    if model_utils.is_subclass_model(model):
      # Convert quantize strategy
      if convert_to_pof2s_quantize_strategy == True:
        logger.info("Convert pof2s_tqt quantize strategy to pof2s.")
        deploy_model = subclass_utils.convert_quantize_strategy(
            model,
            conversion='tqt_to_pof2s',
            use_fixneuron_quant=use_fixneuron_quant)
      else:
        deploy_model = model

      # Remove dropout
      deploy_model = subclass_utils.remove_subclass_dropout(deploy_model)

      # Convert quantize strategy again if needed
      if convert_to_fs_quantize_strategy == True:
        logger.info("Convert pof2s quantize strategy to fs.")
        deploy_model = subclass_utils.convert_quantize_strategy(
            deploy_model,
            conversion='pof2s_to_fs',
            use_framework_quant=use_framework_quant)

      # Save specified format model
      if output_format != '':
        subclass_utils.save_subclass_model(deploy_model, configs,
                dataset=dataset)

      if add_shape_info:
        logger.warning("Getting shape information from subclassed model " \
                       "is not supported yet.")

      return deploy_model
    deploy_model = model_utils.clone_model_with_weights(model)

    # Fold conv_bn_quantize layers
    deploy_model = model_utils.conv_bn_quantize_fold(deploy_model)
    # Convert quantize strategy
    if convert_to_pof2s_quantize_strategy == True:
      logger.info("Convert pof2s_tqt quantize strategy to pof2s.")
      deploy_model = model_utils.convert_quantize_strategy(
          deploy_model,
          conversion='tqt_to_pof2s',
          use_fixneuron_quant=use_fixneuron_quant)

    # Remove dropout
    deploy_model = model_utils.remove_layer(deploy_model, 'Dropout')

    # Post-quant adjustment
    if convert_to_pof2s_quantize_strategy == True:
      quantize_info = model_utils.get_quantize_info(deploy_model)
      adjusted_quantize_info = model_utils.adjust_quantize_info(
          deploy_model,
          quantize_info,
          adjust_vitis_sigmoid=True,
          adjust_shift_cut=True,
          adjust_shift_bias=True,
          adjust_shift_read=True,
          adjust_shift_write=True,
          adjust_shift_swish=True)
      model_utils.set_quantize_info(deploy_model, adjusted_quantize_info)

    # Convert quantize strategy again if needed
    if convert_to_fs_quantize_strategy == True:
      logger.info("Convert pof2s quantize strategy to fs.")
      deploy_model = model_utils.convert_quantize_strategy(
          deploy_model,
          conversion='pof2s_to_fs',
          use_framework_quant=use_framework_quant)
    # Use FixNeuron as quantizer for pof2s quantize strategy
    elif use_fixneuron_quant == 1:
      logger.info("Use FixNeuron as quantizer for pof2s quantize strategy.")
      deploy_model = model_utils.insert_fix_neuron(deploy_model)

    # Save specified format model
    if output_format != '':
      model_utils.save_func_model(deploy_model, configs)

    if add_shape_info:
      logger.info("Start Getting Shape Information...")
      shape_info = model_utils.get_shape(deploy_model, input_shape=input_shape)
      if logger.debug_enabled():
        model_utils.save_shape_info(shape_info, './debug/')
        model_utils.save_model(deploy_model, 'model_with_shape.h5', './debug/')
      logger.info("Getting Shape Information Done.")

    return deploy_model

  @staticmethod
  def dump_model(model,
                 dataset=None,
                 output_dir='./dump_results',
                 dump_float=False,
                 weights_only=False,
                 **kwargs):
    """Dump golden results of quantized model."""
    if model_utils.is_tf_graphdef(model):
      from tensorflow_model_optimization.python.core.quantization.keras.vitis import (
                            vitis_quantize_pb)
      return vitis_quantize_pb.PBQuantizer(model).dump_model(output_dir=output_dir,
              dump_float=dump_float, **kwargs)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    if not weights_only and dataset is None:
      logger.error('`dataset` is needed to dump with activation.')

    logger.info("Start Dumping...")
    model_utils.dump_model_weights(model, dump_float, output_dir)
    if not weights_only:
      model_utils.dump_model_activations(model, dataset, dump_float, output_dir)

  def dump_quantize_strategy(self,
                             dump_file='quantize_strategy.json',
                             verbose=0):
    """Dump the quantize strategy config of current quantizer, users can modify it
    and update the quantizer by set_quantize_strategy(new_config_file).

    Args:
      dump_file: string, path of the dumped config.
      verbose: int, the verbosity level of the dumped config. Set to 0 to only dump
        the user configs. Set to 1 to also dump the pipeline configs. Set to 2 to
        also dump the detailed layer configs.

    Returns:
      dumped string of quantize strategy configs.
    """
    if self._model_format == 'pb':
      logger.warning('This function does not support pb format model.')
      return None

    qs_configs = copy.deepcopy(self._quantize_strategy._qs_configs)
    if verbose < 2:
      qs_configs['quantize_registry_config'].pop('input_quantize_config')
      qs_configs['quantize_registry_config'].pop('layer_quantize_config')
    if verbose < 1:
      qs_configs.pop('optimize_pipeline_config')
      qs_configs.pop('quantize_pipeline_config')
      qs_configs.pop('refine_pipeline_config')
      qs_configs.pop('finalize_pipeline_config')
    data = common_utils.dump_json(qs_configs, dump_file)
    return data

  def set_quantize_strategy(self,
                            new_quantize_strategy='quantize_strategy.json'):
    """Set the quantize strategy config of current quantizer, users can modify dumped
    config and update the quantizer by set_quantize_strategy(new_config_file).

    Args:
      new_quantize_strategy: string or dict, path of the new quantize strategyp json
        file or dict of new quantize strategy configs.
    """
    if self._model_format == 'pb':
      logger.warning('This function does not support pb format model.')
      return None

    if isinstance(new_quantize_strategy, str):
      new_quantize_strategy = common_utils.load_json(new_quantize_strategy)
    elif not isinstance(new_quantize_strategy, dict):
      logger.error(
          'new_quantize_strategy should be filepath or dict, but found {}'
          .format(type(new_quantize_strategy)))

    self._quantize_strategy.update(new_quantize_strategy)
