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
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common.entropy_percentile import calibrator_numpy

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

  def get_entropy_percentile_amax(self, model):
    quantize_info = collections.OrderedDict()
    progress_total = len(model.layers)
    progbar = keras.utils.Progbar(progress_total)
    for progress, layer in enumerate(model.layers):
      progbar.update(progress + 1)
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        layer_quantize_info = layer.get_quantize_info()
        for name, activation in layer_quantize_info.items():
          method = 0
          percentile = 99.9
          if isinstance(activation, dict) and (activation.get('type') in [
              'post_activation', 'pre_activation'
          ]):
            method = layer.get_config()['quantize_config']['config'][
                'activation_quantizers'][0]['quantizer_params']['method']
            percentile = layer.get_config(
            )['quantize_config']['config']['activation_quantizers'][0][
                'quantizer_params']['method_percentile']
          elif isinstance(activation, dict) and (activation.get('type')
                                                 in ['output']):
            method = layer.get_config()['quantize_config']['config'][
                'output_quantizers'][0]['quantizer_params']['method']
            percentile = layer.get_config()['quantize_config']['config'][
                'output_quantizers'][0]['quantizer_params']['method_percentile']
          batch_max = None
          if vitis_quantize_ops.QuantizeMethod.MIN_KL == vitis_quantize_ops.QuantizeMethod(
              method):
            batch_max = calibrator_numpy.numpy_kl_div(
                activation['info']['calib_hist'],
                activation['info']['calib_bin_edges'])
            batch_min = -batch_max
            activation['info']['min_var'] = batch_min
            activation['info']['max_var'] = batch_max
          elif vitis_quantize_ops.QuantizeMethod.PERCENTILE == vitis_quantize_ops.QuantizeMethod(
              method):
            batch_max = calibrator_numpy.numpy_percentile(
                percentile, activation['info']['calib_hist'],
                activation['info']['calib_bin_edges'])
            batch_min = -batch_max
            activation['info']['min_var'] = batch_min
            activation['info']['max_var'] = batch_max
          layer_quantize_info[name] = activation
        quantize_info[layer.layer.name] = layer_quantize_info
      elif isinstance(layer, vitis_quantize_layer.VitisQuantize):
        layer_quantize_info = layer.get_quantize_info()
        if layer.get_quantize_info()['type'] in {'input'}:
          batch_max = None
          method = layer.get_config()['quantizer']['config']['method']
          if vitis_quantize_ops.QuantizeMethod.MIN_KL == vitis_quantize_ops.QuantizeMethod(
              method):
            batch_max = calibrator_numpy.numpy_kl_div(
                layer_quantize_info['info']['calib_hist'],
                layer_quantize_info['info']['calib_bin_edges'])
            batch_min = -batch_max
            layer_quantize_info['info']['min_var'] = batch_min
            layer_quantize_info['info']['max_var'] = batch_max
          elif vitis_quantize_ops.QuantizeMethod.PERCENTILE == vitis_quantize_ops.QuantizeMethod(
              method):
            percentile = layer.get_config(
            )['quantizer']['config']['method_percentile']
            batch_max = calibrator_numpy.numpy_percentile(
                percentile, layer_quantize_info['info']['calib_hist'],
                layer_quantize_info['info']['calib_bin_edges'])
            batch_min = -batch_max
            layer_quantize_info['info']['min_var'] = batch_min
            layer_quantize_info['info']['max_var'] = batch_max
        quantize_info[layer.name] = layer_quantize_info
    quantize_map = copy.deepcopy(quantize_info)
    return quantize_map


class VitisQuantizer(object):
  """Vitis Quantizer main APIs"""

  def __init__(self,
               float_model,
               quantize_strategy='pof2s',
               custom_quantize_strategy=None,
               custom_objects={},
               target=None,
               target_type=None):
    """Init VitisQuantizer.

    Args:
      float_model: tfkeras.Model object, the float model to be quantized.
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
    self._check_near_dropout = None

    # Check float model
    if float_model is None:
      logger.error('`float_model` cannot be None')

    if not isinstance(float_model, keras.Model):
      logger.error('[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
                   '`float_model` can only be a `tf.keras.Model` instance. '
                   'You passed an instance of type: {input}.'.format(
                       input=float_model.__class__.__name__))

    if not isinstance(float_model, keras.Sequential) \
        and not float_model._is_graph_network:  # pylint: disable=protected-access
      logger.error(
          '[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
          '`float_model` can only either be a tf.keras Sequential or '
          'Functional model. Subclassing model is not supported now, '
          'please convert it to Functional model and try again. See '
          'https://www.tensorflow.org/guide/keras/functional for more details.')

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

  def _create_qat_model(self):
    """Create quantize-aware training model."""
    if not self._optimized_model:
      logger.error('Should call `optimize_model()` before `_create_qat_model`.')
    self._qat_model, self._layer_metadata = create_quantize_model(
        self._optimized_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        mode='QAT',
        target=self._target)

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
    self._optimized_model, self._layer_metadata = create_optimize_model(
        self._float_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy)

  def _create_analysed_model(self, dataset):
    """Create analysed model."""
    self._analysed_model, self._layer_metadata = create_quantize_model(
        self._float_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        mode='ANALYSE',
        target=self._target)

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
    self._qcbev_model, self._layer_metadata = create_refine_model(
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

  def _create_finalized_model(self):
    """Finalize the refined model, convert model format and save model."""

    logger.info("Start Model Finalization...")
    self._qcbev_model, self._layer_metadata = create_finalize_model(
        refined_model=self._qcbev_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy)
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
    self._qcb_model, self._layer_metadata = create_quantize_model(
        self._optimized_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        mode='QCB',
        target=self._target)

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
        vitis_fsx_quantize_strategy.VitisFSXQuantizeStrategy
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

    quantize_layers = {}
    for layer in self._qcb_model.layers:
      if model_utils.is_quantize_layer(layer):
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

  def convert_model(self, target_dtype="float32"):
    """Convert model data type to target type.

    args:
       target_dtype: string value, specify target data type, available choices
       ["float32", "float16", "float64"]
    """
    ## need fold BN firstly as BN not support float16
    available_type = ["float32", "float16", "float64"]
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

  def get_analysed_model(self, dataset):
    """Get analysed model."""
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
        self._create_qat_model()

      # Do post training quantization to initialize the quantize-aware training model
      if init_quant:
        logger.info('Start Initialization with Quantize Calibration...')
        self.quantize_model(
            loss=None,
            metrics=None,
            calib_dataset=calib_dataset,
            calib_batch_size=calib_batch_size,
            calib_steps=calib_steps,
            eval_dataset=None,
            verbose=0,
            add_shape_info=False,
            convert_to_pof2s_quantize_strategy=False)
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
    # Configure the quantize strategy
    self._parse_configs(configs, kwargs)
    configs = self._quantize_strategy.get_configs()

    if not self._target and type(
        self._quantize_strategy
    ) == vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy:
      configs['quantize_pipeline_config']['quantize_with_xcompiler'] = False
      logger.info('Quantizing without specific `target`.')

    if "convert_datatype" in configs["quantize_pipeline_config"] and configs[
        "quantize_pipeline_config"]["convert_datatype"]:
      return self.convert_model(
          target_dtype=configs["quantize_pipeline_config"]["convert_datatype"])

    if calib_dataset is None:
      logger.error(
          '[Quantizer_TF2_Invalid_Calib_Dataset][Invalid calibration dataset]'
          'Need to assign `calib_dataset` for when calling quantize_model().')

    if loss and not eval_dataset:
      logger.error(
          'Need to assign `eval_dataset` for when calling quantize_model(loss=loss_fn).'
      )

    add_shape_info = (bool(self.custom_objects) or add_shape_info)
    if input_shape is not None and isinstance(input_shape, dict):
      if len(input_shape) == 1:
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
        
    if input_shape is not None and (isinstance(input_shape, list) or isinstance(input_shape, tuple)):
      if len(input_shape) != len(self._float_model.input_shape):
        logger.error(
            "[Quantizer_TF2_Invalid_Input_Shape][Invalid input shape]"
            "The input_shape {} ndim does not match the model input_shape {} ndim"
            .format(input_shape, self._float_model.input_shape))

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

    # Disable tf.logging warnings during quantization
    log_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')

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
      self._create_finalized_model()

    logger.info("Quantization Finished.")

    tf.get_logger().setLevel(log_level)
    return self._qcbev_model

  @staticmethod
  def get_deploy_model(model, add_shape_info=False, input_shape=None):
    """Convert the QAT model to the deploy model which is compatible with the compiler
    and meet the DPU hardware constraints. """
    deploy_model = model_utils.clone_model_with_weights(model)

    # Fold conv_bn_quantize layers
    deploy_model = model_utils.conv_bn_quantize_fold(deploy_model)

    # Convert quantize strategy
    deploy_model = model_utils.convert_quantize_strategy(
        deploy_model, conversion='tqt_to_pof2s')

    # Remove dropout
    deploy_model = model_utils.remove_layer(deploy_model, 'Dropout')

    # Post-quant adjustment
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
                 weights_only=False):
    """Dump golden results of quantized model."""
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
    if isinstance(new_quantize_strategy, str):
      new_quantize_strategy = common_utils.load_json(new_quantize_strategy)
    elif not isinstance(new_quantize_strategy, dict):
      logger.error(
          'new_quantize_strategy should be filepath or dict, but found {}'
          .format(type(new_quantize_strategy)))

    self._quantize_strategy.update(new_quantize_strategy)


def create_optimize_model(model, candidate_layers, layer_metadata,
                          quantize_strategy):
  """Optimize a `tf.keras` model before quantization, such as bn folding,
  activation folding.

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

  if not isinstance(model, keras.Sequential) \
      and not model._is_graph_network:  # pylint: disable=protected-access
    logger.error(
        '[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
        '`model` can only either be a tf.keras Sequential or '
        'Functional model. Subclassing model is not supported now, '
        'please convert it to Functional model and try again. See '
        'https://www.tensorflow.org/guide/keras/functional for more details.')

  optimize_pipeline = quantize_strategy.get_optimize_pipeline()
  optimized_model, layer_metadata = optimize_pipeline.apply(
      model, candidate_layers, layer_metadata)
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

  if not isinstance(quantized_model, keras.Sequential) \
      and not quantized_model._is_graph_network:  # pylint: disable=protected-access
    logger.error(
        '[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
        '`model` can only either be a tf.keras Sequential or '
        'Functional model. Subclassing model is not supported now, '
        'please convert it to Functional model and try again. See '
        'https://www.tensorflow.org/guide/keras/functional for more details.')

  refine_pipeline = quantize_strategy.get_refine_pipeline()
  refined_model, layer_metadata = refine_pipeline.apply(
      quantized_model, candidate_layers, layer_metadata, optimized_model,
      dataset, batch_size, steps, add_shape_info, input_shape)
  return refined_model, layer_metadata


def create_finalize_model(refined_model, candidate_layers, layer_metadata,
                          quantize_strategy):
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

  if not isinstance(refined_model, keras.Sequential) \
      and not refined_model._is_graph_network:  # pylint: disable=protected-access
    logger.error(
        '[Quantizer_TF2_Unsupported_Model][Unsupported model type] '
        '`refined_model` can only either be a tf.keras Sequential or '
        'Functional model. Subclassing model is not supported now, '
        'please convert it to Functional model and try again. See '
        'https://www.tensorflow.org/guide/keras/functional for more details.')

  finalize_pipeline = quantize_strategy.get_finalize_pipeline()
  finalized_model, layer_metadata = finalize_pipeline.apply(
      refined_model, candidate_layers, layer_metadata)
  return finalized_model, layer_metadata


def create_quantize_model(model, candidate_layers, layer_metadata,
                          quantize_strategy, mode, target):
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

  # 1. Create a copy of the model with the same weights. This ensures
  # modifications don't affect the original model, or its weights.
  try:
    model_copy = model_utils.clone_model_with_weights(model)
  except ValueError:
    logger.error(
        'Unable to clone model. This generally happens if you used custom Keras layers or objects '
        'in your model. Please wrap the functions in the custom_object_scope() with all the custom layers.'
    )

  # 2. Run the pipeline of quantize transforms.
  # Quantizable layers will be wrapped with QuantizeWrapper while others ramain float.
  quantize_pipeline = quantize_strategy.get_quantize_pipeline()
  if isinstance(
      quantize_pipeline,
      vitis_pof2s_transforms_pipeline.VitisPof2SQuantizeTransformsPipeline):
    quantized_model, layer_metadata = quantize_pipeline.apply(
        model_copy, candidate_layers, layer_metadata,
        quantize_strategy.get_quantize_registry(), mode, target)
  else:
    quantized_model, layer_metadata = quantize_pipeline.apply(
        model_copy, candidate_layers, layer_metadata,
        quantize_strategy.get_quantize_registry(), mode)

  return quantized_model, layer_metadata
