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
import collections

import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_config as quantize_config_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_fast_finetune
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_bias_correction
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_pooling
from tensorflow_model_optimization.python.core.quantization.keras.vitis.eight_bit import vitis_8bit_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.eight_bit_fs import vitis_8bit_fs_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_quantize_strategies

logger = common_utils.VAILogger
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
      'QuantizeAwareActivation':
          vitis_quantize_aware_activation.QuantizeAwareActivation,
      'NoQuantizeActivation':
          vitis_quantize_aware_activation.NoQuantizeActivation,
      'QuantizeWrapper':
          vitis_quantize_wrapper.QuantizeWrapper,
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

  def __init__(self,
               float_model,
               quantize_strategy='8bit',
               custom_quantize_strategy=None,
               custom_objects={}):
    """Init VitisQuantizer."""
    self._float_model = float_model
    self._qat_model = None
    self._qcb_model = None
    self._qcbev_model = None
    self._analyse_model = None
    self._optimized_model = None
    self._candidate_layers = None
    self._layer_metadata = None

    # Custom objects
    self._custom_object_scope = tf.keras.utils.custom_object_scope(
        custom_objects)

    # Built-in quantize strategy
    self._quantize_strategy = vitis_quantize_strategies.get(quantize_strategy)

    # Custom quantize strategy
    if custom_quantize_strategy:
      if isinstance(custom_quantize_strategy, str):
        custom_quantize_strategy = common_utils.load_json(
            custom_quantize_strategy)
      self._quantize_strategy.update(custom_quantize_strategy)

  def _create_qat_model(self):
    """Create quantize-aware training model."""
    if not self._optimized_model:
      logger.error('Should call `optimize_model()` before `_create_qat_model`.')
    self._qat_model, self._layer_metadata = create_quantize_model(
        self._optimized_model,
        candidate_layers=self._candidate_layers,
        layer_metadata=self._layer_metadata,
        quantize_strategy=self._quantize_strategy,
        mode='QAT')

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
        mode='ANALYSE')

    logger.info("Start Model Analyse...")
    collector = self._run_model_with_collector(self._analysed_model, dataset,
                                               batch_size, steps)
    logger.info("Model Analyse Done.")
    #  model_info = collector.get_last_quantize_info()
    model_info = collector.get_most_common_quantize_info()
    return model_info

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
        mode='QCB')

    logger.info("Start Quantize Calibration...")
    collector = self._run_model_with_collector(self._qcb_model, calib_dataset,
                                               calib_batch_size, calib_steps)

    #  Create quantize calibration evaluation model
    self._qcbev_model = model_utils.clone_model_with_weights(self._qcb_model)
    model_utils.set_layer_mode(self._qcbev_model, 'QCBEV')

    if type(self._quantize_strategy
           ) == vitis_8bit_quantize_strategy.Vitis8BitQuantizeStrategy:
      # Freeze the quantize info into the model, now using most_common_quantize_info
      #  last_quantize_info = collector.get_last_quantize_info()
      common_quantize_info = collector.get_most_common_quantize_info()
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
    if not isinstance(configs, dict):
      logger.error('Configs should be a Dict.')
    configs = {}
    configs.update(kwargs)
    if configs:
      self._quantize_strategy.update(configs)

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
    return self._optimized_model

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
            verbose=0)
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
                     configs={},
                     **kwargs):
    """Interface of Post-Training Quantize.
    
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
       * include_bias_corr=True
    """
    if calib_dataset is None:
      logger.error(
          'Need to assign `calib_dataset` for when calling quantize_model().')

    if loss and not eval_dataset:
      logger.error(
          'Need to assign `eval_dataset` for when calling quantize_model(loss=loss_fn).'
      )

    # Configure the quantize strategy
    self._parse_configs(configs, kwargs)
    configs = self._quantize_strategy.get_configs()
    # Disable tf.logging warnings during quantization
    log_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')

    with self._custom_object_scope:
      # Optimize model before quantization
      if not self._optimized_model:
        self.optimize_model()

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

        # Post-quantize adjustment (Only for 8bit)
        if type(self._quantize_strategy
               ) == vitis_8bit_quantize_strategy.Vitis8BitQuantizeStrategy:
          logger.info("Start Post-Quantize Adjustment...")
          quantize_info = model_utils.get_quantize_info(self._qcbev_model)
          adjust_sc = configs['quantize_pipeline_config']['adjust_shift_cut']
          adjust_sb = configs['quantize_pipeline_config']['adjust_shift_bias']
          adjusted_quantize_info = model_utils.post_quant_adjust(
              self._qcbev_model, quantize_info, adjust_sc, adjust_sb)
          self._freeze_quantize_info(adjusted_quantize_info)
          logger.info("Post-Quantize Adjustment Done.")

        if logger.debug_enabled():
          model_utils.save_model(self._qcbev_model, 'calibrated_model.h5',
                                 './debug/')

        # Fast finetune
        include_fast_ft = configs['quantize_pipeline_config']['include_fast_ft']
        fast_ft_epochs = configs['quantize_pipeline_config']['fast_ft_epochs']
        if include_fast_ft:
          logger.info("Start Fast Finetuning...")
          vitis_fast_finetune.fast_finetune(self._qcbev_model,
                                            self._optimized_model,
                                            calib_dataset, calib_batch_size,
                                            calib_steps, fast_ft_epochs)
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

        if type(self._quantize_strategy
               ) == vitis_8bit_quantize_strategy.Vitis8BitQuantizeStrategy:
          if logger.debug_enabled():
            quantize_info = model_utils.get_quantize_info(self._qcbev_model)
            model_utils.save_quantize_info(quantize_info, './debug/')

        logger.info("Quantization Finished.")

    tf.get_logger().setLevel(log_level)
    return self._qcbev_model

  @staticmethod
  def get_deploy_model(model):
    """Convert the QAT model to the deploy model which is compatible with the compiler
    and meet the DPU hardware constraints. """
    deploy_model = model_utils.clone_model_with_weights(model)

    # Fold conv_bn_quantize layers
    deploy_model = model_utils.conv_bn_quantize_fold(deploy_model)

    # Convert quantize strategy
    deploy_model = model_utils.convert_quantize_strategy(
        deploy_model, conversion='8bit_tqt_to_8bit')

    # Remove dropout
    deploy_model = model_utils.remove_layer(deploy_model, 'Dropout')

    # Post-quant adjustment
    quantize_info = model_utils.get_quantize_info(deploy_model)
    adjusted_quantize_info = model_utils.post_quant_adjust(
        deploy_model,
        quantize_info,
        adjust_shift_cut=True,
        adjust_shift_bias=True)
    model_utils.set_quantize_info(deploy_model, adjusted_quantize_info)
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


def create_optimize_model(model, candidate_layers, layer_metadata,
                          quantize_strategy):
  """Optimize a `tf.keras` model before quantization, such as bn folding, 
  activation folding."""
  optimize_pipeline = quantize_strategy.get_optimize_pipeline()
  optimized_model, layer_metadata = optimize_pipeline.apply(
      model, candidate_layers, layer_metadata)
  return optimized_model, layer_metadata


def create_quantize_model(to_quantize, candidate_layers, layer_metadata,
                          quantize_strategy, mode):
  """Quantize a `tf.keras` model with the default quantization implementation.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    to_quantize: tf.keras model to be quantized. It can have pre-trained
      weights.
    quantize_strategy: QuantizeStrategy constaining the configurations.

  Returns:
    Returns a new `tf.keras` model prepared for quantization.
  """
  if to_quantize is None:
    logger.error('`to_quantize` cannot be None')

  if not isinstance(to_quantize, keras.Model):
    logger.error('`to_quantize` can only be a `tf.keras.Model` instance. '
                 'You passed an instance of type: {input}.'.format(
                     input=to_quantize.__class__.__name__))

  if not isinstance(to_quantize, keras.Sequential) \
      and not to_quantize._is_graph_network:  # pylint: disable=protected-access
    logger.error('`to_quantize` can only either be a tf.keras Sequential or '
                 'Functional model.')

  AVAILABLE_MODES = ['QCB', 'QAT', 'ANALYSE', 'QCBEV']
  if mode not in AVAILABLE_MODES:
    logger.error('Mode `{}` is not valid, available modes are:{}.'.format(
        mode, AVAILABLE_MODES))

  return quantize_apply(to_quantize, candidate_layers, layer_metadata,
                        quantize_strategy, mode)


def quantize_apply(model, candidate_layers, layer_metadata, quantize_strategy,
                   mode):
  """Quantize a `tf.keras` model that has been annotated for quantization.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

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
    logger.error('`model` cannot be None')

  if not isinstance(model, keras.Model):
    logger.error('`model` can only be a `tf.keras.Model` instance.'
                 'You passed an instance of type: {input}.'.format(
                     input=model.__class__.__name__))

  if not isinstance(model, keras.Sequential) \
      and not model._is_graph_network:  # pylint: disable=protected-access
    logger.error('Only tf.keras Sequential or Functional models are supported.')

  if not model.built:
    logger.error('`model` must be a built model. '
                 'been built yet. Please call `model.build(input_shape)` '
                 'before quantizing your model.')

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
  quantized_model, layer_metadata = quantize_pipeline.apply(
      model_copy, candidate_layers, layer_metadata,
      quantize_strategy.get_quantize_registry(), mode)

  return quantized_model, layer_metadata
