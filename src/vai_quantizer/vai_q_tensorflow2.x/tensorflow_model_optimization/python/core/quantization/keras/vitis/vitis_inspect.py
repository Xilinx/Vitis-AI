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
"""Inspector API functions for tf.keras models."""

from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import textwrap
import pprint
import copy

try:
  import target_factory
except:
  target_factory = None

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy import vitis_quantize_strategy_factory
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vitis_quantize import create_optimize_model
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.vai_utf import vai_utf_parser
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import visualize_utils

logger = common_utils.VAILogger
keras = tf.keras

Usage = (
    '\nVitisInspector is an experimental new feature introduced since Vitis-AI 3.0 '
    'to inspect a float model and show partition results for given DPU target architecture, together '
    'with some hints why layers are not mapped to DPU. Without `target` we can only show some '
    'general, target-independent inspect results. Please assign `target` to get more detailed '
    'inspect results for it. This feature in only available for default pof2s quantize strategy '
    'due to DPU limitation. See user guide for more information. This feature is experimental, please '
    'contact us if you meet with bugs or problems.'
    '\n'
    '\nUsage:'
    '\n[1] Inspect a model:'
    '\n    VitisInspector(target=my_target).inspect_model(my_model, verbose=0)'
    '\n[2] Inspect a model and plot results:'
    '\n    VitisInspector(target=my_target).inspect_model(my_model, plot=True, plot_file="results.png")'
    '\n[3] Inspect a model and save verbose results to text:'
    '\n    VitisInspector(target=my_target).inspect_model(my_model, dump_results=True, dump_results_file="results.txt")'
    '\n'
    '\n[Note] `my_target` is the target DPU to deploy this model, it can be a name(e.g. "DPUCZDX8G_ISA1_B4096"), '
    'a json(e.g. "./U50/arch.json") or a fingerprint.\n')


class InspectResult(object):
  """Container class of inspect result."""

  def __init__(self, device='CPU', origin_layers=[], notes=None):
    self.device = device
    self.origin_layers = origin_layers
    if notes is None:
      self.notes = {}
    for name in self.origin_layers:
      self.notes[name] = []
    self.id = None

  def add_notes(self, new_notes, layer_name=None):
    if layer_name is None:
      layer_name = self.origin_layers[0]

    if layer_name not in self.notes:
      self.notes[layer_name] = []

    if isinstance(new_notes, list):
      self.notes[layer_name].extend(new_notes)
    else:
      self.notes[layer_name].append(new_notes)

  def get_notes(self, layer_name=None):
    if layer_name is None:
      layer_name = self.origin_layers[0]

    if layer_name not in self.notes:
      logger.warning('{} not found in notes.'.format(layer_name))
      return []

    return self.notes[layer_name]

  def clear_notes(self):
    for name in self.origin_layers:
      self.notes[name] = []

  def add_device(self, new_device):
    if self.device == '':
      self.device = new_device

    if self.device != new_device:
      self.device = '+'.join([self.device, new_device])

  def merge_origin_layers(self, other):
    self.origin_layers.extend(other.origin_layers)


class VitisInspector(object):
  """Vitis Inspector main APIs."""

  def __init__(self,
               quantize_strategy='pof2s',
               custom_quantize_strategy=None,
               custom_objects={},
               target=None,
               target_type=None):
    """Init VitisInspector.

    Args:
      quantize_strategy: string, the quantize strategy type . Available choices are
        ['fs', 'pof2s', 'tqt']. Default to 'pof2s', only 'pof2s' is supported now, others
        will be supported in future release.
      custom_quantize_strategy: string, file path of custom quantize strategy json
        file. Default to None.
      custom_objects: dict, mapping names(strings) to custom classes or functions.
        Default to {}.
      target: string, file path of the arch.json. Default to None.
      target_type: string, type of target, choices are ['json', 'fingerprint', 'name'].

    Return:
      The created VitisInspector instance.
    """
    # Custom objects
    self.custom_objects = custom_objects
    self._custom_object_scope = tf.keras.utils.custom_object_scope(
        custom_objects)

    # Built-in quantize strategy
    if quantize_strategy not in ['pof2s']:
      logger.error(
          'Unsupported `quantize_strategy`, VitisInspector only support `pof2s` '
          'quantize_strategy now, other `quantize_strategy` will be supported in future.'
      )
    self._quantize_strategy = vitis_quantize_strategy_factory.get_quantize_strategy(
        quantize_strategy)

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
              "'target_type' supports 'json', 'fingerprint' or 'name'. \n{}".format(Usage))

      if target_type == 'json':
        self._target = common_utils.load_json(target)['target']
      elif target_type == 'name':
        self._target = target
      elif target_type == 'fingerprint':
        self._target = target
      else:
        logger.error(
            "'target_type' supports 'json', 'fingerprint' or 'name'. \n{}".format(Usage))


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

  def _init_inspect_results(self, model):
    layer_metadata = {}
    for layer in model.layers:
      if isinstance(layer, keras.layers.InputLayer):
        ins_res = InspectResult(device='INPUT', origin_layers=[layer.name])
      else:
        ins_res = InspectResult(device='CPU', origin_layers=[layer.name])
      layer_metadata[layer.name] = {'inspect_result': ins_res}
    return layer_metadata

  def _extract_inspect_results(self, float_model, inspect_model,
                               layer_metadata):
    """Extract inspect results."""
    ori_to_final = {}
    for layer in float_model.layers:
      ori_to_final[layer.name] = []

    for layer in inspect_model.layers:
      if layer.name not in layer_metadata:
        logger.error('Cannot find {}\'s metadata.'.format(layer.name))

      ins_res = layer_metadata[layer.name].get('inspect_result', None)
      if ins_res is None:
        logger.error('Cannot find {}\'s inspect result.'.format(layer.name))

      for ori_layer in ins_res.origin_layers:
        ori_to_final[ori_layer].append(layer.name)
        logger.debug('ori_layer: {} -> final_layer: {}'.format(
            ori_layer, layer.name))

    res = {}
    for i, layer in enumerate(float_model.layers):
      res[layer] = InspectResult(device='', origin_layers=[layer.name])
      res[layer].id = i

    for layer in float_model.layers:
      final_layers = ori_to_final.get(layer.name)
      if not final_layers:
        logger.error('Cannot find layer {}\'s final layer.'.format(layer.name))

      for final_layer in final_layers:
        if final_layer is None:
          logger.error('Cannot find {}\'s final layer.'.format(layer.name))

        ins_res = layer_metadata[final_layer].get('inspect_result', None)
        if ins_res is None:
          logger.error('Cannot find {}\'s inspect result.'.format(layer.name))

        logger.debug('{} {} {}'.format(layer.name, ins_res.device,
                                       ins_res.origin_layers))
        res[layer].add_device(ins_res.device)
        res[layer].add_notes(ins_res.get_notes(layer.name))
    return res

  def _print_inspect_results(self,
                             float_model,
                             inspect_results,
                             verbose=0,
                             io='stio'):
    do_print = io == 'stio'
    line_length = 120
    col_ratio = [0.1, 0.2, 0.2, 0.1, 0.4]
    col_width = [int(i * line_length) for i in col_ratio]
    line_format = ''.join(['{:<%d}'] * len(col_width))
    line_format = line_format % (tuple(col_width))

    def _wrap(s, length):
      if s == '':
        return ['']
      return textwrap.wrap(s, length)

    def _print_row(fields, col_width):
      assert len(fields) == len(col_width)
      for i, f in enumerate(fields):
        fields[i] = _wrap(f, col_width[i])

      max_cnt = max([len(s) for s in fields])
      for f in fields:
        f.extend([''] * (max_cnt - 1))

      cnt = 0
      while cnt < max_cnt:
        to_print = [f[cnt] for f in fields]
        line = line_format.format(*to_print)
        lines.append(line)
        cnt += 1

    def _get_layer_type(layer):
      layer_type = layer.__class__.__name__
      activation = layer.get_config().get('activation')
      if activation:
        layer_type += '<{}>'.format(activation)
      return layer_type

    if do_print:
      logger.info('Inspect Results:')

    lines = []
    total_num = len(float_model.layers)

    # Model info
    lines.append('[MODEL INFO]:')
    lines.append('_' * line_length)
    lines.append('Model Name: {}'.format(float_model.name))
    dpu_target_name = self._target if self._target else None
    lines.append('_' * line_length)

    # Layer list
    to_print = ['ID', 'Name', 'Type', 'Device', 'Notes']
    _print_row(to_print, col_width)
    lines.append('=' * line_length)
    for layer, ins_res in inspect_results.items():
      layer_id = ins_res.id
      layer_type = _get_layer_type(layer)
      device = ins_res.device
      notes = ''
      if device != 'DPU' or verbose >= 1:
        notes = '; '.join([s.strip('.') for s in ins_res.get_notes()])
      to_print = [
          '{}/{}'.format(layer_id, total_num - 1), layer.name, layer_type,
          device, notes
      ]
      _print_row(to_print, col_width)
      lines.append('-' * line_length)
    lines.append('=' * line_length)

    # Inspect Summary
    lines.append('[SUMMARY INFO]:')
    type_num = {}
    device_num = {}
    has_unsupported = False
    for layer, ins_res in inspect_results.items():
      layer_type = _get_layer_type(layer)
      if layer_type not in type_num:
        type_num[layer_type] = 1
      else:
        type_num[layer_type] += 1
      device = ins_res.device
      if device not in device_num:
        device_num[device] = 1
      else:
        device_num[device] += 1
      if device not in ['INPUT', 'DPU']:
        has_unsupported = True
    lines.append('- [Target Name]: {}'.format(dpu_target_name))
    lines.append('- [Total Layers]: {}'.format(total_num))
    detail_layers = '- [Layer Types]: '
    for tp, num in type_num.items():
      detail_layers += '{}({}) '.format(tp, num)
    lines.append(detail_layers)

    detail_partition = '- [Partition Results]: '
    for dv, num in device_num.items():
      detail_partition += '{}({}) '.format(dv, num)
    lines.append(detail_partition)

    if not has_unsupported:
      lines.append(
          '\n  All layers are supported and successfully mapped to DPU.')
    lines.append('=' * line_length)

    # Notes summary
    lines.append('[NOTES INFO]:')
    for layer, ins_res in inspect_results.items():
      if ins_res.get_notes():
        layer_type = _get_layer_type(layer)
        device = ins_res.device
        if device != 'DPU' or verbose >= 1:
          lines.append('- [{}/{}] Layer {} (Type:{}, Device:{}):'.format(
              ins_res.id, total_num - 1, layer.name, layer_type, device))
          for i, note in enumerate(ins_res.get_notes()):
            note = note.strip('.')
            lines.append('    * {}'.format(note))
    if not has_unsupported:
      lines.append(
          '\n  All layers are supported and successfully mapped to DPU.')
    lines.append('=' * line_length)

    if do_print:
      print('\n'.join(lines))
    return lines

  def _dump_inspect_results(self,
                            float_model,
                            inspect_results,
                            to_file,
                            verbose=1):
    line_length = 120

    # Show verbose results in text
    lines = self._print_inspect_results(
        float_model, inspect_results, io=None, verbose=verbose)
    lines.append('\nDetailed Model Info: \n')
    for layer, ins_res in inspect_results.items():
      lines.append('Layer ID: {}'.format(ins_res.id))
      lines.append('Layer Name: {}'.format(layer.name))
      lines.append('Layer Type: {}'.format(layer.__class__.__name__))
      lines.append('Device: {}'.format(ins_res.device))

      if ins_res.get_notes():
        lines.append('Notes: ')
        for i, note in enumerate(ins_res.get_notes()):
          lines.append('    {}. {}'.format(i + 1, note))
      else:
        lines.append('Notes: \n    None')

      lines.append('Layer Config:')
      layer_config = layer.get_config()
      formatted = pprint.pformat(layer_config)
      for line in formatted.splitlines():
        lines.append(line)

      lines.append('_' * line_length)

    with open(to_file, 'w') as f:
      f.write('\n'.join(lines))

  # Public Interfaces

  def inspect_model(self,
                    float_model,
                    input_shape=None,
                    plot=False,
                    plot_file='model.svg',
                    dump_model=False,
                    dump_model_file='inspect_model.h5',
                    dump_results=True,
                    dump_results_file='inspect_results.txt',
                    verbose=0,
                    configs={},
                    **kwargs):
    """Interface of model inspection.

    Args:
      float_model: tf.keras.Model instance, the float model to be inspected.
      input_shape: a tuple of list(tuple) contains the input shape for each input layer.
        Use default shape info in InputLayer if not set.
      verbose: int, the logging verbosity level, more detailed logging for higher verbose
        value, default to 0.
      plot: bool, whether to plot the model and save image to disk.
      plot_file: string, path of model image file when ploting the model.
      dump_model: bool, whether to dump the inspected model and save model to disk.
      dump_model_file: string, path of inspected model file.
      dump_results: bool, whether to dump the inspect results and save text to disk.
      dump_results_file: string, path of inspect results txt file.
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
      Dict of the inspect results.
    """
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

    # Disable unnecessary optimizations
    configs.update({
        "include_cle": False,
        "include_fast_ft": False,
        "include_bias_corr": False
    })
    # Configure the quantize strategy
    self._parse_configs(configs, kwargs)
    configs = self._quantize_strategy.get_configs()

    if not self._target:
      if configs['quantize_pipeline_config']['quantize_with_xcompiler']:
        logger.error(
            '[Quantizer_TF2_Invalid_Target][Invalid Target] Inspecting without specific `target`. \n{}'
            .format(Usage))

    # Handle user-defined partition
    input_layers = configs["quantize_registry_config"]['user_quantize_config'][
        'input_layers']
    output_layers = configs["quantize_registry_config"]['user_quantize_config'][
        'output_layers']
    ignore_layers = configs["quantize_registry_config"]['user_quantize_config'][
        'ignore_layers']
    if input_layers or output_layers or ignore_layers:
      input_quantize_config = configs["quantize_registry_config"][
          'input_quantize_config']
      input_quantize_config["input_layers"] = input_layers
      self._quantize_strategy.update(
          {"input_quantize_config": input_quantize_config})
      self._candidate_layers = model_utils.get_candidate_layers(
          self._float_model, input_layers, output_layers, ignore_layers)
    else:
      self._candidate_layers = None

    # Disable tf.logging warnings during quantization
    log_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')

    # Make sure model is built
    if input_shape:
      float_model, _ = model_utils.modify_input_shape(float_model, input_shape)

    if not float_model.built:
      logger.error(
          '`float_model` is not built yet. Please build model with concrete input shape '
          'before inspecting your model or call inspect_model with `input_shape` arg. Use list '
          'of shapes for multiple inputs.e.g. inspect_model(model, input_shape=[1, 224, 224, 3]) '
          'or inspect_model(model, input_shape=[[None, 224, 224, 3], [None, 64]]). All dimension should '
          'have concrete value, batch_size dimension should be None or int.')

    input_shape = float_model.input_shape
    if isinstance(input_shape, tuple):
      if None in input_shape[1:]:
        logger.error(
            '[Quantizer_TF2_Invalid_Input_Shape][Invalid input shape] '
            'Variable input shape {} is not supported yet. Please build model with concrete input shape '
            'before inspecting your model or call inspect_model with `input_shape` arg. Use list '
            'of shapes for multiple inputs.e.g. inspect_model(model, input_shape=[1, 224, 224, 3]) '
            'or inspect_model(model, input_shape=[[None, 224, 224, 3], [None, 64]]). All dimension should '
            'have concrete value, batch_size dimension should be None or int.'
            .format(input_shape[1:]))
    elif isinstance(input_shape, list):
      for per_input_shape in input_shape:
        if None in per_input_shape[1:]:
          logger.error(
              'Variable input shape {} is not supported yet. Please build model with concrete input shape '
              'before inspecting your model or call inspect_model with `input_shape` arg. Use list '
              'of shapes for multiple inputs.e.g. inspect_model(model, input_shape=[1, 224, 224, 3]) '
              'or inspect_model(model, input_shape=[[None, 224, 224, 3], [None, 64]]). All dimension should '
              'have concrete value, batch_size dimension should be None or int.'
              .format(per_input_shape[1:]))
    else:
      logger.error(
          '[Quantizer_TF2_Invalid_Input_Shape][Invalid input shape] Unsupported input shape.'
          .format(input_shape))

    with self._custom_object_scope:
      layer_metadata = self._init_inspect_results(float_model)

      # Optimize model before quantization
      self._quantize_strategy.get_optimize_pipeline().print_configs(
          verbose=verbose)
      optimized_model, layer_metadata = create_optimize_model(
          float_model,
          candidate_layers=self._candidate_layers,
          layer_metadata=layer_metadata,
          quantize_strategy=self._quantize_strategy)

      logger.debug('Quantize Pipeline Configurations:')
      self._quantize_strategy.get_quantize_registry().print_configs(
          verbose=verbose)
      self._quantize_strategy.get_quantize_pipeline().print_configs(
          verbose=verbose)
      self._quantize_strategy.get_refine_pipeline().print_configs(
          verbose=verbose)

      inspect_model, layer_metadata = create_inspect_model(
          optimized_model,
          candidate_layers=self._candidate_layers,
          layer_metadata=layer_metadata,
          quantize_strategy=self._quantize_strategy,
          target=self._target)

      if logger.debug_enabled():
        model_utils.save_model(inspect_model, 'inspect_model.h5', './debug/')

      inspect_results = self._extract_inspect_results(float_model,
                                                      inspect_model,
                                                      layer_metadata)
      self._print_inspect_results(float_model, inspect_results, verbose=verbose)

      if not self._target and type(
          self._quantize_strategy
      ) == vitis_pof2s_quantize_strategy.VitisPof2SQuantizeStrategy:
        logger.warning(
            'Inspecting without specific `target`. \n{}'.format(Usage))

      if plot:
        logger.info("Start plotting model to {}".format(plot_file))
        visualize_utils.plot_model(
            float_model,
            plot_file,
            inspect_results=inspect_results,
            show_inspect_results=True)
        logger.info(
            "Inspected model has been plotted in: {}.".format(plot_file))

      if dump_model:
        logger.info(
            "Start dumping inspected model to {}".format(dump_model_file))
        model_utils.save_model(inspect_model, dump_model_file)
        logger.info(
            "Inspected model has been dumped in: {}.".format(dump_model_file))

      if dump_results:
        logger.info(
            "Start dumping inspected results to {}".format(dump_results_file))
        self._dump_inspect_results(
            float_model, inspect_results, dump_results_file, verbose=verbose)
        logger.info("Inspected results has been dumped in: {}.".format(
            dump_results_file))

    logger.info("Inspect Finished.")
    tf.get_logger().setLevel(log_level)
    return inspect_results

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


def create_inspect_model(model, candidate_layers, layer_metadata,
                         quantize_strategy, target):
  """Inspect a `tf.keras` model with the default quantization implementation.

  Inspected model will have inspect information inside, including placement
  information.

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    model: tf.keras model to be inspected. It can have pre-trained
      weights.
    quantize_strategy: QuantizeStrategy constaining the configurations.

  Returns:
    Returns a new `tf.keras` model with inspect results.
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
      model_copy,
      candidate_layers,
      layer_metadata,
      quantize_strategy.get_quantize_registry(),
      mode='INSPECT',
      target=target)

  return quantized_model, layer_metadata
