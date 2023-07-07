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
"""Model inspector and hardware-aware quantization."""

import os
import collections
import copy
import datetime
import random
import string
import getpass

import tensorflow as tf

try:
  import xir
except:
  xir = None
try:
  import xnnc
except:
  xnnc = None
try:
  import xcompiler
except:
  xcompiler = None

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_custom_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_pooling
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_conv_bn
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy import vitis_quantize_strategy_factory

activations = tf.keras.activations
serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern
logger = common_utils.VAILogger
keras = tf.keras


class LayerInspect(transforms.Transform):
  """Base class for Inspector and Hardware-aware quantization."""

  def __init__(self, input_model, mode, target):
    super(LayerInspect, self).__init__()
    self.input_model = input_model
    self.mode = mode
    self.target = target

    user_name = getpass.getuser()
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    random_string = ''.join(random.sample(string.ascii_letters, 6))
    self.model_inspect_dir = os.path.join('/tmp', user_name,
                                          'tensorflow_model_optimization')
    quantize_model_name = 'tmp_sub_graph_quantized_' + timestamp + '_' + random_string + '.h5'
    xnnc_xmodel_name = 'tmp_sub_graph_xnnc_' + timestamp + '_' + random_string + '.xmodel'

    self.quantize_model_path = os.path.join(self.model_inspect_dir,
                                            quantize_model_name)
    self.xnnc_xmodel_path = os.path.join(self.model_inspect_dir,
                                         xnnc_xmodel_name)
    self.xnnc_args = [
        '--model', self.quantize_model_path, '--out', self.xnnc_xmodel_path,
        '--type', 'tensorflow2', '--layout', 'NHWC'
    ]
    self.xcompiler_args = {"inspector": True, "target": [self.target]}
    self.patition_msg_name = 'partition_msg'

    self.skip_layers = [
        'InputLayer', 'Vitis>VitisQuantize', 'Vitis>QuantizeWrapper',
        'Vitis>VitisConvBN', 'Vitis>VitisConvBNQuantize',
        'Vitis>VitisDepthwiseConvBN', 'Vitis>VitisDepthwiseConvBNQuantize'
    ]

    self.quant_layers = [
        'Vitis>QuantizeWrapper', 'Vitis>VitisConvBN',
        'Vitis>VitisConvBNQuantize', 'Vitis>VitisDepthwiseConvBN',
        'Vitis>VitisDepthwiseConvBNQuantize'
    ]

    self.quant_config = {
        'conv_layer':
            vitis_quantize_configs.VitisQuantizeConfig(
                quantizable_weights=["kernel"],
                weight_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 0,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }],
                quantizable_biases=["bias"],
                bias_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 0,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }],
                quantizable_activations=["activation"],
                activation_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 1,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }]),
        'dwconv_layer':
            vitis_quantize_configs.VitisQuantizeConfig(
                quantizable_weights=["depthwise_kernel"],
                weight_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 0,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": 2,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }],
                quantizable_biases=["bias"],
                bias_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 0,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }],
                quantizable_activations=["activation"],
                activation_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 1,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }]),
        'PReLU_layer':
            vitis_quantize_configs.VitisQuantizeConfig(
                quantizable_weights=["alpha"],
                weight_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 0,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }],
                quantizable_outputs=[0],
                output_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 1,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }]),
        'act_layer':
            vitis_quantize_configs.VitisQuantizeConfig(
                quantizable_activations=["activation"],
                activation_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 1,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }]),
        'output_layer':
            vitis_quantize_configs.VitisQuantizeConfig(
                quantizable_outputs=[0],
                output_quantizers=[{
                    "quantizer_type": "Pof2SQuantizer",
                    "quantizer_params": {
                        "bit_width": 8,
                        "method": 1,
                        "round_mode": 1,
                        "symmetry": True,
                        "per_channel": False,
                        "channel_axis": -1,
                        "unsigned": False,
                        "narrow_range": False
                    }
                }])
    }

    input_quantizer = {
        "quantizer_type": "Pof2SQuantizer",
        "quantizer_params": {
            "bit_width": 8,
            "method": 0,
            "round_mode": 1,
            "symmetry": True,
            "per_channel": False,
            "channel_axis": -1,
            "unsigned": False,
            "narrow_range": False
        }
    }

    self.input_quantizer = vitis_quantize_configs._make_quantizer(
        input_quantizer['quantizer_type'], input_quantizer['quantizer_params'])

    self.merge_msg = 'This op may be fused by the compiler.'
    self.not_excepted_msg = 'The compiler returned results no ops pattern, not as expected.'

  def generate_xcompiler_model(self, quantized_model):

    if xir is None:
      logger.error('Please install `xir` package to quantize with targets.')
    elif xnnc is None:
      logger.error('Please install `xnnc` package to quantize with targets.')
    elif xcompiler is None:
      logger.error(
          'Please install `xcompiler` package to quantize with targets.')

    quantized_model.save(self.quantize_model_path)
    xnnc.main(self.xnnc_args)
    graph_i = xir.Graph.deserialize(self.xnnc_xmodel_path)
    graph_o = xcompiler.xcompiler(graph_i, self.xcompiler_args)
    os.remove(self.quantize_model_path)
    os.remove(self.xnnc_xmodel_path)
    return graph_o


class PadMergeInspect(LayerInspect):
  """Inspect ZeroPadding2D, by Xcompiler decision."""

  def __init__(self, input_model, mode, target):

    super(PadMergeInspect, self).__init__(input_model, mode, target)
    self.allow_multi_consumers = True
    self.pattern_xcompiler_op_types = [
        'pad', 'pad-fix', 'conv2d', 'conv2d-fix', 'transposed-conv2d',
        'transposed-conv2d-fix', 'depthwise-conv2d', 'depthwise-conv2d-fix',
        'avgpool2d', 'maxpool2d', 'pool-fix', 'downsample-fix'
    ]

  def pattern(self):
    return LayerPattern(
        'Conv2D|Conv2DTranspose|DepthwiseConv2D|MaxPooling2D|Vitis>AveragePooling2D|Vitis>VitisGlobalAveragePooling2D',
        {}, [LayerPattern('ZeroPadding2D', {}, [])])

  def replacement(self, match_layer):

    if 'padding' not in 'padding' in match_layer.layer['config']:
      return match_layer

    merge_layer_node = match_layer
    merge_layer = self.input_model.get_layer(
        merge_layer_node.layer['config']['name'])
    merge_metadata = merge_layer_node.metadata
    merge_ins_res = merge_metadata.get('inspect_result', None)

    pad_layer_node = merge_layer_node.input_layers[0]
    pad_layer_name = pad_layer_node.layer['config']['name']
    pad_layer = self.input_model.get_layer(pad_layer_name)
    pad_metadata = pad_layer_node.metadata
    pad_ins_res = pad_metadata.get('inspect_result', None)

    if pad_ins_res and merge_ins_res:
      pad_inspect_layer_quantize_config = vitis_quantize_configs.VitisQuantizeConfig(
      )

      if match_layer.layer['class_name'] in ['Conv2D', 'Conv2DTranspose']:
        merge_inspect_layer_quantize_config = self.quant_config['conv_layer']
      elif match_layer.layer['class_name'] in ['DepthwiseConv2D']:
        merge_inspect_layer_quantize_config = self.quant_config['dwconv_layer']
      elif match_layer.layer['class_name'] in [
          'MaxPooling2D', 'Vitis>AveragePooling2D',
          'Vitis>VitisGlobalAveragePooling2D'
      ]:
        merge_inspect_layer_quantize_config = self.quant_config['output_layer']
      pad_inspect_layer = keras.layers.deserialize(pad_layer_node.layer)

      pad_inspect_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
          pad_inspect_layer, pad_inspect_layer_quantize_config, self.mode)

      merge_layer_node_layer = copy.deepcopy(merge_layer_node.layer)
      if 'activation' in merge_layer_node_layer['config']:
        merge_layer_node_layer['config']['activation'] = 'linear'
      merge_inspect_layer = keras.layers.deserialize(merge_layer_node_layer)
      merge_inspect_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
          merge_inspect_layer, merge_inspect_layer_quantize_config, self.mode)

      inputs = pad_layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = pad_inspect_quant_layer(x)
      x = merge_inspect_quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        pad_ins_res.device = 'DPU'
      else:
        for xir_op in pattern_xir_ops:
          if 'device' in xir_op.get_attrs() and xir_op.get_attrs(
          )['device'] in ['CPU', 'DPU']:
            if xir_op.get_type() in [
                'conv2d', 'conv2d-fix', 'transposed-conv2d',
                'transposed-conv2d-fix', 'depthwise-conv2d',
                'depthwise-conv2d-fix', 'avgpool2d', 'maxpool2d', 'pool-fix',
                'downsample-fix'
            ]:
              if ('pad' and 'pad-fix') not in xcompiler_return_op_types:
                pad_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  pad_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  pad_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
            elif xir_op.get_type() in ['pad', 'pad-fix']:
              pad_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  pad_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  pad_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
          else:
            pad_ins_res.add_notes(
                '`ZeroPadding2D` padding with "CONSTANT"(value=0) not supported by target'
            )

    return match_layer


class PadQuantize(LayerInspect):
  """Quantizes ZeroPadding2D, by wrapping it with QuantizeWrappers.

  ZeroPadding2D Layer => QuantizeWrapper(ZeroPadding2D Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(PadQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = ['pad', 'pad-fix']

  def pattern(self):
    return LayerPattern('ZeroPadding2D', {}, [])

  def replacement(self, match_layer):

    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)
    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)

    quantize_config = vitis_quantize_configs.VitisQuantizeConfig()
    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)
    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      if ins_res.device == 'CPU' and ins_res.get_notes(layer_name) == []:
        ins_res.add_notes(
            '`ZeroPadding2D` padding with "CONSTANT"(value=0) not supported by target'
        )

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class AvgPoollikeQuantize(LayerInspect):
  """Quantizes avgpool2d, by wrapping them with QuantizeWrappers.

  avgpool2d Layer => QuantizeWrapper(avgpool2d Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(AvgPoollikeQuantize, self).__init__(input_model, mode, target)

    self.pattern_xcompiler_op_types = ['avgpool2d', 'pool-fix']

  def pattern(self):
    return LayerPattern(
        'Vitis>AveragePooling2D|Vitis>VitisGlobalAveragePooling2D', {}, [])

  def replacement(self, match_layer):

    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)
    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)
    quantize_config = metadata.get('quantize_config')

    if not quantize_config:
      quantize_config = self.quant_config['output_layer']

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)
      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        ins_res.device = 'DPU'
        ins_res.add_notes(self.merge_msg)

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class MaxPoolQuantize(LayerInspect):
  """Quantizes MaxPooling2D, by wrapping them it QuantizeWrappers.

  MaxPooling2D Layer => QuantizeWrapper(MaxPooling2D Layer inside)
  """

  def __init__(self, input_model, mode, target):

    super(MaxPoolQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'maxpool2d', 'pool-fix', 'downsample-fix'
    ]

  def pattern(self):
    return LayerPattern('MaxPooling2D', {}, [])

  def replacement(self, match_layer):
    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)
    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)
    quantize_config = metadata.get('quantize_config')
    if not quantize_config:
      quantize_config = self.quant_config['output_layer']

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)
      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) != 1:
        ins_res.device = 'CPU'
        ins_res.add_notes(self.not_excepted_msg)

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class ResizeQuantize(LayerInspect):
  """Quantizes UpSampling2D, by wrapping them it QuantizeWrappers.

  UpSampling2D Layer => QuantizeWrapper(UpSampling2D Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(ResizeQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = ['resize', 'upsample-fix']

  def pattern(self):
    return LayerPattern('UpSampling2D', {}, [])

  def replacement(self, match_layer):

    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)
    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)

    quantize_config = self.quant_config['output_layer']

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)
      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) != 1:
        ins_res.device = 'CPU'
        ins_res.add_notes(self.not_excepted_msg)

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class ConcatQuantize(LayerInspect):
  """Quantizes Concatenate, by wrapping it with QuantizeWrappers.

  Concatenate Layer => QuantizeWrapper(UpSampling2D Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(ConcatQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = ['concat', 'concat-fix']

  def pattern(self):
    return LayerPattern('Concatenate', {}, [])

  def replacement(self, match_layer):

    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)

    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)

    quantize_config = self.quant_config['output_layer']

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input

      input_list = []
      for i, input in enumerate(inputs):
        x = vitis_quantize.VitisQuantize(
            self.input_quantizer,
            self.mode,
            name='{}_{}'.format('quant_input_layer', str(i)))(
                input)
        input_list.append(x)
      x = quant_layer(input_list)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) != 1:
        ins_res.device = 'CPU'
        ins_res.add_notes(self.not_excepted_msg)
      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class ConvlikeSwishQuantize(LayerInspect):
  """Quantizes ConvlikeSwish, by wrapping them with QuantizeWrappers.

  ConvlikeSwish Layer => QuantizeWrapper(ConvlikeSwish Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(ConvlikeSwishQuantize, self).__init__(input_model, mode, target)
    self.allow_multi_consumers = True
    self.pattern_xcompiler_op_types = [
        'matmul', 'conv2d', 'conv2d-fix', 'transposed-conv2d',
        'transposed-conv2d-fix', 'hard-sigmoid', 'hard-sigmoid-fix', 'mul',
        'eltwise-fix'
    ]

  def pattern(self):
    return LayerPattern('Multiply', {}, [
        LayerPattern('Conv2D|Conv2DTranspose|Dense', {}, []),
        LayerPattern('Vitis>VitisSigmoid', {},
                     [LayerPattern('Conv2D|Conv2DTranspose|Dense', {}, [])])
    ])

  def replacement(self, match_layer):

    if match_layer.input_layers[0].layer != match_layer.input_layers[
        1].input_layers[0].layer:
      return match_layer

    mul_layer_node = match_layer
    conv_layer_node = mul_layer_node.input_layers[0]
    vitis_sigmoid_layer_node = mul_layer_node.input_layers[1]

    act_type = conv_layer_node.layer['config']['activation']
    if act_type != 'linear' and act_type['class_name'] not in [
        'Vitis>NoQuantizeActivation'
    ]:
      return match_layer

    mul_layer = self.input_model.get_layer(
        mul_layer_node.layer['config']['name'])
    mul_metadata = mul_layer_node.metadata
    mul_ins_res = mul_metadata.get('inspect_result', None)

    conv_layer_name = conv_layer_node.layer['config']['name']
    conv_layer = self.input_model.get_layer(conv_layer_name)
    conv_metadata = conv_layer_node.metadata
    conv_ins_res = conv_metadata.get('inspect_result', None)

    vitis_sigmoid_layer = self.input_model.get_layer(
        vitis_sigmoid_layer_node.layer['config']['name'])
    vitis_sigmoid_metadata = vitis_sigmoid_layer_node.metadata
    vitis_sigmoid_ins_res = vitis_sigmoid_metadata.get('inspect_result', None)

    conv_layer_quantize_config = self.quant_config['conv_layer']
    vitis_sigmoid_layer_quantize_config = self.quant_config['output_layer']
    mul_layer_quantize_config = copy.deepcopy(self.quant_config['output_layer'])

    conv_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        conv_layer, conv_layer_quantize_config, self.mode)

    vitis_sigmoid_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        vitis_sigmoid_layer, vitis_sigmoid_layer_quantize_config, self.mode)

    mul_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        mul_layer, mul_layer_quantize_config, self.mode)

    conv_quant_layer_with_xcompiler = conv_quant_layer
    vitis_sigmoid_quant_layer_with_xcompiler = vitis_sigmoid_quant_layer
    mul_quant_layer_with_xcompiler = mul_quant_layer

    if conv_ins_res and vitis_sigmoid_ins_res and mul_ins_res:

      inputs = conv_layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      conv_x = conv_quant_layer(x)
      vitis_sigmoid_x = vitis_sigmoid_quant_layer(conv_x)
      x = mul_quant_layer([conv_x, vitis_sigmoid_x])
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      # Compiler limitation.
      # The vitis_sigmoid_quant_layer layer sets pos = 7.
      vitis_sigmoid_quant_layer_name = vitis_sigmoid_quant_layer.name
      vitis_sigmoid_quant_layer = quantized_model.get_layer(
          vitis_sigmoid_quant_layer_name)
      vitis_sigmoid_quant_layer_info = vitis_sigmoid_quant_layer.get_quantize_info(
      )
      vitis_sigmoid_quant_layer_info['output_0']['info']['quant_pos_var'] = 7
      vitis_sigmoid_quant_layer.set_quantize_info(
          vitis_sigmoid_quant_layer_info)

      conv_quant_layer_with_xcompiler = conv_quant_layer
      vitis_sigmoid_quant_layer_with_xcompiler = vitis_sigmoid_quant_layer
      mul_quant_layer_with_xcompiler = mul_quant_layer
      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        conv_ins_res.device = 'CPU'
        vitis_sigmoid_ins_res.device = 'CPU'
        mul_ins_res.device = 'CPU'
        conv_ins_res.add_notes(self.not_excepted_msg)
        vitis_sigmoid_ins_res.add_notes(self.not_excepted_msg)
        mul_ins_res.add_notes(self.not_excepted_msg)

      else:
        for xir_op in pattern_xir_ops:
          if 'device' in xir_op.get_attrs() and xir_op.get_attrs(
          )['device'] in ['CPU', 'DPU']:
            if xir_op.get_type() in [
                'matmul', 'conv2d', 'conv2d-fix', 'transposed-conv2d',
                'transposed-conv2d-fix'
            ]:
              conv_ins_res.device = xir_op.get_attrs()['device']
              if ('hard-sigmoid' and
                  'hard-sigmoid-fix') not in xcompiler_return_op_types:
                vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
              if ('mul' and 'eltwise-fix') not in xcompiler_return_op_types:
                mul_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  conv_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  conv_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))

            elif xir_op.get_type() in ['hard-sigmoid', 'hard-sigmoid-fix']:
              vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  vitis_sigmoid_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  vitis_sigmoid_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))

            elif xir_op.get_type() in ['mul', 'eltwise-fix']:
              mul_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  mul_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  mul_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))

    conv_quant_layer_node = LayerNode.from_layer(
        conv_quant_layer_with_xcompiler,
        weights=conv_layer_node.weights,
        metadata=conv_metadata)

    vitis_sigmoid_quant_layer_node = LayerNode.from_layer(
        vitis_sigmoid_quant_layer_with_xcompiler,
        input_layers=[conv_quant_layer_node],
        metadata=vitis_sigmoid_metadata)

    mul_quant_layer_node = LayerNode.from_layer(
        mul_quant_layer_with_xcompiler,
        input_layers=[conv_quant_layer_node, vitis_sigmoid_quant_layer_node],
        metadata=mul_metadata)

    return mul_quant_layer_node


class DwConv2dSwishQuantize(LayerInspect):
  """Quantizes DwConv2dSwish, by wrapping them with QuantizeWrappers.

  DwConv2dSwish Layer => QuantizeWrapper(DwConv2dSwish Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(DwConv2dSwishQuantize, self).__init__(input_model, mode, target)
    self.allow_multi_consumers = True
    self.pattern_xcompiler_op_types = [
        'depthwise-conv2d', 'depthwise-conv2d-fix', 'hard-sigmoid',
        'hard-sigmoid-fix', 'mul', 'eltwise-fix'
    ]

  def pattern(self):
    return LayerPattern('Multiply', {}, [
        LayerPattern('DepthwiseConv2D', {}, []),
        LayerPattern('Vitis>VitisSigmoid', {},
                     [LayerPattern('DepthwiseConv2D', {}, [])])
    ])

  def replacement(self, match_layer):

    if match_layer.input_layers[0].layer != match_layer.input_layers[
        1].input_layers[0].layer:
      return match_layer

    mul_layer_node = match_layer
    dwconv_layer_node = mul_layer_node.input_layers[0]
    vitis_sigmoid_layer_node = mul_layer_node.input_layers[1]

    act_type = dwconv_layer_node.layer['config']['activation']
    if act_type != 'linear' and act_type['class_name'] not in [
        'Vitis>NoQuantizeActivation'
    ]:
      return match_layer

    mul_layer = self.input_model.get_layer(
        mul_layer_node.layer['config']['name'])
    mul_metadata = mul_layer_node.metadata
    mul_ins_res = mul_metadata.get('inspect_result', None)

    dwconv_layer_name = dwconv_layer_node.layer['config']['name']
    dwconv_layer = self.input_model.get_layer(dwconv_layer_name)
    dwconv_metadata = dwconv_layer_node.metadata
    dwconv_ins_res = dwconv_metadata.get('inspect_result', None)

    vitis_sigmoid_layer = self.input_model.get_layer(
        vitis_sigmoid_layer_node.layer['config']['name'])
    vitis_sigmoid_metadata = vitis_sigmoid_layer_node.metadata
    vitis_sigmoid_ins_res = vitis_sigmoid_metadata.get('inspect_result', None)

    dwconv_layer_quantize_config = self.quant_config['dwconv_layer']
    vitis_sigmoid_layer_quantize_config = self.quant_config['output_layer']
    mul_layer_quantize_config = copy.deepcopy(self.quant_config['output_layer'])

    dwconv_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        dwconv_layer, dwconv_layer_quantize_config, self.mode)
    vitis_sigmoid_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        vitis_sigmoid_layer, vitis_sigmoid_layer_quantize_config, self.mode)
    mul_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        mul_layer, mul_layer_quantize_config, self.mode)

    dwconv_quant_layer_with_xcompiler = dwconv_quant_layer
    vitis_sigmoid_quant_layer_with_xcompiler = vitis_sigmoid_quant_layer
    mul_quant_layer_with_xcompiler = mul_quant_layer

    if dwconv_ins_res and vitis_sigmoid_ins_res and mul_ins_res:

      inputs = dwconv_layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      dwconv_x = dwconv_quant_layer(x)
      vitis_sigmoid_x = vitis_sigmoid_quant_layer(dwconv_x)
      x = mul_quant_layer([dwconv_x, vitis_sigmoid_x])
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      # Compiler limitation.
      # The vitis_sigmoid_quant_layer layer sets pos = 7.
      vitis_sigmoid_quant_layer_name = vitis_sigmoid_quant_layer.name
      vitis_sigmoid_quant_layer = quantized_model.get_layer(
          vitis_sigmoid_quant_layer_name)
      vitis_sigmoid_quant_layer_info = vitis_sigmoid_quant_layer.get_quantize_info(
      )
      vitis_sigmoid_quant_layer_info['output_0']['info']['quant_pos_var'] = 7
      vitis_sigmoid_quant_layer.set_quantize_info(
          vitis_sigmoid_quant_layer_info)

      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]
      if len(pattern_xir_ops) == 0:
        dwconv_ins_res.device = 'CPU'
        dwconv_ins_res.add_notes(self.not_excepted_msg)
        vitis_sigmoid_ins_res.device = 'CPU'
        vitis_sigmoid_ins_res.add_notes(self.not_excepted_msg)
        mul_ins_res.device = 'CPU'
        mul_ins_res.add_notes(self.not_excepted_msg)
      else:
        for xir_op in pattern_xir_ops:
          if xir_op.get_type() in ['depthwise-conv2d', 'depthwise-conv2d-fix']:
            dwconv_ins_res.device = xir_op.get_attrs()['device']
            if ('hard-sigmoid' and
                'hard-sigmoid-fix') not in xcompiler_return_op_types:
              vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
            if ('mul' and 'eltwise-fix') not in xcompiler_return_op_types:
              mul_ins_res.device = xir_op.get_attrs()['device']
            if 'CPU' == xir_op.get_attrs()['device']:
              if self.patition_msg_name in xir_op.get_attrs():
                dwconv_ins_res.add_notes(
                    xir_op.get_attrs()[self.patition_msg_name])
              else:
                dwconv_ins_res.add_notes(
                    '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                    .format(op_type=xir_op.get_type()))

          elif xir_op.get_type() in ['hard-sigmoid', 'hard-sigmoid-fix']:
            vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
            if 'CPU' == xir_op.get_attrs()['device']:
              if self.patition_msg_name in xir_op.get_attrs():
                vitis_sigmoid_ins_res.add_notes(
                    xir_op.get_attrs()[self.patition_msg_name])

              else:
                vitis_sigmoid_ins_res.add_notes(
                    '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                    .format(op_type=xir_op.get_type()))

          elif xir_op.get_type() in ['mul', 'eltwise-fix']:
            mul_ins_res.device = xir_op.get_attrs()['device']
            if 'CPU' == xir_op.get_attrs()['device']:
              if self.patition_msg_name in xir_op.get_attrs():
                mul_ins_res.add_notes(
                    xir_op.get_attrs()[self.patition_msg_name])
              else:
                mul_ins_res.add_notes(
                    '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                    .format(op_type=xir_op.get_type()))

    dwconv_quant_layer_node = LayerNode.from_layer(
        dwconv_quant_layer_with_xcompiler,
        weights=dwconv_layer_node.weights,
        metadata=dwconv_metadata)

    vitis_sigmoid_quant_layer_node = LayerNode.from_layer(
        vitis_sigmoid_quant_layer_with_xcompiler,
        input_layers=[dwconv_quant_layer_node],
        metadata=vitis_sigmoid_metadata)

    mul_quant_layer_node = LayerNode.from_layer(
        mul_quant_layer_with_xcompiler,
        input_layers=[dwconv_quant_layer_node, vitis_sigmoid_quant_layer_node],
        metadata=mul_metadata)

    return mul_quant_layer_node


class ConvlikeHsigmoidQuantize(LayerInspect):
  """Quantizes ConvlikeHsigmoid, by wrapping them with QuantizeWrappers.

  ConvlikeHsigmoid Layer => QuantizeWrapper(ConvlikeHsigmoid Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(ConvlikeHsigmoidQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'matmul', 'conv2d', 'conv2d-fix', 'transposed-conv2d',
        'transposed-conv2d-fix', 'hard-sigmoid', 'hard-sigmoid-fix'
    ]

  def pattern(self):
    return LayerPattern('Vitis>VitisSigmoid', {}, [
        LayerPattern('Dense|Conv2D|Conv2DTranspose', {'activation': 'linear'},
                     [])
    ])

  def replacement(self, match_layer):

    vitis_sigmoid_layer_node = match_layer
    vitis_sigmoid_layer = self.input_model.get_layer(
        vitis_sigmoid_layer_node.layer['config']['name'])
    vitis_sigmoid_metadata = vitis_sigmoid_layer_node.metadata
    vitis_sigmoid_ins_res = vitis_sigmoid_metadata.get('inspect_result', None)

    conv_layer_node = vitis_sigmoid_layer_node.input_layers[0]
    conv_layer_name = conv_layer_node.layer['config']['name']
    conv_layer = self.input_model.get_layer(conv_layer_name)
    conv_metadata = conv_layer_node.metadata
    conv_ins_res = conv_metadata.get('inspect_result', None)

    conv_layer_quantize_config = self.quant_config['conv_layer']
    vitis_sigmoid_layer_quantize_config = self.quant_config['output_layer']

    conv_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        conv_layer, conv_layer_quantize_config, self.mode)

    vitis_sigmoid_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        vitis_sigmoid_layer, vitis_sigmoid_layer_quantize_config, self.mode)

    conv_quant_layer_with_xcompiler = conv_quant_layer
    vitis_sigmoid_quant_layer_with_xcompiler = vitis_sigmoid_quant_layer

    if conv_ins_res and vitis_sigmoid_ins_res:

      inputs = conv_layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = conv_quant_layer(x)
      x = vitis_sigmoid_quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      # Compiler limitation.
      # The vitis_sigmoid_quant_layer layer sets pos = 7.
      vitis_sigmoid_quant_layer_name = vitis_sigmoid_quant_layer.name
      vitis_sigmoid_quant_layer = quantized_model.get_layer(
          vitis_sigmoid_quant_layer_name)
      vitis_sigmoid_quant_layer_info = vitis_sigmoid_quant_layer.get_quantize_info(
      )
      vitis_sigmoid_quant_layer_info['output_0']['info']['quant_pos_var'] = 7
      vitis_sigmoid_quant_layer.set_quantize_info(
          vitis_sigmoid_quant_layer_info)

      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        conv_ins_res.device = 'CPU'
        conv_ins_res.add_notes(self.not_excepted_msg)
        vitis_sigmoid_ins_res.device = 'CPU'
        vitis_sigmoid_ins_res.add_notes(self.not_excepted_msg)
      else:
        for xir_op in pattern_xir_ops:
          if 'device' in xir_op.get_attrs() and xir_op.get_attrs(
          )['device'] in ['CPU', 'DPU']:
            if xir_op.get_type() in [
                'matmul', 'conv2d', 'conv2d-fix', 'transposed-conv2d',
                'transposed-conv2d-fix'
            ]:
              conv_ins_res.device = xir_op.get_attrs()['device']
              if ('hard-sigmoid' and
                  'hard-sigmoid-fix') not in xcompiler_return_op_types:
                vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  vitis_sigmoid_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  vitis_sigmoid_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
            elif xir_op.get_type() in ['hard-sigmoid', 'hard-sigmoid-fix']:
              vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  vitis_sigmoid_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  vitis_sigmoid_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
          else:
            conv_ins_res.add_notes(
                '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                .format(op_type=xir_op.get_type()))

    conv_quant_layer_node = LayerNode.from_layer(
        conv_quant_layer_with_xcompiler,
        weights=conv_layer_node.weights,
        metadata=conv_metadata)

    vitis_sigmoid_quant_layer_node = LayerNode.from_layer(
        vitis_sigmoid_quant_layer_with_xcompiler,
        input_layers=[conv_quant_layer_node],
        metadata=vitis_sigmoid_metadata)
    return vitis_sigmoid_quant_layer_node


class DwConv2dHsigmoidQuantize(LayerInspect):
  """Quantizes DwConv2dHsigmoid, by wrapping them with QuantizeWrappers.

  DwConv2dHsigmoid Layer => QuantizeWrapper(DwConv2dHsigmoid Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(DwConv2dHsigmoidQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'depthwise-conv2d', 'depthwise-conv2d-fix', 'hard-sigmoid',
        'hard-sigmoid-fix'
    ]

  def pattern(self):
    return LayerPattern(
        'Vitis>VitisSigmoid', {},
        [LayerPattern('DepthwiseConv2D', {'activation': 'linear'}, [])])

  def replacement(self, match_layer):

    vitis_sigmoid_layer_node = match_layer
    vitis_sigmoid_layer = self.input_model.get_layer(
        vitis_sigmoid_layer_node.layer['config']['name'])
    vitis_sigmoid_metadata = vitis_sigmoid_layer_node.metadata
    vitis_sigmoid_ins_res = vitis_sigmoid_metadata.get('inspect_result', None)

    dwconv_layer_node = vitis_sigmoid_layer_node.input_layers[0]
    dwconv_layer_name = dwconv_layer_node.layer['config']['name']
    dwconv_layer = self.input_model.get_layer(dwconv_layer_name)
    dwconv_metadata = dwconv_layer_node.metadata
    dwconv_ins_res = dwconv_metadata.get('inspect_result', None)

    dwconv_layer_quantize_config = self.quant_config['dwconv_layer']
    vitis_sigmoid_layer_quantize_config = self.quant_config['output_layer']

    dwconv_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        dwconv_layer, dwconv_layer_quantize_config, self.mode)

    vitis_sigmoid_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        vitis_sigmoid_layer, vitis_sigmoid_layer_quantize_config, self.mode)

    dwconv_quant_layer_with_xcompiler = dwconv_quant_layer
    vitis_sigmoid_quant_layer_with_xcompiler = vitis_sigmoid_quant_layer

    if dwconv_ins_res and vitis_sigmoid_ins_res:

      inputs = dwconv_layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = dwconv_quant_layer(x)
      x = vitis_sigmoid_quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      # Compiler limitation.
      # The vitis_sigmoid_quant_layer layer sets pos = 7.
      vitis_sigmoid_quant_layer_name = vitis_sigmoid_quant_layer.name
      vitis_sigmoid_quant_layer = quantized_model.get_layer(
          vitis_sigmoid_quant_layer_name)
      vitis_sigmoid_quant_layer_info = vitis_sigmoid_quant_layer.get_quantize_info(
      )
      vitis_sigmoid_quant_layer_info['output_0']['info']['quant_pos_var'] = 7
      vitis_sigmoid_quant_layer.set_quantize_info(
          vitis_sigmoid_quant_layer_info)

      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        dwconv_ins_res.device = 'CPU'
        dwconv_ins_res.add_notes(self.not_excepted_msg)
        vitis_sigmoid_ins_res.device = 'CPU'
        vitis_sigmoid_ins_res.add_notes(self.not_excepted_msg)

      else:
        for xir_op in pattern_xir_ops:
          if 'device' in xir_op.get_attrs() and xir_op.get_attrs(
          )['device'] in ['CPU', 'DPU']:
            if xir_op.get_type() in [
                'depthwise-conv2d', 'depthwise-conv2d-fix'
            ]:
              dwconv_ins_res.device = xir_op.get_attrs()['device']
              if ('hard-sigmoid' and
                  'hard-sigmoid-fix') not in xcompiler_return_op_types:
                vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  vitis_sigmoid_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  vitis_sigmoid_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))

            elif xir_op.get_type() in ['hard-sigmoid', 'hard-sigmoid-fix']:
              vitis_sigmoid_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  vitis_sigmoid_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  vitis_sigmoid_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
          else:
            dwconv_ins_res.add_notes(
                '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                .format(op_type=xir_op.get_type()))

    dwconv_quant_layer_node = LayerNode.from_layer(
        dwconv_quant_layer_with_xcompiler,
        weights=dwconv_layer_node.weights,
        metadata=dwconv_metadata)

    vitis_sigmoid_quant_layer_node = LayerNode.from_layer(
        vitis_sigmoid_quant_layer_with_xcompiler,
        input_layers=[dwconv_quant_layer_node],
        metadata=vitis_sigmoid_metadata)
    return vitis_sigmoid_quant_layer_node


class ConvlikeActQuantize(LayerInspect):
  """Quantizes ConvlikeAct, by wrapping them with QuantizeWrappers.

  ConvlikeAct Layer => QuantizeWrapper(ConvlikeAct Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(ConvlikeActQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'matmul', 'conv2d', 'conv2d-fix', 'transposed-conv2d',
        'transposed-conv2d-fix', 'relu', 'relu6', ' leaky-relu', 'prelu'
    ]

  def pattern(self):
    return LayerPattern('ReLU|LeakyReLU|PReLU|Activation', {},
                        [LayerPattern('Dense|Conv2D|Conv2DTranspose', {}, [])])

  def replacement(self, match_layer):

    if match_layer.layer['class_name'] == 'Activation' and match_layer.layer[
        'config']['activation'] not in ['relu', 'relu6']:
      return match_layer

    act_layer_node = match_layer
    act_layer = self.input_model.get_layer(
        act_layer_node.layer['config']['name'])
    act_metadata = act_layer_node.metadata
    act_ins_res = act_metadata.get('inspect_result', None)

    conv_layer_node = act_layer_node.input_layers[0]

    act_type = conv_layer_node.layer['config']['activation']
    if act_type != 'linear' and act_type['class_name'] not in [
        'Vitis>NoQuantizeActivation'
    ]:
      return match_layer

    conv_layer_name = conv_layer_node.layer['config']['name']
    conv_layer = self.input_model.get_layer(conv_layer_name)
    conv_metadata = conv_layer_node.metadata
    conv_ins_res = conv_metadata.get('inspect_result', None)
    conv_layer_quantize_config = conv_metadata.get('quantize_config')

    if not conv_layer_quantize_config:
      conv_layer_quantize_config = self.quant_config['conv_layer']

    if match_layer.layer['class_name'] == 'Activation':
      act_layer_quantize_config = self.quant_config['act_layer']

    elif match_layer.layer['class_name'] in ['ReLU', 'LeakyReLU']:
      act_layer_quantize_config = self.quant_config['output_layer']
    elif match_layer.layer['class_name'] in ['PReLU']:
      act_layer_quantize_config = self.quant_config['PReLU_layer']

    conv_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        conv_layer, conv_layer_quantize_config, self.mode)

    act_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        act_layer, act_layer_quantize_config, self.mode)

    conv_quant_layer_with_xcompiler = conv_quant_layer
    act_quant_layer_with_xcompiler = act_quant_layer

    if conv_ins_res and act_ins_res:
      inputs = conv_layer.input

      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = conv_quant_layer(x)
      if match_layer.layer['class_name'] in ['PReLU']:
        act_inspect_layer = keras.layers.deserialize(act_layer_node.layer)
        act_inspect_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
            act_inspect_layer, act_layer_quantize_config, self.mode)
        x = act_inspect_quant_layer(x)
      else:
        x = act_quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        conv_ins_res.device = 'CPU'
        conv_ins_res.add_notes(self.not_excepted_msg)
        act_ins_res.device = 'CPU'
        act_ins_res.add_notes(self.not_excepted_msg)

      else:
        for xir_op in pattern_xir_ops:
          if 'device' in xir_op.get_attrs() and xir_op.get_attrs(
          )['device'] in ['CPU', 'DPU']:

            if xir_op.get_type() in [
                'matmul', 'conv2d', 'conv2d-fix', 'transposed-conv2d',
                'transposed-conv2d-fix'
            ]:
              conv_ins_res.device = xir_op.get_attrs()['device']
              if ('relu' and 'relu6' and ' leaky-relu' and
                  'prelu') not in xcompiler_return_op_types:
                act_ins_res.device = xir_op.get_attrs()['device']

              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  act_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  act_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))

            elif xir_op.get_type() in ['relu', 'relu6', ' leaky-relu', 'prelu']:
              act_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  act_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  act_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
          else:
            conv_ins_res.add_notes(
                '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                .format(op_type=xir_op.get_type()))

    conv_quant_layer_node = LayerNode.from_layer(
        conv_quant_layer_with_xcompiler,
        weights=conv_layer_node.weights,
        metadata=conv_metadata)

    act_quant_layer_node = LayerNode.from_layer(
        act_quant_layer_with_xcompiler,
        input_layers=[conv_quant_layer_node],
        weights=match_layer.weights,
        metadata=act_metadata)

    return act_quant_layer_node


class DwConv2dActQuantize(LayerInspect):
  """Quantizes DwConv2dAct, by wrapping them with QuantizeWrappers.

  DwConv2dAct Layer => QuantizeWrapper(DwConv2dAct Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(DwConv2dActQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'depthwise-conv2d', 'depthwise-conv2d-fix', 'relu', 'relu6',
        ' leaky-relu', 'prelu'
    ]

  def pattern(self):
    return LayerPattern('ReLU|LeakyReLU|PReLU|Activation', {},
                        [LayerPattern('DepthwiseConv2D', {}, [])])

  def replacement(self, match_layer):

    act_layer_node = match_layer
    if act_layer_node.layer[
        'class_name'] == 'Activation' and act_layer_node.layer['config'][
            'activation'] not in ['relu', 'relu6']:
      return match_layer

    dwconv_layer_node = act_layer_node.input_layers[0]
    act_type = dwconv_layer_node.layer['config']['activation']
    if act_type != 'linear' and act_type['class_name'] not in [
        'Vitis>NoQuantizeActivation'
    ]:
      return match_layer

    act_layer = self.input_model.get_layer(
        act_layer_node.layer['config']['name'])
    act_metadata = act_layer_node.metadata
    act_ins_res = act_metadata.get('inspect_result', None)

    dwconv_layer_name = dwconv_layer_node.layer['config']['name']
    dwconv_layer = self.input_model.get_layer(dwconv_layer_name)
    dwconv_metadata = dwconv_layer_node.metadata
    dwconv_ins_res = dwconv_metadata.get('inspect_result', None)

    dwconv_layer_quantize_config = self.quant_config['dwconv_layer']

    if match_layer.layer['class_name'] == 'Activation' and match_layer.layer[
        'config']['activation'] in ['relu', 'relu6']:
      act_layer_quantize_config = self.quant_config['act_layer']

    elif match_layer.layer['class_name'] in ['ReLU', 'LeakyReLU']:
      act_layer_quantize_config = self.quant_config['output_layer']
    elif match_layer.layer['class_name'] in ['PReLU']:
      act_layer_quantize_config = self.quant_config['PReLU_layer']

    dwconv_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        dwconv_layer, dwconv_layer_quantize_config, self.mode)

    act_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        act_layer, act_layer_quantize_config, self.mode)

    dwconv_quant_layer_with_xcompiler = dwconv_quant_layer
    act_quant_layer_with_xcompiler = act_quant_layer

    if dwconv_ins_res and act_ins_res:

      inputs = dwconv_layer.input

      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)

      x = dwconv_quant_layer(x)
      if match_layer.layer['class_name'] in ['PReLU']:
        act_inspect_layer = keras.layers.deserialize(act_layer_node.layer)
        act_inspect_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
            act_inspect_layer, act_layer_quantize_config, self.mode)
        x = act_inspect_quant_layer(x)
      else:
        x = act_quant_layer(x)

      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        dwconv_ins_res.device = 'CPU'
        dwconv_ins_res.add_notes(self.not_excepted_msg)
        act_ins_res.device = 'CPU'
        act_ins_res.add_notes(self.not_excepted_msg)
      else:
        for xir_op in pattern_xir_ops:
          if 'device' in xir_op.get_attrs() and xir_op.get_attrs(
          )['device'] in ['CPU', 'DPU']:
            if xir_op.get_type() in [
                'depthwise-conv2d', 'depthwise-conv2d-fix'
            ]:
              dwconv_ins_res.device = xir_op.get_attrs()['device']
              if ('relu' and 'relu6' and ' leaky-relu' and
                  'prelu') not in xcompiler_return_op_types:
                act_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  act_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  act_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))

            elif xir_op.get_type() in ['relu', 'relu6', ' leaky-relu', 'prelu']:
              act_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  act_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  act_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
          else:
            dwconv_ins_res.add_notes(
                '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                .format(op_type=xir_op.get_type()))

    dwconv_quant_layer_node = LayerNode.from_layer(
        dwconv_quant_layer_with_xcompiler,
        weights=dwconv_layer_node.weights,
        metadata=dwconv_metadata)

    act_quant_layer_node = LayerNode.from_layer(
        act_quant_layer_with_xcompiler,
        input_layers=[dwconv_quant_layer_node],
        weights=match_layer.weights,
        metadata=match_layer.metadata)

    return act_quant_layer_node


class ConvlikeQuantize(LayerInspect):
  """Quantizes Convlike, by wrapping them with QuantizeWrappers.

  Convlike Layer => QuantizeWrapper(Convlike Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(ConvlikeQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'matmul',
        'conv2d',
        'conv2d-fix',
        'transposed-conv2d',
        'transposed-conv2d-fix',
    ]

  def pattern(self):
    return LayerPattern('Dense|Conv2D|Conv2DTranspose', {}, [])

  def replacement(self, match_layer):

    act_type = match_layer.layer['config']['activation']
    if act_type != 'linear' and act_type['class_name'] not in [
        'Vitis>NoQuantizeActivation'
    ]:
      return match_layer

    match_layer.layer['config']['activation'] = 'linear'

    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)
    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)
    quantize_config = metadata.get('quantize_config')

    if not quantize_config:
      quantize_config = self.quant_config['conv_layer']

    new_layer = keras.layers.deserialize(match_layer.layer)
    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        new_layer, quantize_config, self.mode)
    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)
      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) != 1:
        ins_res.device = 'CPU'
        ins_res.add_notes(self.not_excepted_msg)

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class DwConv2dQuantize(LayerInspect):
  """Quantizes DepthwiseConv2D, by wrapping it with QuantizeWrappers.

  DepthwiseConv2D Layer => QuantizeWrapper(DepthwiseConv2D Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(DwConv2dQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'depthwise-conv2d', 'depthwise-conv2d-fix'
    ]

  def pattern(self):
    return LayerPattern('DepthwiseConv2D', {}, [])

  def replacement(self, match_layer):

    act_type = match_layer.layer['config']['activation']
    if act_type != 'linear' and act_type['class_name'] not in [
        'Vitis>NoQuantizeActivation'
    ]:
      return match_layer

    match_layer.layer['config']['activation'] = 'linear'

    layer_name = match_layer.layer['config']['name']

    layer = self.input_model.get_layer(layer_name)

    metadata = match_layer.metadata
    quantize_config = metadata.get('quantize_config')
    ins_res = metadata.get('inspect_result', None)

    if not quantize_config:
      quantize_config = self.quant_config['dwconv_layer']

    new_layer = keras.layers.deserialize(match_layer.layer)
    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        new_layer, quantize_config, self.mode)
    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)

      x = quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)
      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) != 1:
        ins_res.device = 'CPU'
        ins_res.add_notes(self.not_excepted_msg)

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class EltwiseReluQuantize(LayerInspect):
  """Quantizes EltwiseRelu, by wrapping them with QuantizeWrappers.

  EltwiseRelu Layer => QuantizeWrapper(EltwiseRelu Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(EltwiseReluQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'mul', 'add', 'relu', 'eltwise-fix', 'depthwise-fix'
    ]

  def pattern(self):
    return LayerPattern('ReLU|Activation', {},
                        [LayerPattern('Add|Multiply', {}, [])])

  def replacement(self, match_layer):

    if match_layer.layer['class_name'] == 'Activation' and match_layer.layer[
        'config']['activation'] not in ['relu']:
      return match_layer

    relu_layer_node = match_layer
    relu_layer = self.input_model.get_layer(
        relu_layer_node.layer['config']['name'])
    relu_metadata = relu_layer_node.metadata
    relu_ins_res = relu_metadata.get('inspect_result', None)

    add_mul_layer_node = relu_layer_node.input_layers[0]
    add_mul_layer_name = add_mul_layer_node.layer['config']['name']

    add_mul_layer = self.input_model.get_layer(add_mul_layer_name)
    add_mul_metadata = add_mul_layer_node.metadata

    add_mul_ins_res = add_mul_metadata.get('inspect_result', None)
    add_mul_layer_quantize_config = add_mul_metadata.get('quantize_config')
    relu_layer_quantize_config = relu_metadata.get('quantize_config')

    if not add_mul_layer_quantize_config:
      add_mul_layer_quantize_config = self.quant_config['output_layer']

    if not relu_layer_quantize_config:
      if match_layer.layer['class_name'] == 'Activation' and match_layer.layer[
          'config']['activation'] in ['relu']:
        relu_layer_quantize_config = self.quant_config['act_layer']

      elif match_layer.layer['class_name'] in ['ReLU']:
        relu_layer_quantize_config = self.quant_config['output_layer']

    add_mul_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        add_mul_layer, add_mul_layer_quantize_config, self.mode)

    relu_quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        relu_layer, relu_layer_quantize_config, self.mode)

    add_mul_quant_layer_with_xcompiler = add_mul_quant_layer
    relu_quant_layer_with_xcompiler = relu_quant_layer

    if add_mul_ins_res and relu_ins_res:

      inputs = add_mul_layer.input
      input_list = []
      for i, input in enumerate(inputs):
        x = vitis_quantize.VitisQuantize(
            self.input_quantizer,
            self.mode,
            name='{}_{}'.format('quant_input_layer', str(i)))(
                input)
        input_list.append(x)
      x = add_mul_quant_layer(input_list)
      x = relu_quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      graph_o = self.generate_xcompiler_model(quantized_model)
      xcompiler_return_op_types = [
          xir_op.get_type() for xir_op in graph_o.get_ops()
      ]
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        add_mul_ins_res.device = 'CPU'
        add_mul_ins_res.add_notes(self.not_excepted_msg)
        relu_ins_res.device = 'CPU'
        relu_ins_res.add_notes(self.not_excepted_msg)

      else:
        for xir_op in pattern_xir_ops:
          if 'device' in xir_op.get_attrs() and xir_op.get_attrs(
          )['device'] in ['CPU', 'DPU']:
            if xir_op.get_type() in ['eltwise-fix', 'depthwise-fix']:
              add_mul_ins_res.device = xir_op.get_attrs()['device']
              if ('mul' and 'add' and 'relu') not in xcompiler_return_op_types:
                relu_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  relu_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  relu_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))

            elif xir_op.get_type() in ['mul', 'add', 'relu']:
              relu_ins_res.device = xir_op.get_attrs()['device']
              if 'CPU' == xir_op.get_attrs()['device']:
                if self.patition_msg_name in xir_op.get_attrs():
                  relu_ins_res.add_notes(
                      xir_op.get_attrs()[self.patition_msg_name])
                else:
                  relu_ins_res.add_notes(
                      '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                      .format(op_type=xir_op.get_type()))
          else:
            add_mul_ins_res.add_notes(
                '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                .format(op_type=xir_op.get_type()))

    add_mul_layer_node = LayerNode.from_layer(
        add_mul_quant_layer_with_xcompiler,
        weights=match_layer.input_layers[0].weights,
        metadata=add_mul_metadata)

    relu_layer_node = LayerNode.from_layer(
        relu_quant_layer_with_xcompiler,
        input_layers=[add_mul_layer_node],
        weights=match_layer.weights,
        metadata=relu_metadata)
    return relu_layer_node


class EltwiseQuantize(LayerInspect):
  """Quantizes Eltwise, by wrapping them with QuantizeWrappers.

  Add or Multiply Layer => QuantizeWrapper(Add or Multiply Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(EltwiseQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = [
        'mul', 'add', 'eltwise-fix', 'depthwise-fix'
    ]

  def pattern(self):
    return LayerPattern('Add|Multiply', {}, [])

  def replacement(self, match_layer):

    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)

    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)
    layer_quantize_config = metadata.get('quantize_config')

    if not layer_quantize_config:
      quantize_config = self.quant_config['output_layer']

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input

      input_list = []
      for i, input in enumerate(inputs):
        x = vitis_quantize.VitisQuantize(
            self.input_quantizer,
            self.mode,
            name='{}_{}'.format('quant_input_layer', str(i)))(
                input)

        input_list.append(x)
      x = quant_layer(input_list)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) != 1:
        ins_res.device = 'CPU'
        ins_res.add_notes(self.not_excepted_msg)

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class FlattenReshapeQuantize(LayerInspect):
  """Quantizes FlattenReshape, by wrapping them with QuantizeWrappers.

  Flatten or Reshape Layer => QuantizeWrapper(Flatten or Reshape Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(FlattenReshapeQuantize, self).__init__(input_model, mode, target)
    self.pattern_xcompiler_op_types = ['flatten', 'reshape', 'reshape-fix']

  def pattern(self):
    return LayerPattern('Flatten|Reshape', {}, [])

  def replacement(self, match_layer):

    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)

    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)

    quantize_config = vitis_quantize_configs.VitisQuantizeConfig()

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)
      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) == 0:
        ins_res.device = 'DPU'

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class HsigmoidQuantize(LayerInspect):
  """Quantizes Hsigmoid, by wrapping them with QuantizeWrappers.

  Hsigmoid Layer => QuantizeWrapper(Hsigmoid Layer inside)
  """

  def __init__(self, input_model, mode, target):
    super(HsigmoidQuantize, self).__init__(input_model, mode, target)

    self.pattern_xcompiler_op_types = ['hard-sigmoid', 'hard-sigmoid-fix']

  def pattern(self):
    return LayerPattern('Vitis>VitisSigmoid', {}, [])

  def replacement(self, match_layer):

    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)
    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)

    quantize_config = self.quant_config['output_layer']

    quant_layer = vitis_quantize_wrapper.QuantizeWrapper(
        layer, quantize_config, self.mode)

    quant_layer_with_xcompiler = quant_layer

    if ins_res:
      inputs = layer.input
      x = vitis_quantize.VitisQuantize(
          self.input_quantizer, self.mode, name='quant_input_layer')(
              inputs)
      x = quant_layer(x)
      quantized_model = tf.keras.Model(inputs=inputs, outputs=x)

      # Compiler limitation.
      # The vitis_sigmoid_quant_layer layer sets pos = 7.
      vitis_sigmoid_quant_layer_name = quant_layer.name
      vitis_sigmoid_quant_layer = quantized_model.get_layer(
          vitis_sigmoid_quant_layer_name)
      vitis_sigmoid_quant_layer_info = vitis_sigmoid_quant_layer.get_quantize_info(
      )
      vitis_sigmoid_quant_layer_info['output_0']['info']['quant_pos_var'] = 7
      vitis_sigmoid_quant_layer.set_quantize_info(
          vitis_sigmoid_quant_layer_info)

      graph_o = self.generate_xcompiler_model(quantized_model)
      pattern_xir_ops = [
          xir_op for xir_op in graph_o.get_ops()
          if xir_op.get_type() in self.pattern_xcompiler_op_types
      ]

      if len(pattern_xir_ops) != 1:
        ins_res.device = 'CPU'
        ins_res.add_notes(self.not_excepted_msg)

      else:
        xir_op = pattern_xir_ops[0]
        if 'device' in xir_op.get_attrs() and xir_op.get_attrs()['device'] in [
            'CPU', 'DPU'
        ]:
          ins_res.device = xir_op.get_attrs()['device']
          if 'CPU' == xir_op.get_attrs()['device']:
            if self.patition_msg_name in xir_op.get_attrs():
              ins_res.add_notes(xir_op.get_attrs()[self.patition_msg_name])
            else:
              ins_res.add_notes(
                  '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
                  .format(op_type=xir_op.get_type()))
        else:
          ins_res.add_notes(
              '`{op_type}` is not supported by target, or `{op_type}` instance does not meet the conditions supported by the target.'
              .format(op_type=xir_op.get_type()))

    quant_layer_node = LayerNode.from_layer(
        quant_layer_with_xcompiler,
        weights=match_layer.weights,
        metadata=metadata)
    return quant_layer_node


class ActLayerInspect(LayerInspect):
  """Inspect ActLayer."""

  def __init__(self, input_model, mode, target):
    super(ActLayerInspect, self).__init__(input_model, mode, target)

  def pattern(self):
    return LayerPattern('.*')

  def replacement(self, match_layer):

    layer_node = match_layer.layer
    metadata = match_layer.metadata
    ins_res = metadata.get('inspect_result', None)

    if layer_node['class_name'] in self.skip_layers:
      return match_layer

    quantize_config = metadata.get('quantize_config')

    def _get_act_type(layer_node):
      if layer_node['class_name'] == 'ReLU':
        return 'relu'
      elif layer_node['class_name'] == 'LeakyReLU':
        return 'leaky_relu'
      elif layer_node['class_name'] == 'PReLU':
        return 'prelu'
      elif layer_node['class_name'] == 'Activation':
        return layer_node['config']['activation']
      elif layer_node['class_name'] == 'Vitis>VitisSigmoid':
        return 'hard_sigmoid'
      return 'activation'

    # DPU now only supports below activations:
    # 1. Linear
    # 2. ReLU/ReLU6
    # 3. LeakyReLU(alpha==0.1): alpha will be converted to 26/256
    # 4. Sigmoid/HardSigmoid: sigmoid will be converted to hard sigmoid by default
    # 5. Swish/HardSwish: swish will be converted to hard swish by default
    # 6. Softmax: softmax will be mapped to run on Softmax IP
    #
    # Other acitvations will not be quantized, for example:
    # 1. keras.layers.Activation layer with other types of activation function will not be quantized.
    # 2. keras.layers.LeakyReLU layer with alpha!=0.1 will not be quantized.
    if not quantize_config:
      if layer_node['class_name'] in [
          'ReLU', 'LeakyReLU', 'PReLU', 'Activation'
      ]:

        info_msg = 'Standalone activation `{}` layer {} is not supported.'

        logger.info(
            info_msg.format(
                _get_act_type(layer_node), layer_node['config']['name']))

        if ins_res:
          ins_res.add_notes(
              'Standalone activation `{}` is not supported.'.format(
                  _get_act_type(layer_node)))
      return match_layer

    layer_name = match_layer.layer['config']['name']
    layer = self.input_model.get_layer(layer_name)
    if ins_res:
      ins_res.add_notes('`{}` is not supported by target.'.format(
          _get_act_type(layer_node)))

    return match_layer


xcompiler_pattern_id_to_quantize_pattern_name = collections.OrderedDict({
    "id_1": "PadQuantize",
    "id_5": "AvgPoollikeQuantize",
    "id_6": "MaxPoolQuantize",
    "id_8": "ResizeQuantize",
    "id_14": "ConcatQuantize",
    "id_15": "ConvlikeSwishQuantize",
    "id_16": "DwConv2dSwishQuantize",
    "id_17": "ConvlikeHsigmoidQuantize",
    "id_18": "DwConv2dHsigmoidQuantize",
    "id_19": "ConvlikeActQuantize",
    "id_20": "DwConv2dActQuantize",
    "id_21": "ConvlikeQuantize",
    "id_22": "DwConv2dQuantize",
    "id_24": "EltwiseReluQuantize",
    "id_25": "EltwiseQuantize",
    "id_2": "FlattenReshapeQuantize",
    "id_23": "HsigmoidQuantize",
})

quantize_pattern_dict = collections.OrderedDict({
    "PadMergeInspect": PadMergeInspect,
    "PadQuantize": PadQuantize,
    "AvgPoollikeQuantize": AvgPoollikeQuantize,
    "MaxPoolQuantize": MaxPoolQuantize,
    "ResizeQuantize": ResizeQuantize,
    "ConcatQuantize": ConcatQuantize,
    "ConvlikeSwishQuantize": ConvlikeSwishQuantize,
    "DwConv2dSwishQuantize": DwConv2dSwishQuantize,
    "ConvlikeHsigmoidQuantize": ConvlikeHsigmoidQuantize,
    "DwConv2dHsigmoidQuantize": DwConv2dHsigmoidQuantize,
    "ConvlikeActQuantize": ConvlikeActQuantize,
    "DwConv2dActQuantize": DwConv2dActQuantize,
    "ConvlikeQuantize": ConvlikeQuantize,
    "DwConv2dQuantize": DwConv2dQuantize,
    "EltwiseReluQuantize": EltwiseReluQuantize,
    "EltwiseQuantize": EltwiseQuantize,
    "FlattenReshapeQuantize": FlattenReshapeQuantize,
    "HsigmoidQuantize": HsigmoidQuantize,
    "ActLayerInspect": ActLayerInspect
})
