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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.python.util import nest
from tf_nndct.graph import OpTypes
from tf_nndct.graph import dtypes
from tf_nndct.graph import op_def
from tf_nndct.graph import ops
from tf_nndct.graph import utils
from tf_nndct.utils import generic_utils
from tf_nndct.utils import keras_utils
from tf_nndct.utils import registry
from tf_nndct.utils import tensor_utils
from tf_nndct.utils import tf_utils

# Doesn't support RNN after tensorflow 2.9,
# the importing statements below is simply to avoid import error.
if tf_utils.is_tf_version_greater_equal('2.9.0'):
  from keras.layers.rnn import lstm as recurrent_v2
  from keras.layers.rnn import lstm_v1 as recurrent
elif tf_utils.is_tf_version_greater_equal('2.6'):
  # Keras is seperate from tensorflow since tf 2.6
  from keras.layers import recurrent
  from keras.layers import recurrent_v2
else:
  from tensorflow.python.keras.layers import recurrent
  from tensorflow.python.keras.layers import recurrent_v2

keras = tf.keras

_NO_LAYER_NAME = '_NO_LAYER_NAME'

class OpBuilder(object):

  _OP_COUNT = {}

  # TODO(yuwang): Maybe use op object instead of op class ?
  def __init__(self, op_cls, config=None, params=None, *args, **kwargs):
    self._op = op_cls(*args, **kwargs)
    self._config = config if config else {}
    self._params = params if params else {}

    self._AttrName = op_cls.AttrName if hasattr(op_cls, 'AttrName') else None
    self._ParamName = op_cls.ParamName if hasattr(op_cls, 'ParamName') else None

    op_type = self._op.type
    if op_type not in self._OP_COUNT:
      self._OP_COUNT[op_type] = 0
    self._OP_COUNT[op_type] = self._OP_COUNT[op_type] + 1

    for name, value in self._config.items():
      self._op.set_config(name, value)

    for name, value in self._params.items():
      self.param(name, value)

  def config(self, name, value):
    self._op.set_config(name, value)
    return self

  def attr(self, name, value):
    if not self._AttrName:
      raise ValueError('Op {} does not has any attributes'.format(
          type(self._op)))

    for attr in self._AttrName:
      if name == attr.value:
        break
    self._op.set_attr(attr, value)

    return self

  def param(self, name, value):
    if not self._ParamName:
      param = name
    else:
      param = utils.op_param_by_name(self._op, name)
      if not param:
        raise ValueError('{} does not has a param named "{}"'.format(
            type(self._op), name))

    op_type = self._op.type
    index = self._OP_COUNT[op_type] - 1
    # Naming tensor like dense_0:weight
    name = '{}_{}:{}'.format(op_type, index, name)
    tensor = tensor_utils.param_from_tf_numpy(name, value)
    self._op.set_param(param, tensor)
    return self

  def build(self):
    return self._op

_node_converter_registry = registry.Registry('node_converter')

class RegisterNodeConverter(object):
  """A decorator for registering the convertion function that converts
  the framework's op to NNDCT's op.

  For example:
  ```python
  @RegisterNodeConverter("Conv2D")
  def parse_conv2d(attrs):
    ...
    return op
  ```

  The decorator argument `op_type` is the string type of an
  op which corresponds to the `NodeDef.op` field in the proto definition.
  See https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/core/framework/node_def.proto
  """

  def __init__(self, op_types):
    """Creates a new decorator with `op_type` as the Operation type.

    Args:
      op_type: The type of an framework operation.

    Raises:
      TypeError: If `op_type` is not string or `f` is not callable.
    """
    if not isinstance(op_types, (list, tuple)):
      op_types = [op_types]
    self._op_types = op_types

  def __call__(self, f):
    """Registers the function f as converter function for op_type."""
    if not callable(f):
      raise TypeError("conversion_func must be callable.")
    for op_type in self._op_types:
      _node_converter_registry.register(f, op_type)
    return f

def _convert_node_to_generic(node):
  """Convert a node with unregistered type to a generic node by
  saving the node's config as-is.
  """
  op = OpBuilder(op_def.TFGeneric, node.get_config(),
                 node.get_params()).attr('layer_class', type(node.op)).build()
  return create_node(node.name, op, node.input_names, node.output_names)

def convert(node):
  """Convert a parser's computation node to one or more TF graph's nodes.

  Looks up node's convertion function in the registry and calls it to
  generate a new ops.Node object according to the attributes of node.
  The node's name will be used to set the name of the converted node.
  A tf.keras.layers.Layer instance without type registry will be converted
  to a TFGeneric node.

  Args:
    node: A `ComputationNode` object.

  Returns:
    A `ops.Node` converted from `ComputationNode`.
  """

  if node.type in _node_converter_registry:
    convert_func = _node_converter_registry.lookup(node.type)
    nodes = convert_func(node)
  elif node.op_type_name in ["TensorFlowOpLayer", "TFOpLambda"]:
    # op e.g ['+', tf.concat] will bre transfer to
    # TensorFlowOpLayer or TFOpLambda in different version tf
    convert_func = _node_converter_registry.lookup(node.op_type_name)
    nodes = convert_func(node)
  elif isinstance(node.op, layers.Layer):
    nodes = _convert_node_to_generic(node)
  elif isinstance(node.op, tf.Operation):
    nodes = _convert_node_to_generic(node)
  else:
    raise NotImplementedError("Unable to parse {}:\n{}".format(
        node.type, node.op))

  converted_nodes = nest.flatten(nodes)
  for cn in converted_nodes:
    if isinstance(node.op,
                  keras.layers.Layer) and cn.layer_name != _NO_LAYER_NAME:
      cn.layer_name = node.op.name
      cn.inbound_nodes = node.inbound_nodes
  return converted_nodes

_tf_type_to_nndct = {
    'Add': op_def.TFAdd,
    'AddV2': op_def.TFAdd,
    'BiasAdd': op_def.TFBiasAdd,
    'Identity': op_def.TFIdentity,
    'NoOp': op_def.TFNoOp,
    'Reshape': op_def.TFReshape,
    'Sigmoid': op_def.TFSigmoid,
    'Tanh': op_def.TFTanh,
    'GatherV2': op_def.TFGather,
    'RFFT': op_def.TFRFFT,
    'ComplexAbs': op_def.TFComplexAbs,
    'Angle': op_def.TFAngle,
    'Exp': op_def.TFExp,
    'IRFFT': op_def.TFIRFFT,
    'Pad': op_def.TFPad,
    'Transpose': op_def.TFTranspose,
    'Sum': op_def.TFSum,
    'reshape': op_def.TFReshape,
    'concat': op_def.TFConcat,
    'ConcatV2': op_def.TFConcat,
    '__operators__.add': op_def.TFAdd,
}

@RegisterNodeConverter(list(_tf_type_to_nndct.keys()))
def convert_simple_tf_op(node):
  op = _tf_type_to_nndct[node.type]()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('Placeholder')
def convert_op_placeholder(node):
  config = node.get_config()
  shape = tf_utils.tf_shape_to_list(config['shape'])
  dtype = dtypes.from_tf(config['dtype'])
  op = (
      OpBuilder(op_def.TFInput).config('shape', shape).config('dtype',
                                                              dtype).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('Cast')
def convert_op_cast(node):
  config = node.get_config()
  op = (
      OpBuilder(op_def.TFCast, None, None).config('dtype',
                                                  config['DstT']).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('Const')
def convert_op_const(node):
  config = node.get_config()
  ndarray = tf_utils.values_from_tf_const(node.op.node_def)
  # Save ndarray or raw tf.Tensor?
  config['value'] = ndarray
  op = (
      OpBuilder(op_def.TFConst, config, None).param(node.name, ndarray).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('Mul')
def convert_op_mul(node):
  op = OpBuilder(op_def.TFMultiply).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('StridedSlice')
def convert_op_strided_slice(node):
  op = (OpBuilder(op_def.TFStridedSlice, node.get_config()).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('MatMul')
def convert_op_matmul(node):
  # Parse MatMul to Dense without bias.
  op = (OpBuilder(op_def.TFDense).config('use_bias', False).build())
  return create_node(node.name, op, node.input_names, node.output_names)

if tf_utils.is_tf_version_greater_equal('2.6'):
  normalization_layer = layers.Normalization
else:
  normalization_layer = layers.experimental.preprocessing.Normalization

@RegisterNodeConverter(normalization_layer)
def convert_layer_normalization(node):
  """
  Convert layers.Normalization to GeTFNormalizationneric
  """
  params = node.get_params()
  if 'count' in params:
    params['count'] = np.array(params['count'])
  # del params['count']  # 0
  op = OpBuilder(op_def.TFNormalization, node.get_config(), params).build()
  return create_node(node.name, op, node.input_names, node.output_names)

if tf_utils.is_tf_version_greater_equal('2.6'):
  rescaling_layer = layers.Rescaling
else:
  rescaling_layer = layers.experimental.preprocessing.Rescaling

@RegisterNodeConverter(rescaling_layer)
def convert_layer_rescaling(node):
  """
  Convert layers.Rescaling to TFRescaling
  """
  params = node.get_params()
  op = OpBuilder(op_def.TFRescaling, node.get_config(), params).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.BatchNormalization)
def convert_layer_batchnorm(node):
  config = node.get_config()
  params = node.get_params()

  # See https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/python/keras/layers/normalization.py#L358
  if config['scale']:
    param_shape = params['gamma'].shape
  elif config['center']:
    param_shape = params['beta'].shape
  else:
    param_shape = params['moving_mean'].shape

  for dim in param_shape:
    if dim != 1:
      out_dim = dim
      break

  config_axis = config['axis']
  axis = config_axis[0] if isinstance(config_axis, list) else config_axis

  op = (
      OpBuilder(op_def.TFBatchNorm, config,
                params).attr('out_dim', out_dim).attr('axis', axis).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.Dense)
def convert_layer_dense(node):
  config = node.get_config()
  params = node.get_params()

  op = (
      OpBuilder(op_def.TFDense, config, params).config('activation', None).attr(
          'activation',
          config['activation']).attr('in_dim',
                                     params['kernel'].shape[0]).build())

  dense_node = ops.Node(node.name, op)
  dense_node.input_names = copy.deepcopy(node.input_names)

  if config['activation']:
    actv_node = create_activation_node(dense_node, config['activation'],
                                       node.output_names)
    return [dense_node, actv_node]
  else:
    dense_node.output_names = copy.deepcopy(node.output_names)
    return dense_node

@RegisterNodeConverter('TFOpLambda')
# in tf "2.4.4" "2.5.3" "2.6.3" "2.7.1" "2.8.0"..
# the op [e.g '+', 'tf.concat'] will be converted to TFOpLambda
# in different tf_version the TFOpLambda will in different package
# so we using the class name as look up key
def convert_tf_op_lambda(node):
  op_name = node.get_config()["function"]
  if op_name in _tf_type_to_nndct:
    op = OpBuilder(_tf_type_to_nndct[op_name], node.get_config(), None).build()
    return create_node(node.name, op, node.input_names, node.output_names)
  else:
    return _convert_node_to_generic(node)

@RegisterNodeConverter('TensorFlowOpLayer')
# in tf 2.3.4 the op [e.g '+', 'tf.concat'] will be converted to TensorFlowOpLayer
def convert_tensorflow_op_layer(node):
  op_name = node.get_config()['node_def']['op']
  if op_name in _tf_type_to_nndct:
    op = OpBuilder(_tf_type_to_nndct[op_name], node.get_config()).build()
    return create_node(node.name, op, node.input_names, node.output_names)
  else:
    return _convert_node_to_generic(node)

@RegisterNodeConverter([layers.Add, layers.add])
def convert_layer_add(node):
  op = OpBuilder(op_def.TFAdd, node.get_config(), None).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter([layers.Subtract, layers.subtract])
def convert_layer_sub(node):
  op = OpBuilder(op_def.TFSubtract, node.get_config(), None).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter([layers.Multiply, layers.multiply])
def convert_layer_multiply(node):
  op = OpBuilder(op_def.TFMultiplyLayer, node.get_config(), None).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.Embedding)
def convert_layer_embedding(node):
  op = OpBuilder(op_def.TFEmbedding, node.get_config(),
                 node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.Bidirectional)
def convert_wrapper_bidirectional(node):
  layer = node.op
  forward_op = _parse_base_rnn(layer.forward_layer)
  backward_op = _parse_base_rnn(layer.backward_layer)
  backward_op.set_config('go_backwards', True)

  op = (
      OpBuilder(op_def.TFBidirectional, node.get_config(),
                None).config('layer',
                             forward_op).config('backward_layer',
                                                backward_op).build())

  combined_params = {}
  for name, tensor in forward_op.params.items():
    combined_params['forward_rnn/' + name] = tensor
  for name, tensor in backward_op.params.items():
    combined_params['backward_rnn/' + name] = tensor
  for name, tensor in combined_params.items():
    op.set_param(name, tensor)
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.RNN)
def convert_layer_rnn(node):
  layer = node.op
  op = _parse_base_rnn(layer)
  return create_node(node.name, op, node.input_names, node.output_names)

def _parse_base_rnn(layer):
  if keras_utils.is_stacked_rnn_cells(layer.cell):
    cell_op = _parse_stacked_rnn_cells(layer.cell)
  else:
    cell_op = _parse_rnn_layer(layer.cell)

  op = (
      OpBuilder(op_def.TFRNN, layer.get_config(), None).config('cell',
                                                               cell_op).build())

  # Inherit params from cell op.
  for name, tensor in cell_op.params.items():
    op.set_param(name, tensor)
  return op

def _parse_stacked_rnn_cells(layer):
  cell_ops = []
  for cell in layer.cells:
    cell_ops.append(_parse_rnn_layer(cell))

  op = (
      OpBuilder(op_def.TFStackedRNNCells, layer.get_config(),
                None).config('cells', cell_ops).build())
  return op

_rnn_layer_to_op = {
    recurrent.LSTM: OpTypes.LSTM,
    recurrent.LSTMCell: OpTypes.LSTM_CELL,
    recurrent_v2.LSTM: OpTypes.LSTM,
    recurrent_v2.LSTMCell: OpTypes.LSTM_CELL,
}

def _parse_rnn_layer(layer):
  config = layer.get_config()
  params = keras_utils.get_named_weights(layer)

  # Naive method to determine if kernel weights are concated together.
  splited_params = {}
  units = config['units']
  if list(params.values())[0].shape[-1] == units * 4:
    suffix = ['_i', '_f', '_c', '_o']
    for name, value in params.items():
      for i in range(len(suffix)):
        splited_params[name + suffix[i]] = np.copy(value[..., i *
                                                         units:(i + 1) * units])
  else:
    splited_params = params

  op = (
      OpBuilder(ops.Operation, config, splited_params,
                _rnn_layer_to_op[type(layer)]).build())
  return op

@RegisterNodeConverter([
    recurrent.LSTM, recurrent_v2.LSTM, recurrent.LSTMCell, recurrent_v2.LSTMCell
])
def convert_layer_lstm(node):
  layer = node.op
  if type(layer) in [recurrent.LSTM, recurrent_v2.LSTM]:
    cell = layer.cell
    cell_cls = [recurrent.LSTMCell, recurrent_v2.LSTMCell]
    if type(cell) not in cell_cls:
      raise NotImplementedError(
          'Custom LSTM cell is not supported. Expected {}, but got {}'.format(
              cell_cls, type(cell)))

    if cell.recurrent_activation == activations.hard_sigmoid:
      raise ValueError(
          'recurrent_activation="hard_sigmoid" is not allowd, use "sigmoid" instead.'
      )

  op = _parse_rnn_layer(node.op)
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.SimpleRNN)
def convert_layer_simplernn(node):
  op = OpBuilder(op_def.TFSimpleRNN, node.get_config(),
                 node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.GRU)
def convert_layer_gru(node):
  op = OpBuilder(op_def.TFGRU, node.get_config(), node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.Conv2D)
def convert_layer_conv2d(node):
  config = node.get_config()
  params = node.get_params()

  # [*kernel_size, input_channels / groups, filters]
  op = (
      OpBuilder(op_def.TFConv2D,
                config, params).config('activation', None).attr(
                    'activation', config['activation']).attr(
                        'in_dim', params['kernel'].shape[-2]).build())

  conv_node = ops.Node(node.name, op)
  conv_node.input_names = copy.deepcopy(node.input_names)

  if config['activation']:
    actv_node = create_activation_node(conv_node, config['activation'],
                                       node.output_names)
    return [conv_node, actv_node]
  else:
    conv_node.output_names = copy.deepcopy(node.output_names)
    return conv_node

@RegisterNodeConverter(layers.SeparableConv2D)
def convert_layer_separableConv2D(node):
  config = node.get_config()
  params = node.get_params()
  # depthwise_kernel: (5,5,32,1) HWIO &  pointwise_kernel: (1,1,32,11) HWIO

  # [*kernel_size, input_channels / groups, filters]
  op = (
      OpBuilder(op_def.TFSeparableConv2D, config, params).config(
          'activation', None).attr('activation', config['activation']).attr(
              'in_dim', params['depthwise_kernel'].shape[-2]).build())

  conv_node = ops.Node(node.name, op)
  conv_node.input_names = copy.deepcopy(node.input_names)

  if config['activation']:
    actv_node = create_activation_node(conv_node, config['activation'],
                                       node.output_names)
    return [conv_node, actv_node]
  else:
    conv_node.output_names = copy.deepcopy(node.output_names)
    return conv_node

@RegisterNodeConverter(layers.Conv2DTranspose)
def convert_layer_conv2dtranspose(node):
  config = node.get_config()
  params = node.get_params()

  # note in tf Conv2DTranspose the weight save in HWOI rather than HWIO(conv2d)
  # in order to get right dim expand/prune in optimizer stage need transpoed
  # Conv2DTranspose weight: HWOI, Conv2d weight: HWIO so we transpose the weight
  # and re-transpose the weight in tf_nndct.pruning/runner.py [_get_sparse_model, _get_slim_model]
  params['kernel'] = params['kernel'].transpose((0, 1, 3, 2))
  # [*kernel_size, input_channels / groups, filters] output_pad
  op = (
      OpBuilder(op_def.TFConv2DTranspose, config, params).config(
          'activation', None).attr('activation', config['activation']).attr(
              'in_dim', params['kernel'].shape[-2]).build())

  conv_node = ops.Node(node.name, op)
  conv_node.input_names = copy.deepcopy(node.input_names)

  if config['activation']:
    actv_node = create_activation_node(conv_node, config['activation'],
                                       node.output_names)
    return [conv_node, actv_node]
  else:
    conv_node.output_names = copy.deepcopy(node.output_names)
    return conv_node

@RegisterNodeConverter(layers.Conv3D)
def convert_layer_conv3d(node):
  config = node.get_config()
  params = node.get_params()

  # [*kernel_size, input_channels / groups, filters]
  op = (
      OpBuilder(op_def.TFConv3D,
                config, params).config('activation', None).attr(
                    'activation', config['activation']).attr(
                        'in_dim', params['kernel'].shape[-2]).build())

  conv_node = ops.Node(node.name, op)
  conv_node.input_names = copy.deepcopy(node.input_names)

  if config['activation']:
    actv_node = create_activation_node(conv_node, config['activation'],
                                       node.output_names)
    return [conv_node, actv_node]
  else:
    conv_node.output_names = copy.deepcopy(node.output_names)
    return conv_node

@RegisterNodeConverter(layers.Conv3DTranspose)
def convert_layer_conv3dtranspose(node):
  config = node.get_config()
  params = node.get_params()
  # note in tf Conv3DTranspose the weight save in DHWOI rather than DHWIO(conv3d)
  # in order to get right dim expand/prune in optimizer stage need transpoed
  # and re-transpose the weight in tf_nndct.pruning/runner.py [_get_sparse_model, _get_slim_model]
  params['kernel'] = params['kernel'].transpose((0, 1, 2, 4, 3))

  # [*kernel_size, input_channels / groups, filters]
  op = (
      OpBuilder(op_def.TFConv3DTranspose, config, params).config(
          'activation', None).attr('activation', config['activation']).attr(
              'in_dim', params['kernel'].shape[-2]).build())

  conv_node = ops.Node(node.name, op)
  conv_node.input_names = copy.deepcopy(node.input_names)

  if config['activation']:
    actv_node = create_activation_node(conv_node, config['activation'],
                                       node.output_names)
    return [conv_node, actv_node]
  else:
    conv_node.output_names = copy.deepcopy(node.output_names)
    return conv_node

@RegisterNodeConverter(layers.DepthwiseConv2D)
def convert_layer_depthwise_conv2d(node):
  config = node.get_config()
  params = node.get_params()

  depthwise_kernel = params['depthwise_kernel']
  # [*kernel_size, input_dim, depth_multiplier]
  op = (
      OpBuilder(op_def.TFDepthwiseConv2D, config, params).config(
          'activation', None).attr('activation', config['activation']).attr(
              'group', depthwise_kernel.shape[-2]).attr(
                  'in_dim', depthwise_kernel.shape[-2]).attr(
                      'out_dim', depthwise_kernel.shape[-2] *
                      depthwise_kernel.shape[-1]).build())

  conv_node = ops.Node(node.name, op)
  conv_node.input_names = copy.deepcopy(node.input_names)

  if config['activation']:
    actv_node = create_activation_node(conv_node, config['activation'],
                                       node.output_names)
    return [conv_node, actv_node]
  else:
    conv_node.output_names = copy.deepcopy(node.output_names)
    return conv_node

@RegisterNodeConverter(layers.Concatenate)
def convert_layer_concatenate(node):
  op = OpBuilder(op_def.TFConcat, node.get_config()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.Reshape)
def convert_layer_reshape(node):
  op = OpBuilder(op_def.TFReshape, node.get_config()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

_activation_cvt_map = {
    activations.linear: op_def.TFIdentity,
    activations.relu: op_def.TFRelu,
    activations.sigmoid: op_def.TFSigmoid,
    activations.softmax: op_def.TFSoftmax,
    activations.tanh: op_def.TFTanh,
    activations.swish: op_def.TFSwish,
    activations.elu: op_def.TFElu,
    activations.exponential: op_def.TFExponential,
    activations.gelu: op_def.TFGelu,
    activations.hard_sigmoid: op_def.TFHardSigmoid,
    activations.selu: op_def.TFSelu,
    activations.softplus: op_def.TFSoftPlus,
    activations.softsign: op_def.TFSoftSign,
}

def create_activation_node(kernel_node, activation, output_names):
  actv = activations.get(activation)
  #op = ops.Operation(_activation_cvt_map[actv])
  op = _activation_cvt_map[actv]()
  actv_name = activations.serialize(actv)
  actv_node = ops.Node('/'.join([kernel_node.name, actv_name]), op)

  # Connect the activation to the kernel
  kernel_out_tensor = kernel_node.name + ':0'
  kernel_node.output_names = [kernel_out_tensor]
  actv_node.input_names = [kernel_out_tensor]
  actv_node.output_names = copy.deepcopy(output_names)

  # Dettach activation node from its kernel node.
  # The layer name of activation node should not be set to kernel's layer name.
  actv_node.layer_name = _NO_LAYER_NAME
  return actv_node

def create_node(name, op, input_names, output_names):
  node = ops.Node(name, op)
  node.input_names[:] = input_names
  node.output_names[:] = output_names
  return node
