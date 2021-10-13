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
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from enum import Enum, unique, auto
from tensorflow.python.framework import dtypes as tf_dtypes
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import activations
from tensorflow.python.keras import layers

from tf_nndct.graph import OpTypes
from tf_nndct.graph import base_op
from tf_nndct.graph import dtypes
from tf_nndct.graph import ops
from tf_nndct.graph import utils
from tf_nndct.utils import generic_utils
from tf_nndct.utils import keras_utils
from tf_nndct.utils import registry
from tf_nndct.utils import tensor_utils
from tf_nndct.utils import tf_utils

class OpBuilder(object):

  _OP_COUNT = {}

  # TODO(yuwang): Maybe use op object instead of op class ?
  def __init__(self, op_cls, attrs, params, *args, **kwargs):
    self._op = op_cls(*args, **kwargs)
    self._attrs = attrs if attrs else {}
    self._params = params if params else {}

    self._AttrName = op_cls.AttrName if hasattr(op_cls, 'AttrName') else None
    self._ParamName = op_cls.ParamName if hasattr(op_cls, 'ParamName') else None

    self._ignores = []
    self._converter_map = {}

    op_type = self._op.type
    if op_type not in self._OP_COUNT:
      self._OP_COUNT[op_type] = 0
    self._OP_COUNT[op_type] = self._OP_COUNT[op_type] + 1

  def ignore(self, names):
    if not generic_utils.is_list_or_tuple(names):
      names = [names]
    self._ignores.extend(names)
    return self

  def convert(self, name, converter):
    self._converter_map[name] = converter
    return self

  def config(self, name, value):
    self._op.set_config(name, value)
    self._ignores.append(name)
    return self

  def attr(self, name, value):
    if not self._AttrName:
      raise ValueError('Op {} does not has any attributes'.format(type(
          self._op)))
    # if not isinstance(name, self._AttrName):
    #   name = getattr(self._AttrName, name)
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
    tensor = tensor_utils.from_tf_numpy(name, value)
    self._op.set_param(param, tensor)
    return self

  def build(self):
    for name, value in self._attrs.items():
      if name in self._ignores:
        continue

      if name in self._converter_map:
        value = self._converter_map[name](value)
      self._op.set_config(name, value)

    for name, value in self._params.items():
      self.param(name, value)
    return self._op

class Cast(ops.Operation):

  @unique
  class AttrName(base_op.AutoName):
    SRC = auto()
    DST = auto()

  def __init__(self, *args, **kwargs):
    super(Cast, self).__init__(OpTypes.CAST, *args, **kwargs)
    self._attr_value_mem = {self.AttrName.SRC: [], self.AttrName.DST: []}

class TFGeneric(ops.Operation):
  """A generic op that can represent any keras layer."""

  @unique
  class AttrName(base_op.AutoName):
    ORIG_LAYER_CLASS = auto()

  def __init__(self, *args, **kwargs):
    super(TFGeneric, self).__init__(OpTypes.GENERIC, *args, **kwargs)
    self._define_attr(self.AttrName.ORIG_LAYER_CLASS, value_type=type, size=1)

#TODO(yuwang): Use _define_attr to define attr.
class TFInput(ops.Operation):

  @unique
  class AttrName(base_op.AutoName):
    SHAPE = auto()
    DTYPE = auto()

  def __init__(self, *args, **kwargs):
    super(TFInput, self).__init__(OpTypes.INPUT, *args, **kwargs)
    self._attr_value_mem = {self.AttrName.SHAPE: [], self.AttrName.DTYPE: []}

    self._attrs[self.AttrName.SHAPE] = ops.Attr(
        name=self.AttrName.SHAPE,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.SHAPE],
        occurence_type=ops.OccurenceType.REQUIRED,
    )
    self._attrs[self.AttrName.DTYPE] = ops.Attr(
        name=self.AttrName.DTYPE,
        value_type=dtypes.DType,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.DTYPE],
        occurence_type=ops.OccurenceType.REQUIRED,
    )

  @property
  def shape(self):
    return self.attr['shape']

  @shape.setter
  def shape(self, value):
    self.attr['shape'] = value

  @property
  def dtype(self):
    return self.attr['dtype']

  @dtype.setter
  def dtype(self, value):
    self.attr['dtype'] = value

class TFFlatten(base_op.Flatten):

  def __init__(self, *args, **kwargs):
    super(TFFlatten, self).__init__(OpTypes.FLATTEN, *args, **kwargs)

  @property
  def start_dim(self):
    return self._attr_value_mem[self.AttrName.START_DIM][0]

  @start_dim.setter
  def start_dim(self, value):
    self._attr_value_mem[self.AttrName.START_DIM][:] = [value]

  @property
  def end_dim(self):
    return self._attr_value_mem[self.AttrName.END_DIM][0]

  @end_dim.setter
  def end_dim(self, value):
    self._attr_value_mem[self.AttrName.END_DIM][:] = [value]

class TFDense(base_op.Dense):

  @unique
  class ParamName(base_op.AutoName):
    WEIGHTS = 'kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFDense, self).__init__(OpTypes.DENSE, *args, **kwargs)

  @property
  def use_bias(self):
    return self.attr['bias_term']

  @use_bias.setter
  def use_bias(self, value):
    self.attr['bias_term'] = value

  @property
  def units(self):
    return self.attr['out_dim']

  @units.setter
  def units(self, value):
    self.attr['out_dim'] = value

class TFBatchNorm(base_op.BatchNorm):

  @unique
  class ParamName(base_op.AutoName):
    GAMMA = 'weights'
    BETA = 'bias'
    MEAN = auto()
    VAR = auto()

  def __init__(self, *args, **kwargs):
    super(TFBatchNorm, self).__init__(OpTypes.BATCH_NORM, *args, **kwargs)

  @property
  def eps(self):
    return self._attr_value_mem[self.AttrName.EPSILON][0]

  @eps.setter
  def eps(self, value):
    self._attr_value_mem[self.AttrName.EPSILON][:] = [value]

  @property
  def num_features(self):
    return self._attr_value_mem[self.AttrName.OUT_DIM][0]

  @num_features.setter
  def num_features(self, value):
    self._attr_value_mem[self.AttrName.OUT_DIM][:] = [value]

class TFConv1D(base_op.Operation):

  @unique
  class ParamName(base_op.AutoName):
    WEIGHT = 'kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFConv1D, self).__init__(OpTypes.CONV1D, *args, **kwargs)

class TFConv2D(base_op.Conv2d):

  def __init__(self, *args, **kwargs):
    super(TFConv2D, self).__init__(OpTypes.CONV2D, *args, **kwargs)

  @property
  def filters(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @filters.setter
  def filters(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def dilation_rate(self):
    return self._attr_value_mem[self.AttrName.DILATION][::-1]

  @dilation_rate.setter
  def dilation_rate(self, value):
    self._attr_value_mem[self.AttrName.DILATION][:] = value[::-1]

  @property
  def padding(self):
    self._attr_value_mem[self.AttrName.PAD_MODE][::-1]

  @padding.setter
  def padding(self, value):
    mode = 1 if value.lower() == 'same' else 2
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = mode

  @property
  def strides(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @strides.setter
  def strides(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]

  @property
  def groups(self):
    return self._attr_value_mem[self.AttrName.GROUP][0]

  @groups.setter
  def groups(self, value):
    self._attr_value_mem[self.AttrName.GROUP][:] = [value]

  @property
  def bias(self):
    return self._attr_value_mem[self.AttrName.BIAS_TERM][0]

  @bias.setter
  def bias(self, value):
    self._attr_value_mem[self.AttrName.BIAS_TERM][:] = [value]

class TFConvTranspose2d(TFConv2D):

  def __init__(self, *args, **kwargs):
    super(TFConvTranspose2d, self).__init__(OpTypes.CONVTRANSPOSE2D, *args,
                                            **kwargs)

class TFMaxPool1D(base_op.Operation):

  def __init__(self, *args, **kwargs):
    super(TFMaxPool1D, self).__init__(OpTypes.MAX_POOL1D, *args, **kwargs)

class TFMaxPool(base_op.MaxPool):

  def __init__(self, *args, **kwargs):
    super(TFMaxPool, self).__init__(OpTypes.MAX_POOL, *args, **kwargs)

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [0]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]

class TFAvgPool(base_op.AvgPool):

  def __init__(self, *args, **kwargs):
    super(TFAvgPool, self).__init__(OpTypes.AVG_POOL, *args, **kwargs)

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [0]
    self._attr_value_mem[self.AttrName.PAD][:] = [
        value[1], value[1], value[0], value[0]
    ]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]

class TFEmbedding(ops.Operation):

  @unique
  class ParamName(base_op.AutoName):
    EMBEDDINGS = 'embeddings'

  def __init__(self, *args, **kwargs):
    super(TFEmbedding, self).__init__(OpTypes.EMBEDDING, *args, **kwargs)

class TFRNNLayer(ops.Operation):

  @unique
  class AttrName(base_op.AutoName):
    LAYER_CLASS = auto()

  def __init__(self, *args, **kwargs):
    super(TFRNNLayer, self).__init__(OpTypes.RNN_LAYER, *args, **kwargs)
    self._define_attr(self.AttrName.LAYER_CLASS, value_type=type, size=1)

class TFRNN(ops.Operation):

  @unique
  class AttrName(base_op.AutoName):
    CELL = auto()

  def __init__(self, *args, **kwargs):
    super(TFRNN, self).__init__(OpTypes.RNN, *args, **kwargs)
    self._define_attr(self.AttrName.CELL, value_type=ops.Operation, size=1)

class TFStackedRNNCells(ops.Operation):

  @unique
  class AttrName(base_op.AutoName):
    CELLS = auto()

  def __init__(self, *args, **kwargs):
    super(TFStackedRNNCells, self).__init__(OpTypes.STACKED_RNN_CELLS, *args,
                                            **kwargs)

    self._define_attr(self.AttrName.CELLS, value_type=ops.Operation, size=None)

class TFSimpleRNN(ops.Operation):

  @unique
  class ParamName(base_op.AutoName):
    KERNEL = 'kernel'
    RECURRENT_KERNEL = 'recurrent_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFSimpleRNN, self).__init__(OpTypes.SIMPLE_RNN, *args, **kwargs)

class TFLSTM(ops.Operation):

  @unique
  class ParamName(base_op.AutoName):
    KERNEL = 'kernel'
    RECURRENT_KERNEL = 'recurrent_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFLSTM, self).__init__(OpTypes.LSTM, *args, **kwargs)

class TFGRU(ops.Operation):

  @unique
  class ParamName(base_op.AutoName):
    KERNEL = 'kernel'
    RECURRENT_KERNEL = 'recurrent_kernel'
    BIAS = 'bias'

  def __init__(self, *args, **kwargs):
    super(TFGRU, self).__init__(OpTypes.GRU, *args, **kwargs)

class TFBidirectional(ops.Operation):

  def __init__(self, *args, **kwargs):
    super(TFBidirectional, self).__init__(OpTypes.BIDIRECTIONAL_RNN, *args,
                                          **kwargs)

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
  See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
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

def _convert_node_as_is(node):
  """Convert a node with unregistered type by saving node config as-is."""
  op = OpBuilder(TFGeneric, node.get_config(),
                 node.get_params()).attr('orig_layer_class',
                                         type(node.orig_op)).build()
  return create_node(node.name, op, node.input_names, node.output_names)

def convert(node):
  """Convert a parser's computation node to a TF graph's node.

  Looks up node's convertion function in the registry and calls it to
  generate a new ops.Node object according to the attributes of node.
  The node's name will be used to set the name of the converted node.
  A tf.keras.layers.Layer instance without type registry will be converted
  to a TFGeneric node.

  Args:
    node: A `ComputationNode` object.

  Returns:
    A `ops.Node` object converted from `ComputationNode`.
  """

  if node.type in _node_converter_registry.list():
    convert_func = _node_converter_registry.lookup(node.type)
    nndct_nodes = convert_func(node)
  elif isinstance(node.orig_op, layers.Layer):
    nndct_nodes = _convert_node_as_is(node)
  else:
    print(node.orig_op)
    raise NotImplementedError("Unable to parse {}".format(node.type))

  return nndct_nodes

_tf_type_to_nndct = {
    'Add': OpTypes.ADD,
    'AddV2': OpTypes.ADD,
    'BiasAdd': OpTypes.BIAS_ADD,
    'Identity': OpTypes.IDENTITY,
    'Mul': OpTypes.MULTIPLY,
    'NoOp': OpTypes.NOOP,
    'Reshape': OpTypes.RESHAPE,
    'Sigmoid': OpTypes.SIGMOID,
    'Tanh': OpTypes.TANH,
    'GatherV2': OpTypes.GATHER,
    'RFFT': 'rfft',
    'ComplexAbs': 'complex_abs',
    'Angle': 'angle',
    'Exp': 'exp',
    'IRFFT': 'irfft',
    'Pad': 'pad',
    'Transpose': 'transpose',
    'Sum': 'sum',
}

#@RegisterNodeConverter(
#    ['Add', 'AddV2', 'BiasAdd', 'Identity', 'Mul', 'NoOp', 'Sigmoid', 'Tanh'])
@RegisterNodeConverter(list(_tf_type_to_nndct.keys()))
def convert_simple_tf_op(node):
  op_type = _tf_type_to_nndct[node.type]
  op = ops.Operation(op_type)
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('Placeholder')
def convert_op_placeholder(node):
  op = (OpBuilder(TFInput, node.get_config(), None).convert(
      'shape', tf_utils.tf_shape_to_list).convert('dtype',
                                                  dtypes.from_tf).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('Cast')
def convert_op_cast(node):
  config = node.get_config()
  op = (OpBuilder(ops.Operation, None, None,
                  OpTypes.CAST).config('dtype', config['DstT']).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('Const')
def convert_op_const(node):
  attrs = node.get_config()
  ndarray = tf_utils.values_from_tf_const(node.orig_op.node_def)
  # Save ndarray or raw tf.Tensor?
  attrs['value'] = ndarray
  op = (OpBuilder(ops.Operation, attrs, None,
                  OpTypes.CONST).param(node.name, ndarray).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('StridedSlice')
def convert_op_const(node):
  attrs = node.get_config()
  op = (OpBuilder(ops.Operation, attrs, None, OpTypes.STRIDED_SLICE).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter('MatMul')
def convert_op_matmul(node):
  attrs = node.get_config()

  op = (OpBuilder(TFDense, None, None).config('use_bias', False).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.core.Dense)
def convert_layer_dense(node):
  attrs = node.get_config()
  params = node.get_params()

  op = (OpBuilder(TFDense, attrs, params).ignore('activation').config(
      'in_features', params['kernel'].shape[0]).build())

  dense_node = ops.Node(node.name, op)
  actv_node = create_activation_node(dense_node, attrs['activation'])
  dense_node.input_names[:] = node.input_names
  actv_node.output_names[:] = node.output_names

  return [dense_node, actv_node]

@RegisterNodeConverter(layers.Embedding)
def convert_layer_embedding(node):
  op = OpBuilder(TFEmbedding, node.get_config(), node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.wrappers.Bidirectional)
def convert_wrapper_bidirectional(node):
  layer = node.orig_op
  backward_op = _parse_base_rnn(layer.backward_layer)
  forward_op = _parse_base_rnn(layer.forward_layer)
  backward_op.set_config('go_backwards', True)

  op = (OpBuilder(TFBidirectional, node.get_config(), node.get_params()).config(
      'layer', forward_op).config('backward_layer', backward_op).build())
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.RNN)
def convert_layer_rnn(node):
  layer = node.orig_op
  op = _parse_base_rnn(layer)
  return create_node(node.name, op, node.input_names, node.output_names)

def _parse_base_rnn(layer):
  if keras_utils.is_stacked_rnn_cells(layer.cell):
    cell_op = _parse_stacked_rnn_cells(layer.cell)
  else:
    cell_op = _parse_rnn_layer(layer.cell)

  op = (OpBuilder(TFRNN, layer.get_config(),
                  keras_utils.keras_layer_params(layer)).config(
                      'cell', cell_op).build())
  return op

def _parse_stacked_rnn_cells(layer):
  cell_ops = []
  for cell in layer.cells:
    cell_ops.append(_parse_rnn_layer(cell))

  op = (OpBuilder(TFStackedRNNCells, layer.get_config(),
                  None).config('cells', cell_ops).build())
  return op

_rnn_layer_to_op = {
    layers.recurrent.LSTM: OpTypes.LSTM,
    layers.recurrent.LSTMCell: OpTypes.LSTM_CELL,
    layers.recurrent_v2.LSTM: OpTypes.LSTM,
    layers.recurrent_v2.LSTMCell: OpTypes.LSTM_CELL,
}

def _parse_rnn_layer(layer):
  config = layer.get_config()
  params = keras_utils.keras_layer_params(layer)

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

  op = (OpBuilder(ops.Operation, config, splited_params,
                  _rnn_layer_to_op[type(layer)]).build())
  return op

# TODO(yuwang): Parse v1 and v2 separatly and check 'recurrent_activation=sigmoid'.
@RegisterNodeConverter([
    layers.recurrent.LSTM, layers.recurrent_v2.LSTM, layers.recurrent.LSTMCell,
    layers.recurrent_v2.LSTMCell
])
def convert_layer_lstm(node):
  op = _parse_rnn_layer(node.orig_op)
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.SimpleRNN)
def convert_layer_simplernn(node):
  op = OpBuilder(TFSimpleRNN, node.get_config(), node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter([layers.recurrent.GRU, layers.recurrent_v2.GRU])
def convert_layer_gru(node):
  op = OpBuilder(TFGRU, node.get_config(), node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.Conv1D)
def convert_layer_conv1d(node):
  op = OpBuilder(TFConv1D, node.get_config(), node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

@RegisterNodeConverter(layers.MaxPooling1D)
def convert_layer_maxpool(node):
  op = OpBuilder(TFMaxPool1D, node.get_config(), node.get_params()).build()
  return create_node(node.name, op, node.input_names, node.output_names)

_activation_cvt_map = {
    activations.relu: OpTypes.RELU,
    activations.tanh: OpTypes.TANH,
    activations.sigmoid: OpTypes.SIGMOID,
    activations.linear: OpTypes.LINEAR,
}

def create_activation_node(kernel_node, activation):
  actv = activations.get(activation)
  op = ops.Operation(_activation_cvt_map[actv])
  actv_name = activations.serialize(actv)
  actv_node = ops.Node('/'.join([kernel_node.name, actv_name]), op)

  # Connect the activation to the kernel
  kernel_out_tensor = kernel_node.name + ":0"
  kernel_node.output_names.append(kernel_out_tensor)
  actv_node.input_names.append(kernel_out_tensor)
  return actv_node

def create_node(name, op, input_names, output_names):
  node = ops.Node(name, op)
  node.input_names[:] = input_names
  node.output_names[:] = output_names
  return node
