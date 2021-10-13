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
"""Parse a keras model to nndct graph."""
import collections
import json
import tensorflow as tf

from tensorflow.core.protobuf import config_pb2
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.util import nest

from tf_nndct.graph import OpTypes
from tf_nndct.graph import converter
from tf_nndct.graph import ops
from tf_nndct.graph import utils
from tf_nndct.utils import generic_utils
from tf_nndct.utils import keras_utils
from tf_nndct.utils import logging
from tf_nndct.utils import tensor_utils
from tf_nndct.utils import tf_utils

def from_keras_model(model, input_signature):
  logging.vlog(1, 'input_signature: {}'.format(input_signature))
  if not generic_utils.is_list_or_tuple(input_signature):
    input_signature = generic_utils.to_list(input_signature)

  #flat_input_signature = nest.flatten(input_signature)
  #batch_input_signature = []
  #for signature in flat_input_signature:
  #  batch_input_signature.append(
  #      tf.TensorSpec(shape=(1,) + signature.shape, dtype=signature.dtype))
  #batch_input_signature = nest.pack_sequence_as(input_signature,
  #                                              batch_input_signature)

  func_graph = get_func_graph(model, input_signature)

  scope_to_layer = map_scope_to_layer(model, '')
  logging.vlog(
      1, 'scope_name: (layer, parent_layer)\n{}'.format('\n'.join(
          [f'{key}: {value}' for key, value in scope_to_layer.items()])))

  graph = parse_to_graph(func_graph, scope_to_layer)
  graph.name = model.name
  return graph

def get_func_graph(model, input_signature=None, *args, **kwargs):
  func = keras_utils.trace_model_call(model, input_signature)
  concrete_func = func.get_concrete_function(*args, **kwargs)

  frozen_func = tf_utils.convert_to_constants(concrete_func,
                                              lower_control_flow=False)
  graph_def = frozen_func.graph.as_graph_def()
  utils.maybe_export_graph(_FROZEN_FUNC_GRAPH, graph_def)

  input_tensors = [
      tensor for tensor in frozen_func.inputs
      if tensor.dtype != tf.dtypes.resource
  ]
  output_tensors = frozen_func.outputs

  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  #rewrite_options.constant_folding = rewrite_options.ON
  rewrite_options.optimizers.append('constfold')
  graph_def = _run_graph_optimizations(graph_def,
                                       input_tensors,
                                       output_tensors,
                                       config=config,
                                       graph=frozen_func.graph)

  #from tensorflow.python.keras.saving import saving_utils as _saving_utils
  #from tensorflow.python.framework import convert_to_constants as _convert_to_constants
  #from tensorflow.python.framework import dtypes as _dtypes
  #func = _saving_utils.trace_model_call(model, input_signature)
  #concrete_func = func.get_concrete_function()
  #funcs = [concrete_func]

  #frozen_func, graph_def = (
  #    _convert_to_constants.convert_variables_to_constants_v2_as_graph(
  #        funcs[0], lower_control_flow=False))

  #input_tensors = [
  #    tensor for tensor in frozen_func.inputs
  #    if tensor.dtype != _dtypes.resource
  #]
  #output_tensors = frozen_func.outputs

  #config = config_pb2.ConfigProto()
  #rewrite_options = config.graph_options.rewrite_options
  ##rewrite_options.constant_folding = rewrite_options.ON
  #rewrite_options.optimizers.append('constfold')
  #graph_def = _run_graph_optimizations(
  #    graph_def,
  #    input_tensors,
  #    output_tensors,
  #    config=config,
  #    graph=frozen_func.graph)

  utils.maybe_export_graph(_OPT_TF_GRAPH, graph_def)

  logging.vlog(4, 'Optimized GraphDef:\n{}'.format(str(graph_def)))
  with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(graph_def, name='')

  func_graph = concrete_func.graph
  return (tf_graph, func_graph.structured_input_signature,
          func_graph.structured_outputs)

def map_scope_to_layer(layer, scope, parent=None):
  if not isinstance(layer, base_layer.Layer):
    return {}

  scope_to_layer = {}

  layer_scope = "/".join([scope, layer.name]) if scope else layer.name
  scope_to_layer[layer_scope] = (layer, parent)

  # There is no _gather_unique_layers in old version.
  # layers = layer._gather_unique_layers()
  layers = keras_utils.get_layers(layer)
  for sub_layer in layers:
    layer_dict = map_scope_to_layer(sub_layer, layer_scope, layer)
    scope_to_layer.update(layer_dict)

  return scope_to_layer

def parse_to_graph(func_graph, scope_to_layer=None):
  # op_name => ComputationNode name
  tf_graph, input_signature, structured_output_tensors = func_graph

  computation_graph = ComputationGraph.from_tf_graph(tf_graph, scope_to_layer)
  logging.vlog(2, 'ComputationGraph\n {}'.format(computation_graph))

  # Parse computation nodes to nndct nodes.
  nndct_nodes = []
  for node in computation_graph.nodes:
    nndct_nodes.extend(nest.flatten(converter.convert(node)))

  # Create all tensors
  tensors = {}
  for node in nndct_nodes:
    for name in node.output_names:
      tensors[name] = node.produce(name)

  # Build connections.
  for node in nndct_nodes:
    for name in node.input_names:
      node.consume(tensors[name])

  graph = ops.Graph()
  for node in nndct_nodes:
    graph.add_node(node)

  # The tensors in FuncGraph.structured_output_tensors are outputs from
  # Identity node added to the graph. Since all Identity nodes will be removed
  # in graph refining, so we have to find the actual output tensors before
  # that process.
  output_tensors = []
  for tensor in nest.flatten(structured_output_tensors):
    # The output tensors does not exist in graph, so we can't get the output
    # node by tensor's producer, like:
    # node = graph.tensor(tf_tensor.name).producer
    node_name = tf_utils.node_name_from_input(tensor.name)
    node = graph.node(node_name)
    assert node.op.type == OpTypes.IDENTITY
    output_tensors.append(node.in_tensors[0])
  output_tensors = nest.pack_sequence_as(structured_output_tensors,
                                         output_tensors)

  utils.maybe_export_graph(_RAW_NNDCT_GRAPH, graph)
  graph = run_graph_refining(graph)
  utils.maybe_export_graph(_FINAL_NNDCT_GRAPH, graph)

  graph = utils.topological_sort(graph)
  # Get args part from input_signature (args, kwargs)
  graph.input_signature = input_signature[0]
  graph.structured_output_tensors = output_tensors

  logging.vlog(2, 'NndctGraph\n{}'.format(graph))
  logging.vlog(2, 'input_signature:{}'.format(input_signature))
  logging.vlog(2, 'output_tensors:{}'.format(output_tensors))
  return graph

def run_graph_refining(graph):
  # Executed in sequence.
  refiners = [
      FoldConst,
      FoldBiasRefiner,
      RemoveIdentity,
      RemoveRNNRedundantInput,
      RemoveIsolatedNode,
      MergeBidirectionalRNN,
      RenameParamTensor,
  ]

  for refiner_cls in refiners:
    refiner = refiner_cls()
    result = refiner.refine_graph(graph)
    logging.vlog(
        2, 'Result of refining pass [{}]: {}'.format(result.refiner,
                                                     result.message))
  return graph

class ComputationGraph(object):

  def __init__(self):

    self._name_to_node = {}
    self._op_to_node = {}

  @classmethod
  def from_tf_graph(cls, tf_graph, scope_to_layer=None):
    layer_to_inbound_nodes = map_layer_to_inbound_nodes(scope_to_layer)

    graph = cls()
    for op in tf_graph.get_operations():
      graph._add_op(op, scope_to_layer, layer_to_inbound_nodes)

    graph._mark_computation_edges()
    return graph

  def _add_op(self, op, scope_to_layer, layer_to_inbound_nodes):
    # If an operation belongs to a keras layer, we add this op to the layer's
    # scope ops; Otherwise, we treat the op as a standalone computation node.
    layer = belongs_to_keras_layer(op, scope_to_layer)
    # tf.keras.layers.Layer or tf.Operation as a node
    node = layer if layer else op
    computation_node = self._name_to_node.get(node.name, ComputationNode(node))
    computation_node.scope_ops.append(op)
    computation_node.inbound_nodes = layer_to_inbound_nodes.get(node.name, [])
    self._name_to_node[node.name] = computation_node
    self._op_to_node[op.name] = node.name

  def _mark_computation_edges(self):
    for node in self._name_to_node.values():
      for op in node.scope_ops:
        # Remove duplicate input names, although this is rare, but it does exist.
        # We have to make sure that the order of the inputs is the same as the
        # original order in the tf.Operation as each input corresponds to a argument
        # with a different meaning.
        # node {
        #   name: "functional_1/lambda/frame/StridedSlice"
        #   op: "StridedSlice"
        #   input: "args_0"
        #   input: "functional_1/lambda/frame/zeros_like"
        #   input: "functional_1/lambda/frame/concat"
        #   input: "functional_1/lambda/frame/ones_like"
        #   ...
        # }
        inputs = []
        for inp in op.inputs:
          if inp.name not in inputs:
            inputs.append(inp.name)

        for input in inputs:
          op_name = tf_utils.node_name_from_input(input)
          input_node_name = self._op_to_node[op_name]
          # Find input ops that does't belong to this node.
          if input_node_name != node.name:
            input_node = self.node(input_node_name)
            # Avoid appending duplicate names which can cause errors when
            # topologically sorting nodes.
            if input not in node.input_names:
              node.input_names.append(input)
            if input not in input_node.output_names:
              input_node.output_names.append(input)

  def node(self, name):
    return self._name_to_node.get(name, None)

  @property
  def nodes(self):
    return self._name_to_node.values()

  def __str__(self):
    return json.dumps(self.desp(), indent=2, separators=(',', ': '))

  def desp(self):
    nodes_desp = []
    for node in self._name_to_node.values():
      nodes_desp.append(node.desp())
    return nodes_desp

class ComputationNode(object):
  """Intermediate representation of 'computing node' before converting to nndct op"""

  def __init__(self, op):
    # keras.layers.Layer or tf.Operation
    self._op = op

    # If self._op is a tf.Operation object,
    # then scope_ops should be the same as self._op;
    # If self._op is a keras layer, scope_ops should contains all ops
    # generated from the layer.
    self.scope_ops = []

    self.input_names = []
    self.output_names = []

    self.inbound_nodes = []

  def __str__(self):
    return json.dumps(self.desp(), indent=2, separators=(',', ': '))

  def desp(self):
    desp = {}
    desp['name'] = self.name
    desp['type'] = str(self.type)
    desp['scope_ops'] = [op.name for op in self.scope_ops]
    desp['input_names'] = [t for t in self.input_names]
    desp['output_names'] = [t for t in self.output_names]
    return desp

  @property
  def orig_op(self):
    return self._op

  def get_config(self):
    if isinstance(self._op, tf.Operation):
      return tf_utils.parse_attr_proto(self._op.node_def.attr)

    config = self._op.get_config()
    if self.inbound_nodes:
      config['inbound_nodes'] = self.inbound_nodes
    return config

  def get_params(self):
    if isinstance(self._op, tf.Operation):
      return None
    return keras_utils.keras_layer_params(self._op)

  @property
  def name(self):
    return self._op.name

  @property
  def type(self):
    if isinstance(self._op, tf.Operation):
      return self._op.type
    return type(self._op)

def _parent_scope(scope):
  # Given 'model/dense/MatMul', return 'model/dense'.
  return scope.rsplit('/', 1)[0]

def belongs_to_keras_layer(op, scope_to_layer):
  """Get the keras layer the given op is generated from.
  Returns None if op does not belong to any layer.

  Trace back from current scope to parent scope recursively until it reaches
  the outermost scope.
  """

  if not scope_to_layer:
    return None

  layer = None
  scope = op.name
  while True:
    if scope in scope_to_layer:
      layer = scope_to_layer[scope][0]
      break
    parent_scope = _parent_scope(scope)
    # Already to the outtest scope.
    if parent_scope == scope:
      break
    scope = parent_scope

  # Lambda layer is a wrapper, we need to parse ops in the layer individually.
  if type(layer) == tf.keras.layers.Lambda or isinstance(layer, Sequential):
    layer = None
  return layer

def map_layer_to_inbound_nodes(scope_to_layer):
  layer_to_inbound_nodes = {}
  if not scope_to_layer:
    return layer_to_inbound_nodes

  model = None
  # Get a keras model which is a top-level layer.
  for layer, parent_layer in scope_to_layer.values():
    if parent_layer is None:
      model = layer
      break

  if model:
    model_config = model.get_config()
    if 'layers' in model_config:
      layers_config = model_config['layers']
      for config in layers_config:
        if 'inbound_nodes' in config:
          layer_to_inbound_nodes[config['name']] = config['inbound_nodes']
  return layer_to_inbound_nodes

class GraphRefiner(object):

  class RefinerResult(
      collections.namedtuple('RefinerResult', ['refiner', 'message'])):
    pass

  def refine_graph(self, graph):
    raise NotImplementedError

    return graph

  def _msg_for_removing(self, removed_nodes):
    return 'Removed nodes: {}'.format(', '.join(
        [node.name for node in removed_nodes]))

  def _remove_nodes_if(self, graph, cond):
    nodes_to_remove = []
    for node in graph.nodes:
      if cond(node):
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
      graph.remove_node(node)
    return graph, nodes_to_remove

class FoldConst(GraphRefiner):

  def fold_to_dense(self, const_op, dense_op):
    tensor = list(const_op.params.values())[0]
    assert len(tensor.shape) == 2
    dense_op.param['weights'] = tensor
    dense_op.set_config('activation', None)

    dense_op.set_config('units', tensor.shape[0])
    dense_op.attr['in_dim'] = tensor.shape[1]

  def default_fold(self, const_op, op):
    for param, value in const_op.params.items():
      op.set_param(param, tensor_utils.tf_to_nndct(value))

  def refine_graph(self, graph):
    """Fetch the input tensor's value, set it as op's param or attribute
    and remove the original input node.
    """
    fold_map = {OpTypes.DENSE: self.fold_to_dense, OpTypes.BIAS_ADD: None}
    nodes_to_remove = []
    folded_pairs = []
    for node in graph.nodes:
      op = node.op
      if op.type == OpTypes.RESHAPE:
        pass
        #in_tensor = node.input_names[1]
        #print(in_tensor)
        #op.set_config('shape', in_tensor.data.tolist())
        #nodes_to_remove.append(in_tensor.node)
        #folded_pairs.append((in_tensor.node.name, node.name))
      elif op.type in fold_map:
        const_node = None
        for in_node_name in node.in_nodes:
          in_node = graph.node(in_node_name)
          if in_node.op.type == OpTypes.CONST:
            const_node = in_node
            break

        if const_node:
          fold_func = fold_map[op.type]
          if not fold_func:
            fold_func = self.default_fold
          fold_func(const_node.op, op)

          nodes_to_remove.append(const_node)
          folded_pairs.append((const_node.name, node.name))
      else:
        pass

    for node in nodes_to_remove:
      graph.remove_node(node)

    msg = '\n'.join(['Fold {} to {}'.format(p[0], p[1]) for p in folded_pairs])
    return self.RefinerResult('FoldConst', msg)

class FoldBiasRefiner(GraphRefiner):

  def refine_graph(self, graph):
    #biased_ops = ['Conv2D', 'MatMul']
    bias_nodes = []
    folded_pairs = []
    for node in graph.nodes:
      if node.op.type == OpTypes.BIAS_ADD:
        master_node = graph.node(node.in_nodes[0])
        if master_node.op.type == OpTypes.DENSE:
          master_node.op.param['bias'] = list(node.op.params.values())[0]
          master_node.op.set_config('use_bias', True)
        else:
          for param, value in node.op.params.items():
            master_node.op.set_param(param, value)
        bias_nodes.append(node)
        folded_pairs.append((node.name, master_node.name))

    for node in bias_nodes:
      graph.remove_node(node)

    msg = '\n'.join(['Fold {} to {}'.format(p[0], p[1]) for p in folded_pairs])
    return self.RefinerResult('FoldBias', msg)

class RemoveIdentity(GraphRefiner):

  def refine_graph(self, graph):
    graph, removed_nodes = self._remove_nodes_if(
        graph, lambda x: x.op.type == OpTypes.IDENTITY)
    return self.RefinerResult('RemoveIdentity',
                              self._msg_for_removing(removed_nodes))

class RemoveRNNRedundantInput(GraphRefiner):
  """LSTM nodes usually have some redundant inputs, remove all these nodes."""

  def refine_graph(self, graph):
    nodes_to_remove = []
    for node in graph.nodes:
      if node.op.type == OpTypes.LSTM:
        for node_name in node.in_nodes:
          input_node = graph.node(node_name)
          if input_node.op.type in [OpTypes.CONST, OpTypes.LSTM_CELL]:
            nodes_to_remove.append(input_node)

    for node in nodes_to_remove:
      graph.remove_node(node)
    return self.RefinerResult('RemoveRNNRedundantInput',
                              self._msg_for_removing(nodes_to_remove))

class RemoveIsolatedNode(GraphRefiner):

  def refine_graph(self, graph):

    def is_isolated(node):
      return node.num_inputs == 0 and node.num_outputs == 0

    graph, removed_nodes = self._remove_nodes_if(graph, is_isolated)
    return self.RefinerResult('RemoveIsolated',
                              self._msg_for_removing(removed_nodes))

class MergeBidirectionalRNN(GraphRefiner):

  def refine_graph(self, graph):
    nodes_to_remove = []
    for node in graph.nodes:
      if node.op.type == OpTypes.BIDIRECTIONAL_RNN:
        nodes_to_remove.extend(graph.parents(node))

    for node in nodes_to_remove:
      graph.remove_node(node)
    return self.RefinerResult('MergeBidirectional',
                              self._msg_for_removing(nodes_to_remove))

class RenameParamTensor(GraphRefiner):
  """Rename param tensor with a more readable name."""

  def refine_graph(self, graph):
    msg = []
    for node in graph.nodes:
      for param, tensor in node.op.params.items():
        # param either be a string or Enum defined in op's ParamName.
        param_name = param if isinstance(param, str) else param.name.lower()
        new_name = node.name + ':' + param_name
        msg.append('%s -> %s' % (tensor.name, new_name))
        tensor.name = new_name
    return self.RefinerResult('RenameParamTensor', ', '.join(msg))

_FROZEN_FUNC_GRAPH = '.tmp/0_frozen_func_graph.pb'
_OPT_TF_GRAPH = '.tmp/1_opt_tf_graph.pb'
_RAW_NNDCT_GRAPH = '.tmp/2_raw_nndct.pb'
_FINAL_NNDCT_GRAPH = '.tmp/3_final_nndct.pb'
