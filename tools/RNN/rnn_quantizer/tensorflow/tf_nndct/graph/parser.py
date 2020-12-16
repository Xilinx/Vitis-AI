

#
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
  #input_signature = [tf.TensorSpec(shape=batch_input_shape, dtype=dtype)]
  if not generic_utils.is_list_or_tuple(input_signature):
    input_signature = generic_utils.to_list(input_signature)

  flat_input_signature = nest.flatten(input_signature)
  batch_input_signature = []
  for signature in flat_input_signature:
    batch_input_signature.append(
        tf.TensorSpec(shape=(1,) + signature.shape, dtype=signature.dtype))
  batch_input_signature = nest.pack_sequence_as(input_signature,
                                                batch_input_signature)

  func_graph = get_func_graph(model, batch_input_signature)

  scope_to_layer = map_scope_to_layer(model, '')
  logging.vlog(1, 'scope_to_layer:\n{}'.format(scope_to_layer))

  graph = parse_to_graph(func_graph, scope_to_layer)
  graph.name = model.name
  return graph

def get_func_graph(model, input_signature=None, *args, **kwargs):
  func = keras_utils.trace_model_call(model, input_signature)
  concrete_func = func.get_concrete_function(*args, **kwargs)

  frozen_func = tf_utils.convert_to_constants(
      concrete_func, lower_control_flow=False)
  graph_def = frozen_func.graph.as_graph_def()
  utils.maybe_export_graph(_BEFORE_OPT_GRAPH, graph_def)

  input_tensors = [
      tensor for tensor in frozen_func.inputs
      if tensor.dtype != tf.dtypes.resource
  ]
  output_tensors = frozen_func.outputs

  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  #rewrite_options.constant_folding = rewrite_options.ON
  rewrite_options.optimizers.append('constfold')
  graph_def = _run_graph_optimizations(
      graph_def,
      input_tensors,
      output_tensors,
      config=config,
      graph=frozen_func.graph)
  utils.maybe_export_graph(_FINAL_TRACED_GRAPH, graph_def)

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

  # There is no _gather_unique_layers old version.
  # layers = layer._gather_unique_layers()
  layers = keras_utils.get_layers(layer)
  for sub_layer in layers:
    layer_dict = map_scope_to_layer(sub_layer, layer_scope, layer)
    scope_to_layer.update(layer_dict)

  return scope_to_layer

def parse_to_graph(func_graph, scope_to_layer=None):
  # op_name => ComputationNode name
  tf_graph, input_signature, structured_output_tensors = func_graph
  op_to_node = {}
  nodes = {}
  for op in tf_graph.get_operations():
    layer = None
    if scope_to_layer:
      layer = belongs_to_keras_layer(op, scope_to_layer)
    # keras.layers or tf.Operation as a node
    node = layer if layer else op
    if node.name not in nodes:
      nodes[node.name] = ComputationNode(node)
    nodes[node.name].scope_ops.append(op)
    op_to_node[op.name] = node.name

  mark_computation_edges(nodes, op_to_node)
  for node in nodes.values():
    logging.vlog(2, 'ComputationNode\n {}'.format(str(node)))

  # Parse computation nodes to nndct nodes.
  nndct_nodes = []
  for node in nodes.values():
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

  logging.vlog(2, str(graph))
  logging.vlog(2, 'input_signature:{}'.format(input_signature))
  logging.vlog(2, 'output_tensors:{}'.format(output_tensors))
  return graph

def run_graph_refining(graph):
  # Executed in sequence.
  refiners = [
        FoldConstRefiner,
        FoldBiasRefiner,
        RemoveIdentityRefiner,
        RemoveConstRefiner,
        RemoveIsolatedRefiner,
        MergeBidirectionalRefiner,
        RenameParamTensorRefiner,
  ]

  for refiner_cls in refiners:
    refiner = refiner_cls()
    result = refiner.refine_graph(graph)
    logging.vlog(
        2, 'Result of refining pass [{}]: {}'.format(result.refiner,
                                                     result.message))
  return graph

class ComputationNode(object):
  """Intermediate representation of 'computing node' before converting to nndct op"""

  def __init__(self, op):
    # keras.layers/tf.Operation
    self._op = op

    # If self._op is a tf.Operation object,
    # then scope_ops should be the same as self._op;
    # If self._op is a keras layer, scope_ops should contains all ops
    # generated from the layer.
    self.scope_ops = []

    self.input_names = []
    self.output_names = []

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
  def raw_op(self):
    return self._op

  def get_attrs(self):
    if isinstance(self._op, tf.Operation):
      return tf_utils.parse_attr_proto(self._op.node_def.attr)
    return self._op.get_config()

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
  scope = op.name
  while True:
    if scope in scope_to_layer:
      return scope_to_layer[scope][0]
    parent_scope = _parent_scope(scope)
    # Already to the outtest scope.
    if parent_scope == scope:
      break
    scope = parent_scope
  return None

def mark_computation_edges(nodes, op_to_node):
  for node in nodes.values():
    for op in node.scope_ops:
      # Remove duplicate elements which is very rare, but it does exist.
      inputs = {input.name for input in op.inputs}
      #inputs = set(inputs)

      for input in inputs:
        op_name = tf_utils.node_name_from_input(input)
        input_node_name = op_to_node[op_name]
        # Find input ops that does't belong to this node.
        if input_node_name != node.name:
          input_node = nodes[input_node_name]
          node.input_names.append(input)
          if input not in input_node.output_names:
            input_node.output_names.append(input)

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

class FoldConstRefiner(GraphRefiner):

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
        in_tensor = node.input_names[1]
        op.set_config('shape', in_tensor.data.tolist())
        nodes_to_remove.append(in_tensor.node)
        folded_pairs.append((in_tensor.node.name, node.name))
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

class RemoveIdentityRefiner(GraphRefiner):
  def refine_graph(self, graph):
    graph, removed_nodes = self._remove_nodes_if(
        graph, lambda x: x.op.type == OpTypes.IDENTITY)
    return self.RefinerResult('RemoveIdentity',
                              self._msg_for_removing(removed_nodes))

class RemoveConstRefiner(GraphRefiner):
  def refine_graph(self, graph):
    graph, removed_nodes = self._remove_nodes_if(
        graph, lambda x: x.op.type == OpTypes.CONST)
    return self.RefinerResult('RemoveConst',
                              self._msg_for_removing(removed_nodes))

class RemoveIsolatedRefiner(GraphRefiner):
  def refine_graph(self, graph):

    def is_isolated(node):
      return node.num_inputs == 0 and node.num_outputs == 0

    graph, removed_nodes = self._remove_nodes_if(graph, is_isolated)
    return self.RefinerResult('RemoveIsolated',
                              self._msg_for_removing(removed_nodes))

class MergeBidirectionalRefiner(GraphRefiner):
  def refine_graph(self, graph):
    nodes_to_remove = []
    for node in graph.nodes:
      if node.op.type == OpTypes.BIDIRECTIONAL_RNN:
        nodes_to_remove.extend(graph.parents(node))

    for node in nodes_to_remove:
      graph.remove_node(node)
    return self.RefinerResult('MergeBidirectional',
                              self._msg_for_removing(nodes_to_remove))

class RenameParamTensorRefiner(GraphRefiner):
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
    return self.RefinerResult('RenameParamTensorRefiner', ', '.join(msg))

_BEFORE_OPT_GRAPH = '/tmp/1_before_opt.pb'
_FINAL_TRACED_GRAPH = '/tmp/2_final_traced.pb'
_RAW_NNDCT_GRAPH = '/tmp/raw_nndct.pb'
_FINAL_NNDCT_GRAPH = '/tmp/final_nndct.pb'
