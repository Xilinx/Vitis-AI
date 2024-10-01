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
"""Parse a keras model to nndct graph."""
import json
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.core.protobuf import config_pb2
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations

from tf_nndct.graph import OpTypes
from tf_nndct.graph import converter
from tf_nndct.graph import ops
from tf_nndct.graph import refiner
from tf_nndct.graph import utils
from tf_nndct.utils import generic_utils
from tf_nndct.utils import keras_utils
from tf_nndct.utils import logging
from tf_nndct.utils import tf_utils

keras = tf.keras

_EXPORT_DIR = '.vai_tf'
_FROZEN_FUNC_GRAPH = '0_frozen_func_graph.pb'
_OPT_TF_GRAPH = '1_opt_tf_graph.pb'
_RAW_NNDCT_GRAPH = '2_raw_nndct.pb'
_FINAL_NNDCT_GRAPH = '3_final_nndct.pb'

def from_keras_model(model, input_signature=None):
  """Trace model call to get a func graph and convert that func graph
    to nndct graph.
  """

  logging.vlog(1, 'input_signature: {}'.format(input_signature))

  # TODO haoliang
  # Note1: Support `Functional API` format Model, subclassing the `Model` may cause errors
  # Two ways to instantiate a `Model`: https://github.com/keras-team/keras/blob/v2.10.0/keras/engine/training.py#L69
  func_graph = get_func_graph(model, input_signature)

  scope_to_layer = map_scope_to_layer(model)
  logging.vlog(
      1, 'scope_name: (layer, parent_layer)\n{}'.format('\n'.join(
          [f'{key}: {value}' for key, value in scope_to_layer.items()])))

  graph = get_raw_graph(func_graph, scope_to_layer)
  graph = run_graph_refining(graph)
  logging.vlog(2, 'NndctGraph before sorting:\n{}'.format(graph))

  graph = utils.topological_sort(graph)

  graph.name = model.name
  graph.data_format = keras_utils.data_format()
  logging.vlog(2, 'Final parsed graph:\n{}'.format(graph))
  return graph

def get_func_graph(model, input_signature=None):
  # TODO(yuwang) Use trace_model_call from keras function directly.
  #from tensorflow.python.keras.saving import saving_utils
  #func = saving_utils.trace_model_call(model, input_signature)

  concrete_func = keras_utils.trace_model_call(model, [input_signature])

  frozen_func = tf_utils.convert_to_constants(
      concrete_func, lower_control_flow=False)
  graph_def = frozen_func.graph.as_graph_def()
  utils.maybe_export_graph(
      os.path.join(_EXPORT_DIR, _FROZEN_FUNC_GRAPH), graph_def)

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
  utils.maybe_export_graph(os.path.join(_EXPORT_DIR, _OPT_TF_GRAPH), graph_def)

  with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(graph_def, name='')

  func_graph = concrete_func.graph
  return (tf_graph, func_graph.structured_input_signature,
          func_graph.structured_outputs)

def map_scope_to_layer(layer, scope='', parent=None):
  if not isinstance(layer, keras.layers.Layer):
    return {}

  scope_to_layer = {}

  layer_scope = "/".join([scope, layer.name]) if scope else layer.name
  scope_to_layer[layer_scope] = (layer, parent)

  # There is no _gather_unique_layers in earlier TF.
  # layers = layer._gather_unique_layers()
  layers = keras_utils.get_layers(layer)
  for sub_layer in layers:
    layer_dict = map_scope_to_layer(sub_layer, layer_scope, layer)
    scope_to_layer.update(layer_dict)

  return scope_to_layer

def get_raw_graph(func_graph, scope_to_layer=None):
  # op_name => Node name
  tf_graph, input_signature, structured_output_tensors = func_graph

  computation_graph = ComputationGraph.from_tf_graph(tf_graph, scope_to_layer)
  logging.vlog(2, 'ComputationGraph\n {}'.format(computation_graph))

  # Parse computation nodes to nndct nodes.
  nndct_nodes = []
  for node in computation_graph.nodes:
    nndct_nodes.extend(converter.convert(node))

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
  #output_tensors = []
  #for tensor in nest.flatten(structured_output_tensors):
  #  # The output tensors does not exist in graph, so we can't get the output
  #  # node by tensor's producer, like:
  #  # node = graph.tensor(tf_tensor.name).producer
  #  node_name = tf_utils.node_name_from_input(tensor.name)
  #  node = graph.node(node_name)
  #  assert node.op.type == OpTypes.IDENTITY
  #  output_tensors.append(node.in_tensors[0])
  #output_tensors = nest.pack_sequence_as(structured_output_tensors,
  #                                       output_tensors)

  # Get args part from input_signature (args, kwargs)
  graph.input_signature = input_signature[0]
  graph.structured_output_tensors = structured_output_tensors
  utils.maybe_export_graph(os.path.join(_EXPORT_DIR, _RAW_NNDCT_GRAPH), graph)

  return graph

def run_graph_refining(graph):
  graph = refiner.run_graph_refining(graph)
  utils.maybe_export_graph(os.path.join(_EXPORT_DIR, _FINAL_NNDCT_GRAPH), graph)
  return graph

class ComputationGraph(object):

  def __init__(self):

    self._name_to_node = {}
    self._op_to_node = {}

  def add_op(self, op, scope_to_layer, layer_inbound_nodes):
    # If an operation belongs to a keras layer, we add this op to the layer's
    # scope ops; Otherwise, we treat the op as a standalone computation node.
    layer = belongs_to_keras_layer(op, scope_to_layer)
    # tf.keras.layers.Layer or tf.Operation as a node
    op_obj = layer if layer else op
    node = self._name_to_node.get(op_obj.name, Node(op_obj))
    node.scope_ops.append(op)
    node.inbound_nodes = layer_inbound_nodes.get(op_obj.name, [])
    self._name_to_node[node.name] = node
    self._op_to_node[op.name] = node.name

  def connect_node_by_scopes(self):
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

  @classmethod
  def from_tf_graph(cls, tf_graph, scope_to_layer=None):
    layer_inbound_nodes = get_layer_inbound_nodes(
        list(scope_to_layer.values())) if scope_to_layer else {}

    graph = cls()
    for op in tf_graph.get_operations():
      graph.add_op(op, scope_to_layer, layer_inbound_nodes)

    graph.connect_node_by_scopes()
    return graph

  def node(self, name):
    if name not in self._name_to_node:
      raise ValueError('No such node in graph: {}'.format(name))
    return self._name_to_node[name]

  @property
  def nodes(self):
    return list(self._name_to_node.values())

  def __str__(self):
    return json.dumps(self.desp(), indent=2, separators=(',', ': '))

  def desp(self):
    nodes_desp = []
    for node in self._name_to_node.values():
      nodes_desp.append(node.desp())
    return nodes_desp

class Node(object):
  """A computing node is a representation of a keras layer or tf.Operation."""

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
  def op(self):
    return self._op

  def get_config(self):
    if isinstance(self._op, tf.Operation):
      return tf_utils.parse_attr_proto(self._op.node_def.attr)
    else:
      return self._op.get_config()

  def get_params(self):
    if isinstance(self._op, tf.Operation):
      return None
    return keras_utils.get_named_weights(self._op)

  @property
  def name(self):
    return self._op.name

  @property
  def type(self):
    if isinstance(self._op, tf.Operation):
      return self._op.type
    return type(self._op)

  @property
  def op_type_name(self):
    return self._op.__class__.__name__

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
  if type(layer) == keras.layers.Lambda or isinstance(layer, keras.Sequential):
    layer = None
  return layer

def get_layer_inbound_nodes(layer_parent_pairs):
  """Get layer's inbound nodes.

  The config of a layer does not include connectivity information,
  nor the layer class name. These are handled by keras.Model.
  So we extract them from model's config and associate them to the
  corresponding layer.
  """
  layer_inbound_nodes = {}
  model = None
  # Get a keras model which is a top-level layer.
  for layer, parent_layer in layer_parent_pairs:
    if parent_layer is None:
      model = layer
      break

  if getattr(model, '_is_graph_network', None):
    # Only graph network has get_config.
    model_config = model.get_config()
    logging.vlog(4, 'model_config: {}'.format(model_config))

    if 'layers' in model_config:
      layers_config = model_config['layers']
      for config in layers_config:
        if 'inbound_nodes' in config:
          layer_inbound_nodes[config['name']] = config['inbound_nodes']
  return layer_inbound_nodes
