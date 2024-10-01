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
"""Refine a raw graph to get a normalized graph."""
import collections

from tensorflow.python.util import nest

from tf_nndct.graph import OpTypes
from tf_nndct.utils import logging
from tf_nndct.utils import tf_utils

def run_graph_refining(graph):
  # Executed in sequence.
  refiners = [
      FoldConst, FoldBias, RemoveIdentity, RemoveRNNRedundantInput,
      RemoveIsolatedNode, MergeBidirectionalRNN, RenameParamTensor,
      SetAttrForBinaryOp, RemoveConstantFoldingNode,
      RenoveRedundantTensorBetweenTwoNode
  ]

  for refiner_cls in refiners:
    refiner = refiner_cls()
    result = refiner.refine_graph(graph)
    logging.vlog(
        2, 'Refining pass [{}]: {}'.format(result.refiner, result.message))
  return graph

class GraphRefiner(object):

  class RefinerMessage(
      collections.namedtuple('RefinerMessage', ['refiner', 'message'])):
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

  def refiner_message(self, message, refiner=''):
    if not refiner:
      refiner = self.__class__.__name__
    return self.RefinerMessage(refiner, message)

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
      op.set_param(param, value)

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
    return self.refiner_message(msg)

class FoldBias(GraphRefiner):

  def refine_graph(self, graph):
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
    return self.refiner_message(msg)

class RemoveConstantFoldingNode(GraphRefiner):
  # for layers.Normalization , activations.gelu
  # after get the tf_graph there may generate another node that
  # does not belong to the original model, and with name beginning with 'ConstantFolding'
  # like 'ConstantFolding/net/normalization/truediv_recip' &
  # 'ConstantFolding/net/gelu/Gelu/truediv_recip'
  # we need the rm these nodes and del the tensor connected with them,
  # usually ConstantFolding node connects with one node,
  # and the tensor may have a circle(two tensor) or single-direction(one tensor)
  def refine_graph(self, graph):
    need_rm_tensor = []
    need_rm_node = []
    msg = ""
    for node in graph.nodes:
      # can not judge from node.op.type
      if node.name.startswith('ConstantFolding/'):
        msg += "for node:\t" + node.name
        for output_tensor in node._out_tensors:
          need_rm_tensor.append(output_tensor)
          node.remove_output(output_tensor)
          msg += " rm out_t:\t" + output_tensor.name
        for input_tensor in node._in_tensors:
          node.remove_input(input_tensor)
          msg += " rm in_t:\t" + input_tensor.name
          source_node = input_tensor.producer.name
          graph.node(source_node).remove_output(input_tensor)
          msg += " and rm out_t:\t" + input_tensor.name + " from: " + source_node
        # then the constantfold_node will be a isolate node
        need_rm_node.append(node)
        msg += "\n"

    # the we rm input tensor for each node from need_rm_tensor
    for rm_tensor in need_rm_tensor:
      for node in graph.nodes:
        if node.is_consuming(rm_tensor):
          node.remove_input(rm_tensor)
          msg += " rm in_tensor:\t" + rm_tensor.name + " for: " + node.name + "\n"
    for node in need_rm_node:
      graph.remove_node(node)
    return self.refiner_message(msg)

class RenoveRedundantTensorBetweenTwoNode(GraphRefiner):
  # condition 1:
  # if layers.Rescaling -> layers.Normalization
  # Rescaling = x * scale + offset
  # Normalization = (x - mean) / sqrt(var)
  # Rescaling->Normalization = (x * scale + offset - mean) / sqrt(var)
  # so after getting the tf_graph the Normalization will receive two tensors from rescale
  # scale and (offset - mean) we need rm one tensor to let the net tope correct
  def refine_graph(self, graph):

    def is_rescaling_to_normalization(node):
      # if Rescaling -> Normalization and
      # tensor in these two node are 2
      if node.op.type == OpTypes.RESCALING and \
        len(node.out_nodes) == 1 and \
        graph.node(node.out_nodes[0]).op.type == OpTypes.NORM and \
        node.num_outputs >= 2:
        return True
      return False

    need_rm_tensor_node = []
    for node in graph.nodes:
      if is_rescaling_to_normalization(node):
        need_rm_tensor_node.append(node)

    # only remain one tensor
    msg = ""
    for node in need_rm_tensor_node:
      child_node = graph.node(node.out_nodes[0])
      msg += "\n process node between {}--and--{}".\
        format(node.name, child_node.name)
      for tensor in node._out_tensors[1:]:
        node.remove_output(tensor)
        child_node.remove_input(tensor)
        msg += "\t rm tensor:\t{}".format(tensor.name)
    return self.refiner_message(msg)

class RemoveIdentity(GraphRefiner):

  def refine_graph(self, graph):
    nodes_to_remove = []
    for node in graph.nodes:
      if node.op.type == OpTypes.IDENTITY:
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
      # Graph's structured_output_tensors are output tensors from leaf
      # Identity nodes. We need to update the structured_output_tensors
      # when these Identity nodes are deleted.
      # For example, the orginal graph is as follows:
      # Dense(dense:0) -> Identity(dense/linear:0) -> Identity(No output tensor)
      # The output tensor is "Identity:0". After the two Identity nodes
      # are removed, the output tensor should be updated to "dense:0".
      output_tensors = []
      for tensor in nest.flatten(graph.structured_output_tensors):
        # As the output tensors does not exist in graph, so we can't
        # get the output node by tensor's producer, like:
        # node = graph.tensor(tensor.name).producer
        node_name = tf_utils.node_name_from_input(tensor.name)
        if node.name == node_name:
          if node.op.type != OpTypes.IDENTITY:
            raise RuntimeError(
                'The leaf tensors must be generated from Identity node.')
          output_tensors.append(node.in_tensors[0])
        else:
          output_tensors.append(tensor)
      graph.structured_output_tensors = nest.pack_sequence_as(
          graph.structured_output_tensors, output_tensors)
      graph.remove_node(node)

    return self.refiner_message(self._msg_for_removing(nodes_to_remove))

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
    return self.refiner_message(self._msg_for_removing(nodes_to_remove))

class RemoveIsolatedNode(GraphRefiner):

  def refine_graph(self, graph):

    def is_isolated(node):
      return node.num_inputs == 0 and node.num_outputs == 0

    graph, removed_nodes = self._remove_nodes_if(graph, is_isolated)
    return self.refiner_message(self._msg_for_removing(removed_nodes))

class MergeBidirectionalRNN(GraphRefiner):

  def refine_graph(self, graph):
    nodes_to_remove = []
    for node in graph.nodes:
      if node.op.type == OpTypes.BIDIRECTIONAL_RNN:
        nodes_to_remove.extend(graph.parents(node))

    for node in nodes_to_remove:
      graph.remove_node(node)

    for node in graph.nodes:
      if node.op.type != OpTypes.BIDIRECTIONAL_RNN:
        continue
      in_tensors = node.in_tensors
      assert in_tensors[0].name == in_tensors[1].name
      node.remove_input(in_tensors[1])

    return self.refiner_message(self._msg_for_removing(nodes_to_remove))

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
    return self.refiner_message(', '.join(msg))

class SetAttrForBinaryOp(GraphRefiner):
  """Set 'input' and 'other' for binary operations. These two attrs are used
  for exporting to xir.
  """

  def refine_graph(self, graph):
    refined_ops = []
    binary_ops = [OpTypes.MULTIPLY]
    for node in graph.nodes:
      if node.op.type not in binary_ops:
        continue
      assert len(
          node.in_tensors
      ) == 2, 'Binary operation should have 2 inputs, but got {}'.format(
          len(node.in_tensors))
      node.op.attr['input'] = node.in_tensors[0]
      node.op.attr['other'] = node.in_tensors[1]
      refined_ops.append(node.name)
    return self.refiner_message('{}'.format(refined_ops))
