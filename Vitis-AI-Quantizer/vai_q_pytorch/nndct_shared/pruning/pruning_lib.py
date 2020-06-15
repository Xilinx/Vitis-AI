from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pickle
import six

from nndct_shared.base.key_names import NNDCT_OP
from nndct_shared.pruning import node_group as node_group_lib
from nndct_shared.pruning import logging
from nndct_shared.utils import registry

_op_modifier = registry.Registry("Pruning Modifier Functions")

class RegisterOpModifier(object):
  """A decorator for registering the modification function for an op.
  This decorator can be defined for an op type so that it can infer the op's
  NodePruningInfo and modify its attribute by existing pruning result.

  If a modifier for an op is registered multiple times, a KeyError will be
  raised.
  For example, you can define a new modifier for a Conv2D operation
  by placing this in your code:
  ```python
  @pruning.RegisterOpModifier("Conv2D")
  def _modify_conv2d(graph, node, pruning_res):
    ...
  ```
  Then in client code you can retrieve the value by making this call:
  ```python
  pruning_lib.modify_node_by_pruning(graph, node, pruning_res)
  ```
  """

  def __init__(self, op_type):
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string.")
    if "," in op_type:
      raise TypeError("op_type must not contain a comma.")
    self._op_type = op_type

  def __call__(self, f):
    """Registers "f" as the statistics function for "op_type"."""
    # TODO(yuwang): Validate function signature before it is registered.
    _op_modifier.register(f, self._op_type)
    return f

def _set_input_pruning_by_prev_output(node, pruning_res):
  node_pruning = pruning_res[node.name]
  input_pruning = pruning_res[node.in_nodes[0]]
  node_pruning.removed_inputs = input_pruning.removed_outputs
  node_pruning.in_dim = input_pruning.out_dim
  return node_pruning

def _set_output_pruning_by_input(node, pruning_res):
  node_pruning = pruning_res[node.name]
  node_pruning.removed_outputs = node_pruning.removed_inputs
  node_pruning.out_dim = node_pruning.in_dim
  return node_pruning

def inherit_node_pruning(node, pruning_res):
  _set_input_pruning_by_prev_output(node, pruning_res)
  _set_output_pruning_by_input(node, pruning_res)

def _is_depthwise(op):
  if op.type == NNDCT_OP.DEPTHWISE_CONV2D:
    return True

  if op.type != NNDCT_OP.CONV2D:
    return False

  out_channels = op.attr['out_dim']
  in_channels = op.attr['in_dim']
  group = op.attr['group']
  return group == in_channels and out_channels % in_channels == 0

def _modify_depthwise_conv2d(graph, node, pruning_res):
  node_pruning = _set_input_pruning_by_prev_output(node, pruning_res)
  if not node_pruning.in_dim:
    return

  dw_multiplier = node.op.attr['out_dim'] // node.op.attr['in_dim']
  removed_outputs = []
  for c in node_pruning.removed_inputs:
    for k in range(dw_multiplier):
      removed_outputs.append(c * dw_multiplier + k)
  # The shape of dw conv weights is [out_channels, 1, height, width] as
  # in_channels // groups always equals to 1, so there are no in_channels
  # to be removed.
  node_pruning.removed_inputs = []
  node_pruning.removed_outputs = removed_outputs
  node_pruning.out_dim = node_pruning.in_dim * dw_multiplier

  node.op.attr['in_dim'] = node_pruning.in_dim
  node.op.attr['group'] = node_pruning.in_dim
  node.op.attr['out_dim'] = node_pruning.out_dim

@RegisterOpModifier(NNDCT_OP.DEPTHWISE_CONV2D)
def modify_depthwise_conv2d(graph, node, pruning_res):
  _modify_depthwise_conv2d(graph, node, pruning_res)

@RegisterOpModifier(NNDCT_OP.CONV2D)
def modify_conv2d(graph, node, pruning_res):
  # In pytorch, dw conv is repesented by conv2d with groups == in_channels and
  # out_channels == K * in_channels, where K is a positive integer.
  if _is_depthwise(node.op):
    _modify_depthwise_conv2d(graph, node, pruning_res)
    return

  node_pruning = _set_input_pruning_by_prev_output(node, pruning_res)
  if node_pruning.in_dim:
    node.op.attr['in_dim'] = node_pruning.in_dim

  if node_pruning.out_dim:
    node.op.attr['out_dim'] = node_pruning.out_dim

@RegisterOpModifier(NNDCT_OP.ADAPTIVEAVGPOOL2D)
def modify_adaptive_avg_pool2d(graph, node, pruning_res):
  output_size = node.op.attr['global']
  if output_size == [1, 1] or output_size == (1, 1) or output_size == 1:
    inherit_node_pruning(node, pruning_res)
  else:
    raise NotImplementedError(
        "output_size={} not supported for AdaptiveAvgPool2d".format(
            output_size))

@RegisterOpModifier(NNDCT_OP.DENSE)
def modify_dense(graph, node, pruning_res):
  node_pruning = _set_input_pruning_by_prev_output(node, pruning_res)
  if node_pruning.in_dim:
    node.op.attr['in_dim'] = node_pruning.in_dim

@RegisterOpModifier(NNDCT_OP.RESHAPE)
def modify_reshape(graph, node, pruning_res):
  node_pruning = pruning_res[node.name]
  input_pruning = pruning_res[node.in_nodes[0]]
  assert input_pruning.out_dim, "Unknown out_dim of reshape\'s input node"

  shape = node.node_attr('shape')
  assert len(shape) == 2, "Node {} has an unexpected shape {}".format(
      node.name, shape)
  orig_out_depth = len(input_pruning.removed_outputs) + input_pruning.out_dim
  spatial_size = shape[-1] // orig_out_depth

  # Two data formats:
  # [-1, 7, 7, 64(32)] => [-1, 3136(1518)]
  # [-1, 32(16), 5, 5] => [-1, 800(400)]
  removed_outputs = []
  input_node = graph.node(node.in_nodes[0])
  layout = input_node.node_attr('layout')
  if layout == 'NHWC':
    for s in range(spatial_size):
      for c in input_pruning.removed_outputs:
        removed_outputs.append(s * spatial_size + c)
  else:
    for c in input_pruning.removed_outputs:
      for s in range(spatial_size):
        removed_outputs.append(c * spatial_size + s)

  node_pruning.removed_outputs = removed_outputs
  node_pruning.out_dim = spatial_size * input_pruning.out_dim

  pruned_shape = list(shape)
  pruned_shape[-1] = node_pruning.out_dim
  node.op.attr['shape'] = pruned_shape

@RegisterOpModifier(NNDCT_OP.CONCAT)
def modify_concat(graph, node, pruning_res):
  node_pruning = pruning_res[node.name]
  cur_offset = 0
  out_dim = 0
  removed_outputs = []
  for input_tensor in node.in_tensors:
    input_node = input_tensor.node
    input_pruning = pruning_res[input_node.name]
    for ro in input_pruning.removed_outputs:
      removed_outputs.append(ro + cur_offset)
    out_dim += input_pruning.out_dim
    cur_offset += len(input_pruning.removed_outputs) + input_pruning.out_dim
  node_pruning.removed_outputs = removed_outputs
  node_pruning.out_dim = out_dim

@RegisterOpModifier(NNDCT_OP.BATCH_NORM)
def modify_batchnorm(graph, node, pruning_res):
  node_pruning = pruning_res[node.name]
  input_pruning = pruning_res[node.in_nodes[0]]

  node_pruning.removed_inputs = input_pruning.removed_outputs
  node_pruning.in_dim = input_pruning.out_dim

  node_pruning.removed_outputs = input_pruning.removed_outputs
  node_pruning.out_dim = input_pruning.out_dim

  if node_pruning.out_dim:
    node.op.attr['out_dim'] = node_pruning.out_dim

def update_node_by_pruning(graph, node, pruning_res):
  """Looks up the node's modification function in the registry and calls it.
  This function takes a NndctGraph object, a NndctNode from it, and the
  dictionary of PruningInfo and if there's an associated modification method,
  calls it. If no function has been registered for the particular op type,
  a general fucntion will be called which simply set current node's pruning info
  and do not update the node's attribute.
  statistics object
  Args:
    graph: A NndctGraph that the pruning is performed on.
    node: A NndctNode describing the operator.
    pruning_res: A dictionary of `NodePruningInfo`.
  """
  op_type = node.op.type
  if op_type in _op_modifier.list():
    mod_func = _op_modifier.lookup(op_type)
    mod_func(graph, node, pruning_res)
  else:
    inherit_node_pruning(node, pruning_res)

class GroupSparsity(
    collections.namedtuple('GroupSparsity', ['nodes', 'sparsity'])):
  """The sparsity after pruning for each group."""

class PruningSpec(object):
  """Specification indicates how to prune the network."""

  def __init__(self, channel_batch=2, groups=None):
    self._channel_batch = 2
    self._groups = groups if groups else []
    self._node_to_group = {}

  def __str__(self):
    return "PruningSpec(groups=[%s], channel_batch=%d)" % (", ".join(
        [str(group) for group in self._groups]), self._channel_batch)

  @property
  def channel_batch(self):
    return self._channel_batch

  @channel_batch.setter
  def channel_batch(self, channel_batch):
    if channel_batch <= 0:
      raise ValueError("'channel_batch' must be positive.")
    self._channel_batch = channel_batch

  @property
  def groups(self):
    return self._groups

  def add_group(self, group):
    if not isinstance(group, GroupSparsity):
      raise ValueError("'group' must be a dictionary.")

    self._groups.append(group)
    for node_name in group.nodes:
      self._node_to_group[node_name] = group

  def group(self, node_name):
    return self._node_to_group.get(node_name, None)

class GroupSens(collections.namedtuple('NetSens', ['nodes', 'vals'])):
  pass

class NetSens(object):
  """The sensitivity results of the network generated by model analysis."""

  def __init__(self):
    self.groups = []

  def load(self, fp):
    self.groups = pickle.load(fp)

  def dump(self, fp):
    # TODO(yuwang): Use pickle instead of plain text.
    pickle.dump(self.groups, fp)

  def __repr__(self):
    strs = []
    for group in self.groups:
      strs.append(repr(group))
    return "\n".join(strs)

def read_sens(filename):
  net_sens = NetSens()
  with open(filename, 'rb') as f:
    net_sens.load(f)
    return net_sens

def write_sens(net_sens, filename):
  with open(filename, 'wb') as f:
    net_sens.dump(f)

def sens_path(model):
  # md5 = hashlib.md5()
  # md5.update()
  # md5.hexdigest()
  return '.ana'

class NodePruningInfo:
  """A data class that saves the pruning info of the `Node` object."""

  def __init__(self,
               node_name,
               removed_outputs=None,
               out_dim=None,
               removed_inputs=None,
               in_dim=None,
               framework_weights=None,
               master=False):
    self.node_name = node_name
    self.removed_outputs = removed_outputs
    self.out_dim = out_dim
    self.removed_inputs = removed_inputs
    self.in_dim = in_dim
    #self.framework_weights = framework_weights if framework_weights else []
    self.master = master

  def __str__(self):
    return "NodePruningInfo({}, in_dim={}, out_dim={})".format(
        self.node_name, self.in_dim, self.out_dim)

  def __repr__(self):
    return ("NodePruningInfo<name={}, removed_inputs={}, in_dim={}, "
            "removed_outputs={}, out_dim={}>").format(self.node_name,
                                                      self.removed_inputs,
                                                      self.in_dim,
                                                      self.removed_outputs,
                                                      self.out_dim)

def group_nodes(graph, nodes_to_exclude=[]):
  """Divide conv2d nodes into different groups.
  The nodes that connected with each other by elementwise operation
  will be divided into a group.
  """
  #TODO(yuwang): Check if graph is valid.
  node_group = node_group_lib.NodeGroup()
  for node in graph.nodes:
    if node.op.type == NNDCT_OP.CONV2D:
      node_group.add_node(node.name)

  for node in graph.nodes:
    if node.op.type != NNDCT_OP.ADD:
      continue
    eltwise_inputs = []
    for name in node.in_nodes:
      input_node = graph.node(name)
      # Depthwise conv must be treated as a slave node.
      if input_node.op.type == NNDCT_OP.CONV2D and not _is_depthwise(
          input_node.op):
        eltwise_inputs.append(name)
      else:
        ancestor = find_node_ancestor(graph, input_node, [NNDCT_OP.CONV2D],
                                      [NNDCT_OP.CONCAT])
        if ancestor and not _is_depthwise(input_node.op):
          eltwise_inputs.append(ancestor.name)
    if len(eltwise_inputs) < 2:
      continue
    logging.vlog(2, "Union ({}, {})".format(eltwise_inputs[0],
                                            eltwise_inputs[1]))
    node_group.union(eltwise_inputs[0], eltwise_inputs[1])

  all_groups = node_group.groups()
  groups = []
  for group in all_groups:
    skip = False
    for node in nodes_to_exclude:
      if node in group:
        skip = True
        break
    if not skip:
      groups.append(group)
  return groups

def find_node_ancestor(graph, node, target_ops, barrier_ops=[]):
  visited = set()
  queue = []

  # start from input nodes
  for input_name in node.in_nodes:
    queue.append(input_name)

  ancestor = None
  while len(queue) != 0:
    node_name = queue.pop(0)
    node = graph.node(node_name)
    if node.op.type in target_ops:
      ancestor = node
      break
    elif node.op.type in barrier_ops:
      continue
    else:
      for input_name in node.in_nodes:
        if input_name not in visited:
          queue.append(input_name)
          visited.add(input_name)
  return ancestor

def get_sparsity_by_threshold(net_sens, threshold, excludes=[]):
  net_sparsity = []
  for group in net_sens.groups:
    skip = False
    for node in group.nodes:
      for exclude in excludes:
        if node == exclude:
          skip = True
          break

    if skip:
      continue

    num_exps = len(group.vals)
    for exp in range(num_exps - 1, 0, -1):
      if abs(group.vals[exp] - group.vals[0]) < threshold:
        break

    if exp > 0:
      net_sparsity.append(GroupSparsity(group.nodes, exp * 0.1))

  return net_sparsity
