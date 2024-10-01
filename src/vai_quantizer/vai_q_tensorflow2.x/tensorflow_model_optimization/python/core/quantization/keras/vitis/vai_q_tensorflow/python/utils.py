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

import os
from copy import deepcopy
import json
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
import numpy as np

FLAGS = None

# error code
ERROR_CODE_PREFIX="QUANTIZE_TF1"

INVALID_INPUT = "[{}_INVALID_INPUT]".format(ERROR_CODE_PREFIX)
INVALID_BITWITH = "[{}_INVALID_BITWIDTH]".format(ERROR_CODE_PREFIX)
INVALID_METHOD = "[{}_INVALID_METHOD]".format(ERROR_CODE_PREFIX)
LENGHT_MISSMATCH = "[{}_LENGTH_MISMATCH]".format(ERROR_CODE_PREFIX)
INVALID_INPUT_FN = "[{}_INVALID_INPUT_FN]".format(ERROR_CODE_PREFIX)
INVALID_TARGET_DTYPE = "[{}_INVALID_TARGET_DTYPE]".format(ERROR_CODE_PREFIX)
UNSUPPORTED_OP = "[{}_UNSUPPORTED_OP]".format(ERROR_CODE_PREFIX)


NOT_FOUND_MESSAGE = "[Not found]"
INVALID_PARAM_MESSAGE = "[Invalid parameter]"
FAIL_IMPORT_MESSAGE = "[Fail to import]"
UNSUPPORTED_OP_MESSAGE = "[Unsupported Op type]"


_string2dtype = {"tf.float32" : tf.float32,
        "float32": tf.float32,
        "float": tf.float32,
        "int32": tf.int32,
        "int64": tf.int64,
        "bool": tf.bool}
def get_name_to_nodes_map(graph_def):
  name_to_nodes = {}
  for node in graph_def.node:
    name_to_nodes[node.name] = node
  return name_to_nodes

def check_node_names(graph_def, node_names):
  """Check if graph_def has node names"""
  if not isinstance(node_names, list):
    raise TypeError('node_names should be list(str)')

  node_list = []
  for node in graph_def.node:
    node_list.append(node.name)
  for node_name in node_names:
    if not node_name in node_list:
      raise NameError("Node '{}' not found in graph.".format(node_name))


def gen_quantized_node_names(graph, node_names):
  """Generate quantized node name from normal node names"""
  node_list = []
  for node in graph.get_operations():
    node_list.append(node.name)
  quantized_node_names = []
  for node_name in node_names:
    if node_name + "/wquant" in node_list:
      quantized_node_names.append(node_name + "/wquant")
    elif node_name + "/aquant" in node_list:
      quantized_node_names.append(node_name + "/aquant")
    else:
      quantized_node_names.append(node_name)
  return quantized_node_names


def get_node_dtypes(input_graph_def, target_nodes):
  """Get target node dtypes form input_graph_def"""
  node_dtypes = []
  for target_node in target_nodes:
    for node in input_graph_def.node:
      if node.name == target_node:
        if 'dtype' in node.attr:
          node_dtypes.append(tf.as_dtype(node.attr['dtype'].type).name)
        elif 'T' in node.attr:
          node_dtypes.append(tf.as_dtype(node.attr['T'].type).name)
        elif 'type' in node.attr:
          node_dtypes.append(tf.as_dtype(node.attr['type'].type).name)
        else:
          raise ValueError("Fail to get data_type of node: {}".format(node))
  return node_dtypes


def get_node_shapes(input_graph_def, target_nodes):
  """Get shapes of target nodes from input_graph_def, shapes may be partial"""
  node_shapes = []
  for target in target_nodes:
    for node in input_graph_def.node:
      if node.name == target:
        if not 'shape' in node.attr:
          print("Warning: Fail to get output shape of node: {}".format(node))
        node_shapes.append(
            tensor_shape.as_shape(node.attr['shape'].shape).as_list())
  return node_shapes


def get_quantized_nodes(input_graph_def, target_nodes):
  """Get the quantized fix_neuron nodes for target nodes. Some nodes will be followed
  by fix_neuron node(named xxx/aquant) during quantization, if exists return the fix_neuron_node,
  otherwise return itself"""
  node_list = [
      node.name.replace('/aquant', '')
      for node in input_graph_def.node
      if node.op == "FixNeuron"
  ]
  quantized_nodes = []
  for node in target_nodes:
    quantized_nodes.append(node + "/aquant" if node in node_list else node)
  return quantized_nodes


def save_pb_file(graph_def, filename):
  """Save graph_def to pb_file"""
  with tf.io.gfile.GFile(filename, mode='wb') as f:
    f.write(graph_def.SerializeToString())


def show_pb_in_tensorboard(graph_def, port=6006):
  """Show pb_file in tensorboard"""
  _ = tf.graph_util.import_graph_def(graph_def, name="")
  summary_write = tf.summary.FileWriter("./logdir/", graph)
  os.system('tensorboard --logdir ./logdir/ --port 6006')


def get_edge_tensors(graph, input_nodes, output_nodes):
  """Get input and output tensors of a graph"""
  input_tensors = [
      graph.get_tensor_by_name(name + ":0") for name in input_nodes
  ]
  output_tensors = [
      graph.get_tensor_by_name(name + ":0")
      for name in gen_quantized_node_names(graph, output_nodes)
  ]
  return input_tensors, output_tensors


def gen_feed_dict(input_tensors, inputs):
  """Generate feed dict"""
  feed_dict = dict()
  if not isinstance(inputs, dict):
    raise ValueError(
        "Expect dict(input_node_name, numpy.Array) for input_fn, but got: ",
        type(inputs))
  if len(inputs) != len(input_tensors):
    raise ValueError(
        "len(inputs) != len(input_nodes), please check your input_fn.")
  for input_tensor in input_tensors:
    name = input_tensor.op.name
    if name not in inputs:
      raise ValueError(
          "key {} not found, please check your input_fn.".format(name))
    feed_dict[input_tensor] = inputs[name]
  return feed_dict


def set_nodedef_attr(node, key, val):
  if isinstance(val, tf.DType):
    node.attr[key].type = val.as_datatype_enum
  elif key[:5] == "shape":
    shape_attr = node.attr[key].shape
    if (hasattr(shape_attr, "dim")):
      node.attr[key].shape.Clear()
    for v in val:
      node.attr[key].shape.dim.add(size=v)
  elif key == "_output_shapes":
    shape_list = node.attr[key].list.shape
    shape_list.extend([tensor_shape.TensorShape(v).as_proto() for v in val])
  elif isinstance(val, bool):
    node.attr[key].b = val
  elif isinstance(val, int):
    node.attr[key].i = val
  elif isinstance(val, float):
    node.attr[key].f = val
  elif isinstance(val, list):
    if any(isinstance(n, float) for n in val):
      node.attr[key].list.f.extend(val)
    elif all(isinstance(n, int) for n in val):
      node.attr[key].list.i.extend(val)
    else:
      raise TypeError("Only lists of floats or ints are currently supported.")
  else:
    print(key, val)
    raise TypeError("Unrecognized Type: {}, key : {}".format(key, str(type(val))))
  return node


def set_shape_info(graph_def, shape_info, plugin_output_nodes):
  for node in graph_def.node:
    if node.name in plugin_output_nodes:
      ## set the processed node's shape as the output node shape of the namescope
      plugin_outputs = plugin_output_nodes[node.name]
      if len(plugin_outputs) > 1:
        print("WARNING: plugin node ({}) has more than one output nodes \
                [{}]".format(node.name, ",".join(plugin_outputs)))
      ## plugin_outputs is a list, choose the first output node
      ## in fact if it is collapsed, it should have only one output node
      output_shapes = []
      for idx, o_node in enumerate(plugin_outputs):
        # attr_name = "shape" if idx == 0 else "shape_{}".format(idx)
        output_shapes.append(shape_info[plugin_outputs[idx]])
      set_nodedef_attr(node, "_output_shapes", output_shapes)
    elif node.name in shape_info:
      # set_nodedef_attr(node, "shape", shape_info[node.name])
      if shape_info[node.name] is not None:
        set_nodedef_attr(node, "_output_shapes", [shape_info[node.name]])
    else:
      pass
      # print("WARNING: can not find shape info of node: {}".format(node.name))
  return graph_def

def get_real_node_name(node_name):
  node_name = node_name.replace("^", "")
  node_name = node_name.split(":")[0]
  return node_name


def get_ns_nodes(graph_def, namescope_map, exclude_nodes=[]):
  def _starts_with(node_name, namescope):
    namescope = namescope.split("/")
    node_name = node_name.split("/")
    if len(namescope) > len(node_name):
      return False
    for item_np, item_nn in zip(namescope, node_name):
      if item_np != item_nn:
        return False
    return True

  ns_nodes = {}
  # collect all nodes in namescope
  for ns in namescope_map:
    ns_nodes[ns] = []
  for node in graph_def.node:
    if node.name in exclude_nodes:
      continue
    for ns in namescope_map:
      if _starts_with(node.name, ns):
        ns_nodes[ns].append(node.name)
  return ns_nodes

def get_graph_outputs(graph_def):
  input_nodes = set()
  for node in graph_def.node:
    for in_name in node.input:
      input_nodes.add(get_real_node_name(in_name))
  output_nodes = []
  for node in graph_def.node:
    if node.name not in input_nodes:
      output_nodes.append(node.name)
  return output_nodes


def get_plugin_nodes(graph_def, namescope_map):
  ns_nodes = get_ns_nodes(graph_def, namescope_map)
  plugin_nodes = {}
  # collect all nodes that are collapsed into plugin node
  for ns, plugin in ns_nodes.items():
    name_ls = plugin_nodes.setdefault(namescope_map[ns].name, [])
    name_ls.extend(ns_nodes[ns])
  return plugin_nodes


def check_if_has_duplicate_name(namescope_map):
  name_cnt = {}
  for key in namescope_map:
    node_def = deepcopy(namescope_map[key])
    namescope_map[key] = node_def
    name = node_def.name
    name_cnt[name] = 0

  final_name_list = []
  for key in namescope_map:
    name = namescope_map[key].name
    if name_cnt[name] == 0:
      final_name_list.append(name)
      name_cnt[name] += 1
    else:
      cnt = name_cnt[name]
      final_name = name + "_{}".format(cnt)
      final_name_list.append(final_name)
      name_cnt[name] += 1

  for idx, key in enumerate(namescope_map):
    namescope_map[key].name = final_name_list[idx]
  return namescope_map


def check_if_ns_in_graphdef(namescope_map, graph_def):
  ns_nodes = get_ns_nodes(graph_def, namescope_map)
  for ns in ns_nodes:
    if len(ns_nodes[ns]) == 0:
      namescope_map.pop(ns)
  return namescope_map


def check_namescope_map(namescope_map, graph_def):
  namescope_map = check_if_ns_in_graphdef(namescope_map, graph_def)
  ## conflict between this and collapse multi namescope into one plugin node
  # namescope_map = check_if_has_duplicate_name(namescope_map)
  return namescope_map

def get_plugin_output(graph_def, namescope_map):
  # ns_nodes = get_ns_nodes(graph_def, namescope_map)
  plugin_nodes = get_plugin_nodes(graph_def, namescope_map)
  graph_output_nodes = get_graph_outputs(graph_def)

  # find node that is the input of nodes(that are not in namescope)
  plugin_output_nodes = {}
  # pn = plugin_name
  for pn in plugin_nodes:
    plugin_output_nodes[pn] = []
    for node in graph_def.node:
      if node.name in plugin_nodes[pn]:
        if node.name in graph_output_nodes:
          plugin_output_nodes[pn].append(node.name)
      else:
        for in_node_name in node.input:
          in_node_name = get_real_node_name(in_node_name)
          if in_node_name in plugin_nodes[pn]:
            plugin_output_nodes[pn].append(in_node_name)
    if len(plugin_output_nodes[pn]) > 1:
      print("INFO: namescope ({}) has more than one output nodes \
              [{}]".format(pn, ",".join(plugin_output_nodes[pn])))
  return plugin_nodes, plugin_output_nodes

def load_json(json_file):
  """Load json file."""
  with open(json_file, 'r') as f:
    try:
      data = json.loads(f.read())
    except Exception as e:
      raise(
          'Fail to load the json file `{}`, please check the format. \nError: {}'
          .format(json_file, e))
  return data

def get_node_from_config(node_config):
  """
  node_config is dict has keys {"name", "op", "T", "attrs"}
  """
  node = tf.NodeDef()
  node.name = node_config["name"]
  node.op = node_config["op"]
  set_nodedef_attr(node, "T", _string2dtype[node_config["T"]])
  for key, val in node_config["attrs"].items():
    set_nodedef_attr(node, key, val)
  return node

def get_fuse_config(json_file):
  fuse_config = load_json(json_file)
  target_node_configs = fuse_config["target_nodes"]
  # map namescope_map: target_node_name
  namescope_map = fuse_config["namescope_map"]
  name_to_target_node = {}
  for config in target_node_configs:
    name_to_target_node[config["name"]] = get_node_from_config(config)
  # set namescope_map value as target node(NodeDef)
  for ns in namescope_map:
    namescope_map[ns] = name_to_target_node[namescope_map[ns]]
  return namescope_map

def map_node_plugin(ns_nodes, node_name):
  namescope = None
  for ns, node_list in ns_nodes.items():
    if node_name in node_list:
      namescope = ns
  return namescope

def fuse_ops(graph_def, namescope_map, exclude_nodes=[], save_const=False):
  def _update_input_for_plugin(plugin_node, in_name):
    in_node_ns = map_node_plugin(ns_nodes, get_real_node_name(in_name))
    if in_node_ns:
      # input is plugin node
      if namescope_map[in_node_ns].name == plugin_node.name:
        return
      in_plugin_node = namescope_map[in_node_ns]
      if in_plugin_node.name not in plugin_node.input:
        plugin_node.input.append(in_plugin_node.name)
    else:
      if in_name not in plugin_node.input:
        plugin_node.input.append(in_name)

  def _update_input_for_general(node, idx, in_name):
    in_node_ns = map_node_plugin(ns_nodes, get_real_node_name(in_name))
    if in_node_ns:
      # input is plugin node
      # node.input.remove(in_name)
      node.input.pop(idx)
      in_plugin_node = namescope_map[in_node_ns]
      if in_plugin_node.name not in node.input:
        node.input.insert(idx, in_plugin_node.name)


  ns_nodes = get_ns_nodes(graph_def, namescope_map, exclude_nodes=exclude_nodes)
  for ns in ns_nodes:
    nodes_num = len(ns_nodes[ns])
    if nodes_num == 0:
      print("INFO: namescope {} has no corresponding part in graph_def," \
        " this does not affect the program, but better check it.".format(ns))

  for node in graph_def.node:
    namescope = map_node_plugin(ns_nodes, node.name)
    if namescope:
      plugin_node = namescope_map[namescope]
      for in_name in node.input:
        _update_input_for_plugin(plugin_node, in_name)
      if save_const and node.op == "Const":
        plugin_node.attr[node.name].CopyFrom(node.attr["value"])
    else:
      for idx, in_name in enumerate(node.input):
        _update_input_for_general(node, idx, in_name)

  ns_total = 0
  idx = 0
  while idx < len(graph_def.node):
    node = graph_def.node[idx]
    if map_node_plugin(ns_nodes, node.name):
      graph_def.node.remove(node)
      ns_total += 1
    else:
      idx += 1
  for ns, node_list in ns_nodes.items():
    if len(node_list) < 1:
      continue
    if namescope_map[ns] not in graph_def.node:
      graph_def.node.append(namescope_map[ns])
  return graph_def

def add_shapes_to_graph_def(input_graph_def):
  graph = tf.Graph()
  with graph.as_default():
    tf.graph_util.import_graph_def(input_graph_def, name='')
    input_graph_def = graph.as_graph_def(add_shapes=True)
  return input_graph_def
