import json
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.contrib.decent_q import utils

string2dtype = {"tf.float32" : dtypes.float32,
        "float32": dtypes.float32,
        "float": dtypes.float32,
        "int32": dtypes.int32,
        "int64": dtypes.int64,
        "bool": dtypes.bool}


###############################
# tmp utils
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
def _parse_input_graph(input_graph):
  """Parse input_graph configurations"""
  if input_graph == '':
    raise ValueError("No --input_graph assigned.")
  if not gfile.Exists(input_graph):
    raise ValueError("Input graph file '" + input_graph + "' does not exist!")
  graph_def = graph_pb2.GraphDef()
  with gfile.GFile(input_graph, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def

###############################

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
  node = node_def_pb2.NodeDef()
  node.name = node_config["name"]
  node.op = node_config["op"]
  #TODO:delete utils
  utils.set_nodedef_attr(node, "T", string2dtype[node_config["T"]])
  for key, val in node_config["attrs"].items():
    #TODO:delete utils
    utils.set_nodedef_attr(node, key, val)
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
    in_node_ns = map_node_plugin(ns_nodes, utils.get_real_node_name(in_name))
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
    in_node_ns = map_node_plugin(ns_nodes, utils.get_real_node_name(in_name))
    if in_node_ns:
      # input is plugin node
      # node.input.remove(in_name)
      node.input.pop(idx)
      in_plugin_node = namescope_map[in_node_ns]
      if in_plugin_node.name not in node.input:
        node.input.insert(idx, in_plugin_node.name)


  ### TODO: delete utis
  ns_nodes = utils.get_ns_nodes(graph_def, namescope_map,
          exclude_nodes=exclude_nodes)
  import pdb; pdb.set_trace()
  ns_total = 0
  for ns in ns_nodes:
    ns_total += len(ns_nodes[ns])
    # TODO: add warning
    print(ns, len(ns_nodes[ns]))

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

data = load_json("./config.json")
namescope_map = get_fuse_config("./config.json")
# graph_def = _parse_input_graph("./frozen_inference_graph.pb")
graph_def = _parse_input_graph("./quantize_eval_model.pb")
exclude_nodes = [n.name for n in graph_def.node if n.op == "FixNeuron" or n.name.endswith("hard_swish/mul_1")]
graph_def = fuse_ops(graph_def, namescope_map, exclude_nodes=exclude_nodes)
filename = "./fused_quantize_eval_model.pb"
with gfile.GFile(filename, mode='wb') as f:
    f.write(graph_def.SerializeToString())
print("Fuse ops successfully, saving processed graph to ", filename)
