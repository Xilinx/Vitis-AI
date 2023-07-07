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
def _parse_input_graph(input_graph):
  """Parse input_graph configurations"""
  if input_frozen_graph == '':
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

data = load_json("./config.json")
namescope_map = get_fuse_config("./config.json")
import pdb; pdb.set_trace()
