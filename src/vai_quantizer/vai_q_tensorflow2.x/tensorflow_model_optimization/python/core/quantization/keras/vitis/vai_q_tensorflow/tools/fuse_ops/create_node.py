from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.contrib.decent_q import utils

import pdb; pdb.set_trace()
string2dtype = {"tf.float32" : dtypes.float32,
        "float32": dtypes.float32,
        "float": dtypes.float32,
        "int32": dtypes.int32,
        "int64": dtypes.int64,
        "bool": dtypes.bool}

def get_node_from_config(node_config):
  """
  node_config is dict has keys {"name", "op", "T", "attrs"}
  """
  node = node_def_pb2.NodeDef()
  node.name = node_config["name"]
  node.op = node_config["op"]
  #TODO:delete utils
  utils.set_nodedef_attr(node, "T", config["T"])
  for key, val in node_config["attrs"]:
    #TODO:delete utils
    utils.set_nodedef_attr(node, key, val)
  return node
