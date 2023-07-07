import os
import copy
# import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# from tensorflow.core.framework.tensor_pb2 import TensorProto
# from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto


"""
remove nodes before and include `target_node` then set input of graph as a
placeholder node
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

src_graph = "./quantize_eval_model.pb"
dst_graph_dir = "./"
dst_graph = "removed_quantize_eval_model.pb"

def get_name_to_node(graph_def):
  name_to_node = {}
  for node in graph_def.node:
    name_to_node[node.name] = node
  return name_to_node


def get_redundant_nodes(graph_def):
  remove_nodes = []
  name_to_node = get_name_to_node(graph_def)
  for node in graph_def.node:
    if node.op == "FixNeuron" and name_to_node[node.input[0]].op == "FixNeuron":
      remove_nodes.append(node.name)
  return remove_nodes

def main():
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())

  remove_nodes = get_redundant_nodes(src_graph_def)
  # import pdb; pdb.set_trace()
  removed_graph_def = src_graph_def
  for target_name in remove_nodes:
    name_to_node = get_name_to_node(removed_graph_def)
    temp_graph_def = tf.GraphDef()
    ## graph not include target node
    if not target_name in name_to_node:
      continue
    if len(name_to_node[target_name].input) != 1:
      print("WARNING: the number of target node inputs is not 1")
    for node in removed_graph_def.node:
      if node.name == target_name:
        print("remove node:", node.name)
        continue
      if node.input and target_name in node.input:
        print("re-map input node to ", node.name)
        new_node = copy.deepcopy(node)
        # import pdb; pdb.set_trace()
        for i, input_node in enumerate(node.input):
          if input_node == target_name:
            new_node.input[i] = name_to_node[target_name].input[0]
        temp_graph_def.node.extend([new_node])
      else:
        temp_graph_def.node.extend([copy.deepcopy(node)])
    removed_graph_def = temp_graph_def
  tf.io.write_graph(removed_graph_def, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
