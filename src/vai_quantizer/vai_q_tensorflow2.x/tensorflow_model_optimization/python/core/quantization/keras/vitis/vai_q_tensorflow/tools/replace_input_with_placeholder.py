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

src_graph = "/group/dphi_software/software/workspace/linqiang/vai_1.4/gitentp_aisw_1.4/vitis-ai-staging/model-test/mlperf_resnet50_v1.5_update/mlperf_resnet50_v1.5/float/resnet50_v1.pb"
dst_graph_dir = "/group/dphi_software/software/workspace/linqiang/vai_1.4/gitentp_aisw_1.4/vitis-ai-staging/model-test/mlperf_resnet50_v1.5_update/mlperf_resnet50_v1.5/float"
dst_graph = "modified_input_resnet50_v1.pb"
target_node = "input_tensor"
placeholder_name = "input_tensor"
shape=[None, None, None, 3]

def get_remove_nodes(name_to_node, target_node):
  remove_list = []
  remove_list.append(target_node)
  input_nodes = name_to_node[target_node].input
  if not input_nodes:
    return remove_list
  # when input is one of the input node's multi-output eg:['IteratorGetNext:1']
  input_node = input_nodes[0].split(":")[0]
  remove_list.extend(get_remove_nodes(name_to_node, input_node))
  return remove_list

def main():
  name_to_node = {}
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())
    for node in src_graph_def.node:
      # import pdb; pdb.set_trace()
      name_to_node[node.name] = node
    ### get nodes to be removed
    # import pdb; pdb.set_trace()
    remove_nodes = get_remove_nodes(name_to_node, target_node)
    placeholder = tf.placeholder(tf.float32, shape=shape, name=placeholder_name)
    ### remove nodes and add placehold during cloning graph_def
    replaced_graph = tf.GraphDef()
    replaced_graph.node.extend([placeholder.op.node_def])
    for node in src_graph_def.node:
      if node.name in remove_nodes:
        print("remove node:", node.name)
        continue
      if node.input and node.input[0] == target_node:
        print("re-map input node to ", node.name)
        new_node = copy.deepcopy(node)
        new_node.input[0] = placeholder_name
        replaced_graph.node.extend([new_node])
      else:
        replaced_graph.node.extend([copy.deepcopy(node)])
    tf.io.write_graph(replaced_graph, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
