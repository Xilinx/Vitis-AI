import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# from tensorflow.core.framework.tensor_pb2 import TensorProto
# from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto


"""
remove nodes before and include `target_node` then set input of graph as a
placeholder node
"""

tf.app.flags.DEFINE_string('input_pb', '', 'input pb file')
tf.app.flags.DEFINE_string('dst_dir', 'results', 'save dir')

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

src_graph = FLAGS.input_pb
dst_graph = os.path.join(FLAGS.dst_dir, FLAGS.input_pb)

def save_pb(dst_graph, graph_def):
  dst_dir = os.path.dirname(dst_graph)
  try:
    os.makedirs(dst_dir)
  except OSError as error:
    pass
    # print(error)
  with gfile.GFile(dst_graph, mode='wb') as f:
    f.write(graph_def.SerializeToString())
  print("saing processed grapb pb to ", dst_graph)

def load_pb(src_graph):
  with gfile.FastGFile(src_graph,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def main():
  src_graph_def = load_pb(src_graph)
  name_to_node = {}
  for node in src_graph_def.node:
    name_to_node[node.name] = node

  channel_info_node_type = ["Add", "BiasAdd"]
  pattern_list = []
  ## find (Relu6, Add) pair
  for node in src_graph_def.node:
    if node.op == "Relu6" and len(node.input) == 1:
      node.op = "Relu"
      inp_node = name_to_node[node.input[0]]
      if inp_node.op == "Add":
        add_node = inp_node
        biasadd_fn = name_to_node[add_node.input[0]]
        biasadd_node = name_to_node[biasadd_fn.input[0]]
        bias_fn = name_to_node[biasadd_node.input[1]]
        bias = name_to_node[bias_fn.input[0]]

        add_y_fn = name_to_node[add_node.input[1]]
        add_y = name_to_node[add_y_fn.input[0]]

        channel = bias.attr["value"].tensor.tensor_shape.dim[0].size
        val = np.array([3.0] * channel, dtype=np.float32)
        add_y.attr["value"].tensor.tensor_shape.CopyFrom(tf.TensorShape(val.shape).as_proto())
        # add_y.attr["value"].tensor.tensor_content = bytes(val)
        add_y.attr["value"].tensor.tensor_content = val.tobytes()

  save_pb(dst_graph, src_graph_def)


if __name__ == '__main__':
  main()
