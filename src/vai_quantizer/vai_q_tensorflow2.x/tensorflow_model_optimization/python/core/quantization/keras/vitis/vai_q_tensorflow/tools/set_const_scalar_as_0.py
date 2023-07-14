import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# from tensorflow.core.framework.tensor_pb2 import TensorProto
# from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
# import tensorflow.contrib.decent_q
from tensorflow.python.framework import tensor_util
import vai_q_tensorflow


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

  for node in src_graph_def.node:
    if node.op == "Const":
      val = node.attr["value"].tensor
      val = tensor_util.MakeNdarray(val)
      shape_len = len(val.shape)
      if val.all() == 0:
        print(node)
        print()
        float_val = node.attr["value"].tensor.float_val
        node.attr["value"].tensor.float_val[0] = 0.0001
        # print(node)


  save_pb(dst_graph, src_graph_def)


if __name__ == '__main__':
  main()
