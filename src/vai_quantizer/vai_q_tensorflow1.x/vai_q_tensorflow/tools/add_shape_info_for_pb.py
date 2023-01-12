import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import vai_q_tensorflow as decent_q
# from tensorflow.contrib import decent_q


os.environ["DECENT_DEBUG"] = "0"

src_graph = "./removed_optimized_quantize_model.pb"
dst_graph_dir = "./"
dst_graph = "removed_optimized_quantize_model.pb"


def main():
  name_to_node = {}
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
      tf.graph_util.import_graph_def(src_graph_def, name='')
      dst_graph_def = graph.as_graph_def(add_shapes=True)
    tf.io.write_graph(dst_graph_def, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
