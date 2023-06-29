import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import vai_q_tensorflow as decent_q
# from tensorflow.contrib import decent_q


os.environ["DECENT_DEBUG"] = "0"

src_graph = "./quantize_results/quantize_eval_model.pb"
dst_graph_dir = "./"
dst_graph = "optimized_quantize_model.pb"
input_nodes = ["input_tensor"]
output_nodes = ["softmax_tensor"]
# output_nodes = ["image"]
input_shapes=[[-1, 224, 224, 3]]


def main():
  name_to_node = {}
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())
    config = decent_q.QuantizeConfig(input_nodes=input_nodes,
                                       output_nodes=output_nodes,
                                       input_shapes=input_shapes,
                                       weight_bit=8,
                                       activation_bit=8,
                                       method=1,
                                       simulate_dpu=0,
                                       output_dir=dst_graph_dir)
    folded_graph_def = decent_q.CreateOptimizedGraphDef(src_graph_def,
            config)
    tf.io.write_graph(folded_graph_def, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
