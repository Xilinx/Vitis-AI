import os
import copy
# import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.contrib.decent_q.python import quantize_graph


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

src_graph = "./frozen_quantize_eval_model_20200520161753.pb"
dst_graph_dir = "./"
dst_graph = "foldedBN_quantize_eval_model.pb"


def main():
  name_to_node = {}
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())
    config = tf.contrib.decent_q.QuantizeConfig(input_nodes=["net_in"],
                                       output_nodes=["Squeeze", "concat_1"],
                                       input_shapes=[[-1, 300, 300, 3]],
                                       weight_bit=8,
                                       activation_bit=8,
                                       method=1,
                                       # align_concat=2,
                                       output_dir="./quantize_results")
    folded_graph_def = quantize_graph.ConvertFoldedBatchnorms(src_graph_def,
            config)
    tf.io.write_graph(folded_graph_def, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
