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
remove_nodes = [
        "FeatureExtractor/MobilenetV2/MobilenetV2/input/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv/input/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_1/input/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_1/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_1/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_2/input/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_2/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_2/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_3/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_3/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_4/input/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_4/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_4/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_5/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_5/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_6/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_6/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_7/input/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_7/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_7/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_8/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_8/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_9/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_9/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_10/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_10/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_11/input/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_11/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_11/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_12/expansion_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_12/depthwise_output/aquant",
        "FeatureExtractor/MobilenetV2/expanded_conv_13/expansion_output/aquant"
        ]

def main():
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())

    removed_graph_def = src_graph_def
    for target_name in remove_nodes:
      name_to_node = {}
      temp_graph_def = tf.GraphDef()
      for node in removed_graph_def.node:
        name_to_node[node.name] = node
      ## graph not include target node
      if not target_name in name_to_node:
        continue
      if len(name_to_node[target_name].input) > 1:
        print("WARNING: the number of target node({}) inputs is > 1, this may"
              " lead to generate some node that independent of main graph.".format(target_name))
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
              # import pdb; pdb.set_trace()
              target_node = name_to_node[target_name]
              if target_node.input:
                new_node.input[i] = target_node.input[0]
              else:
                new_node.input.remove(target_name)
          temp_graph_def.node.extend([new_node])
        else:
          temp_graph_def.node.extend([copy.deepcopy(node)])
      removed_graph_def = temp_graph_def
    tf.io.write_graph(removed_graph_def, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
