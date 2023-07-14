# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

src_graph = "./optimized_quantize_model.pb"
dst_graph_dir = "./"
dst_graph = "removed_optimized_quantize_model.pb"

def get_name_to_node(graph_def):
  name_to_node = {}
  for node in graph_def.node:
    name_to_node[node.name] = node
  return name_to_node


def main():
  with gfile.FastGFile(src_graph,'rb') as f:
    src_graph_def = tf.GraphDef()
    src_graph_def.ParseFromString(f.read())

  # import pdb; pdb.set_trace()
  temp_graph_def = tf.GraphDef()
  visited_node = []
  for node in src_graph_def.node:
    if node.name not in visited_node:
      visited_node.append(node.name)
      new_node = copy.deepcopy(node)
      temp_graph_def.node.extend([new_node])
    else:
      continue
  removed_graph_def = temp_graph_def
  tf.io.write_graph(removed_graph_def, dst_graph_dir, dst_graph, as_text=False)


if __name__ == '__main__':
  main()
