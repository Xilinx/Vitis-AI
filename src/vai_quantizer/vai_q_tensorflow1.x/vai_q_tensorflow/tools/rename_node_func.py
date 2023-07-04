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


import tensorflow as tf
import copy

def rename_node(graph_def, src_name, target_name):
  modified_graph_def = tf.GraphDef()
  for node in graph_def.node:
    new_node = copy.deepcopy(node)
    if node.name == src_name:
      print("rename node {} to {} ".format(src_name, target_name))
      new_node.name = target_name
      modified_graph_def.node.extend([new_node])
      continue
    for i, in_name in enumerate(node.input):
      if node.input[i] == src_name:
        print("rename node's input", node.name)
        new_node.input[i] = target_name
      else:
        new_node.input[i] = in_name
    modified_graph_def.node.extend([new_node])
  return modified_graph_def

