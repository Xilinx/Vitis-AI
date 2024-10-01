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

def get_normal_align(fn_node, name_to_node):
  """
  fn_node is a FixNeuron, check if this fn_node is a avg_pool pattern
  output of normal node[concat, maxpool], with no scale factor
  set input pos as the same with out_pos
  """
  out_pos = fn_node.attr["quantize_pos"].i
  name_to_align_pos = {}
  normal_node = name_to_node[fn_node.input[0]]
  if not normal_node.op in ["ConcatV2", "Concat", "Maxpool"]:
    return name_to_align_pos
  for in_name in normal_node.input:
    if name_to_node[in_name].op == "FixNeuron":
      name_to_align_pos[in_name] = out_pos
  return name_to_align_pos

def get_avgpool_align(fn_node, name_to_node):
  """
  fn_node is a FixNeuron, check if this fn_node is a avg_pool pattern
  output of avg_pool node, with scale factor
  set output pos as the same with in put FixNeuron pos

  return: name_to_align_pos
  """
  name_to_align_pos = {}
  mul_node = name_to_node[fn_node.input[0]]
  if not mul_node.op in ["Mul"]:
    return name_to_align_pos
  avgpool_node = name_to_node[mul_node.input[0]]
  if not avgpool_node.op in ["AvgPool"]:
    return name_to_align_pos
  for in_name in avgpool_node.input:
    in_fn_node = name_to_node[in_name]
    if in_fn_node.op == "FixNeuron":
      in_pos = in_fn_node.attr["quantize_pos"].i
      name_to_align_pos[fn_node.name] = in_pos
  return name_to_align_pos

def implement_algin(graph_def, name_to_align_pos):
  ### implement align
  do_align = False
  for node in graph_def.node:
    # if node.name in name_to_align_pos:
    #   print(node.attr["quantize_pos"].i, name_to_align_pos[node.name])
    if node.name in name_to_align_pos and \
        node.attr["quantize_pos"].i != name_to_align_pos[node.name]:
      print("modify {} pos from {} to {}, in order to keep input pos and output pos the same".format(
          node.name, node.attr["quantize_pos"].i, name_to_align_pos[node.name]))
      node.attr["quantize_pos"].i = name_to_align_pos[node.name]
      do_align = True
  return graph_def, do_align

def align_pos(graph_def, align_concat=True, align_maxpool=True, align_avgpool=False):
  name_to_node = {}
  for node in graph_def.node:
    name_to_node[node.name] = node

  name_to_align_pos = {}
  do_align = False
  iter_num = 0
  while True:
    iter_num += 1
    if iter_num > 100:
      print("Program may enter the infinite loop, please contact with the author")
      break
    for node in graph_def.node:
      if node.op == "FixNeuron":
        ### record align node and its pos
        name_to_align_pos.update(get_normal_align(node, name_to_node))
        name_to_align_pos.update(get_avgpool_align(node, name_to_node))
    graph_def, do_align = implement_algin(graph_def, name_to_align_pos)
    if not do_align:
      break
  return graph_def
