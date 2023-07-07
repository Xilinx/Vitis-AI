#
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
#


class TransMatmulActvHandler(object):
  def __call__(self, *args, **kwargs):
    _, node_set = args
    trans_node = node_set[0]
    matmul_node = node_set[1]

    def is_node_just_exchange_last_two_dim(node):
      node_order = node.node_attr(node.op.AttrName.ORDER)
      if len(node_order) < 2:
        return False
      flag = True
      for i in range(len(node_order)):
        if i < len(node_order) - 2 and node_order[i] != i:
          flag = False
          break
        elif i == len(node_order) - 2 and ((i != node_order[-1]) or (i+1 != node_order[-2])):
          flag = False
          break
      return flag

    def only_one_child_node(node):
      if node.out_nodes:
        out_nodes_list = node.out_nodes
        out_nodes_list_without_dequant = []
        for out_node_name in out_nodes_list:
          if node.owning_graph.node(out_node_name).op.type != 'dequant_stub':
            out_nodes_list_without_dequant.append(out_node_name)
        if len(out_nodes_list_without_dequant) == 1:
          return True
        else:
          return False
      else:
        return False

    matmul_node.is_trans_b = is_node_just_exchange_last_two_dim(trans_node) and only_one_child_node(trans_node)
    if matmul_node.is_trans_b:
      matmul_node.set_node_attr(matmul_node.op.AttrName.TRANSPOSE_B, True)
