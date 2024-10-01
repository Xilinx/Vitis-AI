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


class TransReductionOpActvHandler(object):
  def __call__(self, *args, **kwargs):
    _, node_set = args
    trans_node = node_set[0]
    reduction_op_node = node_set[1]

    def can_fuse_trans_reduction_op(reduction_op_node, trans_node):
      flag = True
      
      if reduction_op_node.op.keepdim:
          flag = False
          return flag

      transpose_order = trans_node.node_attr(trans_node.op.AttrName.ORDER)
      reduction_dim = reduction_op_node.op.dim
      order_after_transpose = []
      new_reduction_dim = []

      for i in range(len(transpose_order)):
        if i not in reduction_dim:
          order_after_transpose.append(transpose_order[i])
        else:
          new_reduction_dim.append(transpose_order[i])
      for i in range(len(order_after_transpose) - 1):
        if order_after_transpose[i + 1] <= order_after_transpose[i]:
          flag = False
          break

      return flag, new_reduction_dim

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
    
    flag, new_reduction_dim = can_fuse_trans_reduction_op(reduction_op_node, trans_node)
    reduction_op_node.fuse_trans_reduction_op = flag and only_one_child_node(reduction_op_node)
    if reduction_op_node.fuse_trans_reduction_op and (len(new_reduction_dim) > 0):
        reduction_op_node.new_dims = new_reduction_dim
