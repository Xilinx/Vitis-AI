

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

import copy
from typing import Dict
from .torch_graph import TorchNode, TorchValue
import itertools


class OptPass(object):
  
  
  @staticmethod
  def merge_node_sets_to_list(node_sets):
    if not node_sets:
      return []
    nodes_list = []
    for _, nodes in node_sets.items():
      nodes_list.extend(nodes)
    return nodes_list
  
  @staticmethod
  def penetrate_tuple_pick(raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
      
    for nodeset in nodes_list:
      tuple_node = nodeset[0]
      index_node = nodeset[1]
      if index_node.inputs[1].is_plain_value():
        index = index_node.inputs[1].data
      else:
        continue
      
      remove_nodes.append(index_node)
      node2idx = {}
      for c_node in raw_graph.children(index_node):
        for i, inp in enumerate(c_node.inputs):
          if inp.node is index_node:
            node2idx[c_node] = i
            
      for node, i in node2idx.items():
        node.inputs[i] = tuple_node.inputs[index]  
        
      if index_node.outputs[0].name in raw_graph.ret_values():
        ori_return_values = copy.copy(raw_graph.ret_values())
        raw_graph.ret_values().clear()
        for name, value in ori_return_values.items():
          if name == index_node.outputs[0].name:
            raw_graph.add_ret_value(tuple_node.inputs[index])
          else:
            raw_graph.add_ret_value(value)
    
    for r_node in remove_nodes:
      raw_graph.remove_node(r_node)
      
  @staticmethod
  def penetrate_pack_unpack(raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return 
    for nodeset in nodes_list:
      pack_node = nodeset[0]
      unpack_node = nodeset[1]
      assert len(pack_node.inputs) == len(unpack_node.outputs)
      remove_nodes.extend([pack_node, unpack_node])
      for outp, inp in zip(unpack_node.outputs, pack_node.inputs):
        for node in unpack_node.out_nodes:
          if outp in node.inputs:
            index = node.inputs.index(outp)
            node.inputs[index] = inp
          
        if outp.name in raw_graph.ret_values():
          ori_return_values = copy.copy(raw_graph.ret_values())
          raw_graph.ret_values().clear()
          for name, value in ori_return_values.items():
            if name == outp.name:
              raw_graph.add_ret_value(inp)
            else:
              raw_graph.add_ret_value(value)
              
    for r_node in remove_nodes:
      raw_graph.remove_node(r_node)
    
    if remove_nodes:
      raw_graph.reconnect_nodes()
    
    
  @staticmethod
  def unpack_ListUnpack_op(raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return 
    for nodeset in nodes_list:
      # unpack_in_node = nodeset[0]
      unpack_node = nodeset[0]
      if len(unpack_node.outputs) < 2:
        continue
      remove_nodes.append(unpack_node)
      for unpack_in_node in raw_graph.parents(unpack_node): 
        node2index = {}
        for on in unpack_in_node.out_nodes:
          for i, ip in enumerate(on.inputs):
            if ip in unpack_in_node.outputs:
              j = unpack_in_node.outputs.index(ip)
              node2index[id(on)] = (i, j)
          
        unpack_in_node.outputs = unpack_node.outputs
        for output in unpack_in_node.outputs:
          output.node = unpack_in_node  
                        
        for on in unpack_in_node.out_nodes:
          if id(on) in node2index:
            i, j = node2index[id(on)]
            on.inputs[i] = unpack_in_node.outputs[j]  
          
    for r_node in remove_nodes:
      raw_graph.remove_node(r_node)
    
    if remove_nodes:
      raw_graph.reconnect_nodes()

  
  @staticmethod
  def pack_ListConstruct_op(raw_graph, node_sets):
    ListConstruct_nodes = set()
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for nodeset in nodes_list:
      packed_node = nodeset[0]
      for packed_out_node in raw_graph.children(packed_node):
        id2list = {}
        for i, ip in enumerate(packed_out_node.inputs):
          if isinstance(ip, list):
            continue
          if ip.node and ip.node is packed_node and len(ip.node.outputs) == 1:
            id2list[i] = packed_node.inputs
            ListConstruct_nodes.add(packed_node)
          
        for i, lst in id2list.items():
          packed_out_node.inputs[i] = lst
      
      if packed_node.outputs[0].name in raw_graph.ret_values():
        ListConstruct_nodes.add(packed_node)
        raw_graph.ret_values()[packed_node.outputs[0].name] = packed_node.inputs

    for node in ListConstruct_nodes:
      raw_graph.remove_node(node)

    if ListConstruct_nodes:
      raw_graph.reconnect_nodes()
    
    
  @staticmethod
  def slice_to_strided_slice(raw_graph, node_sets):
    strided_slice_nodes = {}
    slice_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return

    sorted_node_list = sorted(nodes_list, key=lambda nodeset: nodeset[0].idx)

    slice_end_point = []
    for nodeset in sorted_node_list:
      node = nodeset[0]
      slice_nodes.append(node)
      if node.inputs[0].node.kind != "strided_slice" or (node.inputs[0].node in slice_end_point):
        strided_node = TorchNode()
        strided_node.idx = node.idx
        strided_node.kind = "strided_slice"
        strided_node.name = node.name
        strided_node.add_input(node.inputs[0])
        strided_node.dtype = node.dtype
        for ip in node.inputs[1:]:
          strided_node.add_input([ip])
        strided_node.add_output(node.outputs[0])
        # strided_node.outputs
        strided_slice_nodes[node.name] = strided_node
      else:
        node.name = node.inputs[0].node.name
        strided_node = strided_slice_nodes[node.inputs[0].node.name]
        if len(node.out_nodes) > 1: slice_end_point.append(strided_node)
        for i, ip in enumerate(node.inputs[1:], 1):
          strided_node.inputs[i].append(ip)
        strided_node.outputs[0] = node.outputs[0]
        strided_node.outputs[0].node = strided_node
    
    for node in strided_slice_nodes.values():
      # raw_graph.nodes[node.idx] = node
      raw_graph.set_node_with_idx(node, node.idx)

    for node in slice_nodes:
      if node in raw_graph.nodes:
        raw_graph.remove_node(node)
        
    if slice_nodes:                        
      raw_graph.reconnect_nodes() 
    
  
  @staticmethod
  def select_to_slice_inplace_copy(raw_graph, node_sets):
    select_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for node_set in nodes_list:
      select_node = node_set[0]
      copy_node = node_set[1]
      select_nodes.append(select_node)
      copy_node.kind = "slice_tensor_inplace_copy"
      inputs = select_node.inputs[1:]
      copy_node.inputs[0] = select_node.inputs[0]
      for ip in inputs:
        copy_node.add_input(ip)
  
    for node in select_nodes:
      raw_graph.remove_node(node)

    if select_nodes:
      raw_graph.reconnect_nodes()
    
  
  @staticmethod
  def create_stride_slice_inplace_copy(raw_graph, node_sets):
    stride_slice_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for node_set in nodes_list:
      stride_slice_node = node_set[0]
      copy_node = node_set[1]
      if copy_node.inputs[0] is not stride_slice_node.outputs[0]:
        continue
      
      stride_slice_nodes.append(stride_slice_node)
      copy_node.kind = "strided_slice_inplace_copy"
      source_input = copy_node.inputs[1]
      copy_node.inputs.clear()
      # inputs = stride_slice_node.inputs[1:]
      # copy_node.inputs[0] = select_node.inputs[0]
      for ip in stride_slice_node.inputs + [source_input]:
        copy_node.add_input(ip)
  
    for node in stride_slice_nodes:
      raw_graph.remove_node(node)
    
    if stride_slice_nodes:
      raw_graph.reconnect_nodes()
    
  
  @staticmethod
  def stride_slice_to_index_inplace_put(raw_graph, node_sets):
    select_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for node_set in nodes_list:
      select_node = node_set[0]
      put_node = node_set[1]
    
      select_nodes.append(select_node)    
      inputs = select_node.inputs[1:]
      put_node.inputs[0] = select_node.inputs[0]
      for ip in inputs:
        put_node.add_input(ip)
            
      for cn in put_node.inputs[0].node.out_nodes:
        index = cn.inputs.index(put_node.inputs[0])
        cn.inputs[index] = put_node.outputs[0]

    for node in select_nodes:
      raw_graph.remove_node(node)

    if select_nodes:
      raw_graph.reconnect_nodes()
    
  @staticmethod
  def strip_reduantant_tensors_in_embedding_bag(raw_graph, node_sets):
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for node_set in nodes_list:
      emb = node_set[0]
      emb.outputs = emb.outputs[:1]
  
  
  @staticmethod
  def merge_matmul_with_add(raw_graph, node_sets):
    matmul_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return

    for node_set in nodes_list:
      if not all([node.inputs[1].name in raw_graph.param_names() for node in node_set]):
        continue
      
      mat_mul, add = node_set
      matmul_nodes.append(mat_mul)
      add.kind = "addmm"
      addmm_inputs = []
      addmm_inputs.extend([add.inputs[1], mat_mul.inputs[0], mat_mul.inputs[1]]) 
      add.inputs.clear()
      for ip in addmm_inputs:
        add.add_input(ip)
        
    for node in matmul_nodes:
      raw_graph.remove_node(node)
    
    if matmul_nodes:
      raw_graph.reconnect_nodes(raw_graph)
    
  @staticmethod
  def merge_param_transpose_with_addmm(raw_graph, node_sets):
    t_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return

    for node_set in nodes_list:
      if node_set[1].kind == "addmm":
        t, addmm = node_set
        if (t.inputs[0].name not in raw_graph.param_names()) or (addmm.inputs[0].name not in raw_graph.param_names()):
          continue
        
        t_nodes.append(t)
        addmm.inputs[2] = t.inputs[0]
      else:
        t, matmul = node_set
        if (t.inputs[0].name not in raw_graph.param_names()):
          continue
        
        t_nodes.append(t)
        matmul.inputs[1] = t.inputs[0]
        
    for node in t_nodes:
      raw_graph.remove_node(node)
      
    if t_nodes:
      raw_graph.reconnect_nodes()

  @staticmethod
  def merge_select_to_strided_slice(raw_graph, node_sets):
    def have_inplace_copy_child(node, visited):
      visited.append(node)
      if node.kind == "copy_":
        return True
      
      for cn in raw_graph.children(node):
        if cn not in visited and node.outputs[0] is cn.inputs[0]: 
          if have_inplace_copy_child(cn, visited):
            return True
      return False    
    
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return

    sorted_node_list = sorted(nodes_list, key=lambda nodeset: nodeset[0].idx)
    for node_set in sorted_node_list:
      select_node = node_set[0]
      strided_slice = node_set[1]
      
      if len(select_node.outputs) > 1 or select_node.outputs[0] is not strided_slice.inputs[0]:
        continue
      
      visited = []
      if not have_inplace_copy_child(strided_slice, visited):
        continue
      
      
      remove_nodes.append(select_node)
      strided_slice.inputs[0] = select_node.inputs[0]
      select_dim = select_node.inputs[1]
      select_index = select_node.inputs[2]
      
      new_slice_dim = select_dim
      new_slice_start = select_index
      new_slice_end = select_index
      new_slice_step = TorchValue(1)     
      
      strided_slice_dims = []
      for i, dim in enumerate([new_slice_dim] + strided_slice.inputs[1]):
        if i > 0:
          dim.data += 1
        strided_slice_dims.append(dim)
      
      strided_slice.inputs[1] = strided_slice_dims
      strided_slice.inputs[2] = [new_slice_start] + strided_slice.inputs[2][:]
      strided_slice.inputs[3] = [new_slice_end] + strided_slice.inputs[3][:]
      strided_slice.inputs[4] = [new_slice_step] + strided_slice.inputs[4][:]
      
    for node in remove_nodes:
      raw_graph.remove_node(node)
      
    if remove_nodes:
      raw_graph.reconnect_nodes()

    return True if remove_nodes else False
    
      
  @staticmethod
  def merge_consecutive_strided_slice(raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return   
    
    filter_nodes_list = [node_set for node_set in nodes_list if len(node_set[0].out_nodes) == 1]

    sorted_node_list = sorted(filter_nodes_list, key=lambda nodeset: nodeset[0].idx)
        
    consecutive_slice_group = []
    for nodeset in sorted_node_list:
      if consecutive_slice_group:
        if nodeset[0] in consecutive_slice_group[-1]:
          consecutive_slice_group[-1].append(nodeset[1])
        else:
          consecutive_slice_group.append(nodeset)
      else:
        consecutive_slice_group.append(nodeset)
     
    for slice_group in consecutive_slice_group:
      dim = list(itertools.chain.from_iterable([slice_op.inputs[1] for slice_op in slice_group]))
      start = list(itertools.chain.from_iterable([slice_op.inputs[2] for slice_op in slice_group]))
      end = list(itertools.chain.from_iterable([slice_op.inputs[3] for slice_op in slice_group]))
      step = list(itertools.chain.from_iterable([slice_op.inputs[4] for slice_op in slice_group]))
      slice_group[-1].inputs[0] = slice_group[0].inputs[0]
      slice_group[-1].inputs[1] = dim
      slice_group[-1].inputs[2] = start
      slice_group[-1].inputs[3] = end
      slice_group[-1].inputs[4] = step
      remove_nodes.extend(slice_group[:-1])
      
    
    for node in remove_nodes:
      raw_graph.remove_node(node)
      
    if remove_nodes:
      raw_graph.reconnect_nodes()
      

  @staticmethod
  def remove_reduantant_view(raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return   
    
    for nodeset in nodes_list:
      view = nodeset[0]
      if view.inputs[0].shape == view.outputs[0].shape:
        remove_nodes.append(view)
        for node in raw_graph.children(view):
          for i, inp in enumerate(node.inputs):
            if inp is view.outputs[0]:
              node.inputs[i] = view.inputs[0]
      
    for node in remove_nodes:
      raw_graph.remove_node(node)
      
    if remove_nodes:
      raw_graph.reconnect_nodes()
          
    
  @staticmethod
  def merge_empty_with_zero(raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return   
    
    for nodeset in nodes_list:
      empty = nodeset[0]
      zero = nodeset[1]
      zero.kind = 'zeros'
      zero.inputs.clear()
      for input in empty.inputs[:-1]:
        zero.inputs.append(input)
      
      remove_nodes.append(empty)
      
    for node in remove_nodes:
      raw_graph.remove_node(node)
      
    if remove_nodes:
      raw_graph.reconnect_nodes()

      
  @staticmethod
  def transform_const_scalar_to_const_tensor(raw_graph, node_sets):
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return   
    
    const_nodes = []
    for nodeset in nodes_list:
      binary_op = nodeset[0]
      if not binary_op.inputs[0].is_plain_value() and binary_op.inputs[1].is_plain_value():
        binary_op.inputs[1].convert_plain_value_to_tensor()
        const_node = TorchNode()
        const_node.kind = "Constant"
        const_node.name = binary_op.inputs[1].name
        const_node.add_output(binary_op.inputs[1])
        binary_op.inputs[1].node = const_node
        const_nodes.append(const_node)
       
                
    if const_nodes:
      copied_nodes = list(raw_graph.nodes)
      raw_graph.clean_nodes()
      for node in const_nodes + copied_nodes:
        raw_graph.add_node(node)
      raw_graph.reconnect_nodes()
               
  @staticmethod
  def merge_internal_type_as(raw_graph, node_sets):
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return

    type_as_nodes = []
    for node_set in nodes_list:
      type_as, op = node_set
      if len(type_as.out_nodes) > 1:
        continue
      type_as_nodes.append(type_as)
      inp_index = op.inputs.index(type_as.outputs[0])
      for inp in type_as.inputs:
        if inp not in op.inputs:
          op.inputs[inp_index] = inp
        
    for node in type_as_nodes:
      raw_graph.remove_node(node)
      
    if type_as_nodes:
      raw_graph.reconnect_nodes()
          
          
  @staticmethod
  def remove_reduantant_int(raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return   
    
    for nodeset in nodes_list:
      int_op = nodeset[0]
      if int_op.inputs[0].dtype == "torch.long":
        remove_nodes.append(int_op)
        for node in raw_graph.children(int_op):
          for i, inp in enumerate(node.inputs):
            if inp is int_op.outputs[0]:
              node.inputs[i] = int_op.inputs[0]
      
    for node in remove_nodes:
      raw_graph.remove_node(node)
      
    if remove_nodes:
      raw_graph.reconnect_nodes()
      
      
