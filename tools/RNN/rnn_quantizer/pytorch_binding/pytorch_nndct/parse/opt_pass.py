

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

from .torch_graph import TorchNode


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
  def unpack_ListUnpack_op(graph_handler, raw_graph, node_sets):
    remove_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return 
    for nodeset in nodes_list:
      # unpack_in_node = nodeset[0]
      unpack_node = nodeset[0]
      remove_nodes.append(unpack_node)
      for unpack_in_node in raw_graph.parents(unpack_node): 
        node2index = {}
        for on in unpack_in_node.out_nodes:
          for i, ip in enumerate(on.inputs):
            if ip in unpack_in_node.outputs:
              j = unpack_in_node.outputs.index(ip)
              node2index[id(on)] = (i, j)
                  
        unpack_in_node.outputs = unpack_node.outputs
                
        for on in unpack_in_node.out_nodes:
          if id(on) in node2index:
            i, j = node2index[id(on)]
            on.inputs[i] = unpack_in_node.outputs[j]  
          
    for r_node in remove_nodes:
      raw_graph.remove_node(r_node)
      
    graph_handler.reconnect_nodes(raw_graph)

  
  @staticmethod
  def pack_ListConstruct_op(graph_handler, raw_graph, node_sets):
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
          if ip.node and ip.node.kind == packed_node.kind:
            id2list[i] = packed_node.inputs
            ListConstruct_nodes.add(packed_node)
          
        for i, lst in id2list.items():
          packed_out_node.inputs[i] = lst

    for node in ListConstruct_nodes:
      raw_graph.remove_node(node)

    graph_handler.reconnect_nodes(raw_graph)
    
    
  @staticmethod
  def slice_to_strided_slice(graph_handler, raw_graph, node_sets):
    strided_slice_nodes = {}
    slice_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for nodeset in nodes_list:
      node = nodeset[0]
      slice_nodes.append(node)
      if node.inputs[0].node.kind != "strided_slice":
        strided_node = TorchNode()
        strided_node.idx = node.idx
        strided_node.kind = "strided_slice"
        strided_node.name = node.name
        strided_node.add_input(node.inputs[0])
        strided_node.dtype = node.dtype
        for ip in node.inputs[1:]:
          strided_node.add_input([ip])
        strided_node.add_output(node.outputs[0])
        strided_node.outputs
        strided_slice_nodes[node.name] = strided_node
      else:
        node.name = node.inputs[0].node.name
        strided_node = strided_slice_nodes[node.inputs[0].node.name]
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

    graph_handler.reconnect_nodes(raw_graph) 
    
  
  @staticmethod
  def select_to_slice_inplace_copy(graph_handler, raw_graph, node_sets):
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

    graph_handler.reconnect_nodes(raw_graph)
    
  
  @staticmethod
  def create_stride_slice_inplace_copy(graph_handler, raw_graph, node_sets):
    stride_slice_nodes = []
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for node_set in nodes_list:
      stride_slice_node = node_set[0]
      copy_node = node_set[1]
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

    graph_handler.reconnect_nodes(raw_graph)
    
  
  @staticmethod
  def stride_slice_to_index_inplace_put(graph_handler, raw_graph, node_sets):
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

    graph_handler.reconnect_nodes(raw_graph)
    
  @staticmethod
  def strip_reduantant_tensors_in_embedding_bag(graph_handler, raw_graph, node_sets):
    nodes_list = OptPass.merge_node_sets_to_list(node_sets)
    if not nodes_list:
      return
    for node_set in nodes_list:
      emb = node_set[0]
      emb.outputs = emb.outputs[:1]
  
  
  @staticmethod
  def merge_matmul_with_add(graph_handler, raw_graph, node_sets):
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
    
    graph_handler.reconnect_nodes(raw_graph)
