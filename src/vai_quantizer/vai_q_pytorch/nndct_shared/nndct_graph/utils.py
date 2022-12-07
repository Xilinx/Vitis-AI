
import sys
from typing import List, Optional

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph, Operation, Node, Block
from nndct_shared.nndct_graph import operator_definition as base_op
from nndct_shared.utils import NndctOption, NndctScreenLogger


def glue_group_members(graph, groups, start_node, c_node):
  assert groups[c_node][0] == c_node
  name_lst = groups[start_node]
  for g in groups[c_node]:
    if g not in name_lst:
      name_lst.append(g)
  for n in name_lst:
    groups[n] = name_lst
  return groups


def reset_group_members(graph, groups, host, servent):
  if servent not in groups[host]:
    members = sorted(groups[host] + [servent], key=lambda n: graph.node(n).idx)
    groups[host] = members
    groups[servent] = members
  return groups


def group_up(graph, groups, OpType=None, POpType=None):

  def __is_valid_parent(node):
    if len(graph.children(node)) > 1 or node.op.is_custom_op:
      return False
    
    if POpType is None:
      return True
    elif node.op.type == POpType:
      return True
    return False

  for n in graph.all_nodes():
    if not n.in_quant_part or n.blocks:
      continue
    if NndctOption.nndct_stat.value > 2:
      print('node name: {} parent number: {}'.format(n.name, len(graph.parents(n.name))))
    if groups[n.name][0] == n.name and n.op.type == OpType and \
        len(graph.parents(n.name)) == 1 and __is_valid_parent(graph.parents(n.name)[0]):
      start_node = groups[graph.parents(n.name)[0].name][0]
      groups = glue_group_members(graph, groups, start_node, n.name)
      if NndctOption.nndct_stat.value > 2:
        print('---- Grouping node %s and %s' % (start_node, n.name))
  return groups

# TODO: Remove this function
def reorder_multi_subgraph_nodes(graphs: List[Graph]) -> None:
  node_index = 0
  for graph in graphs:
    graph.clear_node_id_map()
    for node in graph.nodes:
      node.idx = node_index
      node_index += 1

# deprecated
def merge_multi_subgraphs(graphs: List[Graph],
                          graph_name="Nndctgraph") -> Graph:
  top_graph = Graph(graph_name)
  for graph in graphs:
    for node in graph.nodes:
      top_graph.add_node(node)
  return top_graph

def merge_multi_graphs_to_single_graph(graphs, graph_name="Nndctgraph"):
  top_graph = Graph(graph_name)
  op = base_op.CustomOp(NNDCT_OP.PLACEHOLDER)
  input_node = Node(name="input_placeholder", op=op, in_quant_part=False)
  input_node.owning_graph = top_graph
  op = base_op.CustomOp(NNDCT_OP.PLACEHOLDER)
  return_node = Node(name="return_placeholder", op=op, in_quant_part=False)
  return_node.owning_graph = top_graph
  
  top_block = Block(top_graph, None, input_node, return_node)
  top_graph.set_top_block(top_block)
  for graph in graphs:
    block_node = convert_graph_to_block_node(top_graph, top_block, graph)
    if not block_node.in_node_list():
      top_graph.append_node(block_node)
  return top_graph

def convert_graph_to_block_node(top_graph, top_block, graph):
  op = base_op.CustomOp(NNDCT_OP.BLOCK)
  block_node = Node(name=graph.name, op=op, in_quant_part=True)
  block_node.owning_block = top_block
  block_node.owning_graph = top_graph
  block_node.add_block(graph.block)
  for block in graph.all_blocks():
    block.owning_graph = top_graph
    
  for tensor in graph.tensors:
    top_graph.add_tensor(tensor)
  
  for node in graph.all_nodes():
    node._idx = -1
    node.owning_graph = top_graph

  for param_name in graph.param_names():
    top_graph.add_param_name(param_name)

  return block_node

  
     

  
