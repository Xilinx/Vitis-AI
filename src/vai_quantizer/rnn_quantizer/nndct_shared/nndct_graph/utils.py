
import sys
from typing import List, Optional

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph, Operation
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

  for n in graph.nodes:
    if not n.in_quant_part:
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


def reorder_multi_subgraph_nodes(graphs: List[Graph]) -> None:
  node_index = 0
  for graph in graphs:
    graph.id2node_map.clear()
    for node in graph.nodes:
      graph.set_node_id(node_index, node)
      node_index += 1


def merge_multi_subgraphs(graphs: List[Graph],
                          graph_name="Nndctgraph") -> Graph:
  top_graph = Graph(graph_name)
  for graph in graphs:
    for node in graph.nodes:
      top_graph.add_node(node)
  return top_graph


  
def collect_all_blocks(graph: Graph, blocks: Optional[List[Graph]] = None) -> List[Graph]:
  if blocks is None:
    blocks: List[Graph] = []
            
  for subgraph in graph.block_subgraphs():
    blocks.append(subgraph)
    if list(subgraph.block_subgraphs()):
      collect_all_blocks(subgraph, blocks)

  return blocks  
