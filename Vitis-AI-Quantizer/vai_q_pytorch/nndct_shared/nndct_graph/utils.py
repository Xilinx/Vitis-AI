
from typing import List

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph.base_graph import Graph


def should_pass_by_elemwise_node(node, graph):
  return node.op.type in [NNDCT_OP.ADD, NNDCT_OP.MULTIPLY] and all(
      p.op.type == NNDCT_OP.SHAPE for p in graph.parents(node.name))


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
    if POpType is None:
      return True
    elif node.op.type == POpType:
      return True
    return False

  for n in graph.nodes:
    if groups[n.name][0] == n.name and n.op.type == OpType and \
        len(graph.parents(n.name)) == 1 and __is_valid_parent(graph.parents(n.name)[0]):
      start_node = groups[graph.parents(n.name)[0].name][0]
      groups = glue_group_members(graph, groups, start_node, n.name)
      # print('---- Grouping node %s and %s' % (start_node, n.name))
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
