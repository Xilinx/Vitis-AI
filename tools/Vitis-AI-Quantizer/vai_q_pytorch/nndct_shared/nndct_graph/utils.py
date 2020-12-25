

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

from typing import List

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph.base_graph import Graph


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
    if not n.in_quant_part:
      continue
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


def transformed_axis(src: str, dst: str, ndim: int, dim: int) -> int:
  """NCHW -> NHWC/ NHWC ->NCHW"""
  if ndim != 4:
    return dim
  if src == dst:
    return dim
  if src == "NCHW" and dst == "NHWC":
    return dim + [0, 2, -1, -1][dim]
  elif src == "NHWC" and dst == "NCHW":
    return dim + [0, 1, 1, -2][dim]
  
