from networkx.algorithms import isomorphism
from nndct_shared.base import NNDCT_OP
from .graph import Graph, Node


class Match(object):
  def __init__(self):
    self.quant_node = []
    self.nodeset = []
  
  @classmethod
  def create_match(cls, pattern, matched_subgraph):
    match = cls()
    nndct_fix = {NNDCT_OP.FIX}
    fix_anchors = []
    for node in pattern.nodes:
      if pattern.get_node_types(node) == nndct_fix and len(pattern.parents(node)) != 0:
        fix_anchors.append(pattern.parents(node)[0])

    sorted_subgraph = sorted(matched_subgraph.items(), key=lambda x: x[0].topo_position)
    for node, template_name in sorted_subgraph:
      if template_name in fix_anchors:
        match.quant_node.append(node)
      match.nodeset.append(node)
    return match

class SubgraphMatcher(object):
  def __init__(self, nndct_graph):
    self._nndct_graph = nndct_graph
    self._graph = self._create_graph_from_nndct_graph()

  def findPatternMatches(self, pattern, node_match):
    matched_subgraphs = self._find_matched_subgraphs(pattern, node_match)
    for i in range(len(matched_subgraphs)):
      subgraph = matched_subgraphs[i]
      new_subgraph = {self._nndct_graph.node(k):v for k, v in subgraph.items()}
      matched_subgraphs[i] = new_subgraph
  
    return matched_subgraphs

  def _find_matched_subgraphs(self, pattern, node_match):
    GM = isomorphism.DiGraphMatcher(self._graph.graph, pattern.graph, node_match=isomorphism.generic_node_match("node", None, node_match))
    matched_subgraphs = []
    for subgraph in GM.subgraph_isomorphisms_iter():
      matched_subgraphs.append(subgraph)

    matched_subgraphs = self._remove_node_usage_out_of_match(pattern, matched_subgraphs)    
    # matched_subgraphs = self._remove_shared_subgraphs(matched_subgraphs)
    return matched_subgraphs
  
  def _remove_shared_subgraphs(self, matched_subgraphs):
    subgraph_id_to_mininum_topo_id = {}
    for id, subgraph in enumerate(matched_subgraphs):
      subgraph_id_to_mininum_topo_id[id] = min([self._nndct_graph.node(k).topo_position for k, _ in subgraph.items()])
    
    sorted_subgraph_id = sorted(subgraph_id_to_mininum_topo_id.items(), key=lambda x: x[1])
    visited_node = set()
    subgraphs = []
    for id in sorted_subgraph_id:
      node_set = {k for k, _ in matched_subgraphs.items()}
      if not (node_set & visited_node):
        visited_node.update(node_set)
        subgraphs.append(matched_subgraphs[id])
    
    return subgraphs
  
  def _remove_node_usage_out_of_match(self, pattern, matched_subgraphs):
    subgraphs = []
    for subgraph in matched_subgraphs:
      matched_nodeset = {node_name for node_name, _ in subgraph.items()}
      out_of_match = False
      for node_name, template_name in subgraph.items():
        if len(pattern.children(template_name)) != 0 and not(set(self._nndct_graph.node(node_name).out_nodes).issubset(matched_nodeset)):
          out_of_match = True
          break
      if out_of_match is False:
        subgraphs.append(subgraph)
    return subgraphs

  def _create_graph_from_nndct_graph(self):
    pattern_graph = Graph(self._nndct_graph.name)
    for node in self._nndct_graph.nodes:
      pattern_graph.add_node(node.name, node)
    
    for node in self._nndct_graph.nodes:
      for in_node in node.in_nodes:
        pattern_graph.add_edge(in_node, node.name)
      for out_node in node.out_nodes:
        pattern_graph.add_edge(node.name, out_node)
    return pattern_graph

  