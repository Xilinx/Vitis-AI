import networkx as nx
from networkx.algorithms import isomorphism, is_isomorphic

from typing import List, Optional, Set

class Graph(object):
  def __init__(self, name):
    self._graph = nx.DiGraph(name=name)
  
  def __str__(self):
    print_str = ""
    sorted_nodes = nx.algorithms.topological_sort(self._graph)
    for node in sorted_nodes:
      print_str += f"{node}:{'/'.join(self.get_node_types(node))}\n"

    return print_str

  def __eq__(self, other):
    def node_match(node_1, node_2):
      return node_1.get_types() == node_2.get_types()
    
    return is_isomorphic(self.graph, other.graph, node_match=isomorphism.generic_node_match("node", None, node_match))


  def add_node(self, id, obj):
    self._graph.add_node(id, node=obj)

  def add_edge(self, u, v):
    self._graph.add_edge(u, v)

  def children(self, n):
    return list(self._graph.successors(n))
  
  def parents(self, n):
    return list(self._graph.predecessors(n))

  def get_node_types(self, id):
    return self._graph.nodes[id]["node"].get_types()

  def set_node_types(self, id, types):
    return self._graph.nodes[id]["node"].set_types(types)

  def remove_node(self, n):
    return self._graph.remove_node(n)
  
  def remove_one_node(self, n):
    assert len(self.parents(n)) <= 1
    if len(self.parents(n)) == 0:
      self.remove_node(n)
    else:
      pn = self.parents(n)[0]
      for cn in self.children(n):
        self.remove_edge(n, cn)
        self.add_edge(pn, cn)
      self.remove_node(n)
  
  
  def remove_edge(self, u, v):
    self._graph.remove_edge(u, v)

  def visualize(self):
    import matplotlib.pyplot as plt
    lables = {}
    for node in self.nodes:
      lables[node] = "/".join(self.get_node_types(node))
    fig = plt.figure()
    # pos = nx.nx_pydot.graphviz_layout(self._graph)
    nx.draw_networkx(self._graph, labels=lables, node_color="white", alpha=.5)
    plt.savefig(f"{self._graph.name}.png")
    plt.close()
  
  @classmethod
  def copy(cls, original_graph):
    graph_copied = cls(original_graph.name)
    graph_copied._graph = original_graph.graph.copy()
    return graph_copied

    
  @property
  def nodes(self):
    for node in self._graph:
      yield node

  @property
  def op_types(self):
    op_types = set()
    for n in self._graph:
      op_types.update(self.get_node_types(n))
    return op_types

  @property
  def graph(self):
    return self._graph
  
  @property
  def name(self):
    return self._graph.name

  
class Node(object):
  def __init__(self, op_types: Set[str]):
    self._types = op_types
  
  def get_types(self):
    return self._types
  
  def set_types(self, types):
    self._types = types

  

  