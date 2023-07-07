

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

from collections import defaultdict, deque

from nndct_shared.utils import PatternMatcher
from abc import ABC


class WindowBase(ABC):
  def __init__(self, window_len):
    self._window = deque(maxlen=window_len)
 
  def get_node_by_index(self, idx):
    return list(self._window)[idx]
  
  def __contains__(self, v):
    return True if v in self._window else False
  
  def remove(self, v):
    self._window.remove(v)

  def append_to_window(self, v):
    self._window.append(v)
  
  def clear(self):
    self._window.clear()
  
  def is_empty(self):
    return len(self._window) == 0
  
  
class NodeTypeWindow(WindowBase):
  def __init__(self, window_len):
    super().__init__(window_len)
    
  def get_type_pattern(self):
    return [node.op_type for node in self._window]
  
  
class GraphSearcher:
  def __init__(self, graph):
    self._graph = graph
    self._sliding_window = None
   
  def _find_node_from_type_and_apply_action(self, node, pattern_matcher, visited_nodes, node_sets=None):
    
    if pattern_matcher.get_max_pattern_len() == 1:
      if node and node in visited_nodes:
        return 
    else:
      if node and node in visited_nodes:
        if self._sliding_window.is_empty():
          return 
        elif node is self._sliding_window.get_node_by_index(0):
          return
        
    visited_nodes.add(node)
    self._sliding_window.append_to_window(node)
    window_pattern = self._sliding_window.get_type_pattern()
    matched_pattern_indices_dict = pattern_matcher.get_multiple_patterns_indices(window_pattern)
    if matched_pattern_indices_dict:
      for pattern_id, matched_pattern in matched_pattern_indices_dict.keys():
        for idx in matched_pattern_indices_dict[(pattern_id, matched_pattern)]:
          node_set = [self._sliding_window.get_node_by_index(i) 
                       for i in range(idx, idx + len(matched_pattern.pattern))]
          if node_sets is not None and node_set not in node_sets[pattern_id]:
            node_sets[pattern_id].append(node_set)
            if matched_pattern.action:
              matched_pattern.action(matched_pattern, node_set, graph=self._graph)
    
    for cn in self._graph.children(node):
      self._find_node_from_type_and_apply_action(cn, pattern_matcher, visited_nodes, node_sets)
      
    if node in self._sliding_window:
      self._sliding_window.remove(node)
         
  def find_nodes_from_type(self, type_pattern_list):
    pattern_removed = []
    for pattern_type in type_pattern_list:
      if not set(pattern_type.pattern).issubset(self._graph.op_types):
        pattern_removed.append(pattern_type)
    [type_pattern_list.remove(x) for x in pattern_removed]
    if not type_pattern_list:
      return {}
    
    input_nodes = []
    for node in self._graph.nodes:
      if not node.in_nodes:
        input_nodes.append(node)
        
    visited_nodes = set()
    node_sets = defaultdict(list)
    pattern_matcher = PatternMatcher(type_pattern_list)
    self._sliding_window = NodeTypeWindow(pattern_matcher.get_max_pattern_len())
    
    for node in input_nodes:
      self._find_node_from_type_and_apply_action(node, pattern_matcher, visited_nodes, node_sets)
        
    return node_sets
