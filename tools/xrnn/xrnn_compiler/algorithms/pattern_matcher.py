

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
from typing import List, Tuple, Union
from abc import ABC

class PatternType(object):
  def __init__(self, pattern, action=None):
    self._pattern = pattern
    self._act = action

  @property
  def pattern(self):
    return self._pattern
  
  @pattern.setter
  def pattern(self, pattern):
    self._pattern = pattern

  @property
  def action(self):
    return self._act
  
  
class PatternMatcher(object):
  def __init__(self, patterns: List[PatternType]):
    self._ref_patterns = patterns
  
  def _get_min_pattern_len(self):
    return min([len(pat.pattern) for pat in self._ref_patterns])
  
  def get_max_pattern_len(self):
    return max([len(pat.pattern) for pat in self._ref_patterns])
  
  def get_multiple_patterns_indices(self, window_patterns):
    min_len = self._get_min_pattern_len()
    window_len = len(window_patterns)
    pattern_start_indices_map = {}
    while window_len >= min_len:
      for i in range(len(window_patterns)):
        if i + window_len <= len(window_patterns):
          sub_window = window_patterns[i: i + window_len]
          found, matched_pattern, pattern_id = self._find_pattern(sub_window)
          if found:
            if matched_pattern not in pattern_start_indices_map:
              indices = {i}
              pattern_start_indices_map[(pattern_id, matched_pattern)] = indices
            else:
              pattern_start_indices_map[(pattern_id, matched_pattern)].add(i)
      window_len = window_len - 1        
    return pattern_start_indices_map
  
  def _find_pattern(self, target_pattern) -> Tuple[bool, Union[PatternType, None], Union[int, None]]:
    for pattern_id, pattern_type in enumerate(self._ref_patterns):
      if len(pattern_type.pattern) == len(target_pattern):
        if "." in pattern_type.pattern:
          status = True
          for ref, target in zip(pattern_type.pattern, target_pattern):
            if ref == ".":
              continue
            if ref != target:
              status = False
              break
            
          if status:
            new_pattern_type = PatternType(pattern=target_pattern, action=pattern_type.action)
            return True, new_pattern_type, pattern_id
              
        if target_pattern == pattern_type.pattern:
          return True, pattern_type, pattern_id
    
    return False, None, None

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
    return [node.get_type() for node in self._window]
  
  
class GraphSearcher:
  def __init__(self, graph):
    self._graph = graph
    self._sliding_window = None
   
  def _find_node_from_type_and_apply_action(self, node, pattern_matcher, visited_nodes, node_sets=None):
    '''
    if pattern_matcher.get_max_pattern_len() == 1:
      if node and node.get_name() in visited_nodes:
        return 
    else:
      if node and node.get_name() in visited_nodes:
        if self._sliding_window.is_empty():
          return 
        elif node.get_name() == self._sliding_window.get_node_by_index(0).get_name():
          return
    '''
    #if node and node.get_name() in visited_nodes:
    #  return
        
    self._sliding_window.append_to_window(node)
    if set(node.get_name() for node in self._sliding_window._window).issubset(visited_nodes):
      return
    visited_nodes.add(node.get_name())
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
              matched_pattern.action(matched_pattern, node_set)
    
    #for cn in self._graph.children(node):
    if node.get_fanout_num()>0:
      for cn in node.get_fanout_ops():
        self._find_node_from_type_and_apply_action(cn, pattern_matcher, visited_nodes, node_sets)
      
    if node in self._sliding_window:
      self._sliding_window.remove(node)
         
  def find_nodes_from_type(self, type_pattern_list):
    pattern_removed = []
    graph_ops_type = set([op.get_type() for op in self._graph.get_ops()])
    for pattern_type in type_pattern_list:
      if not set(pattern_type.pattern).issubset(graph_ops_type):
        pattern_removed.append(pattern_type)
    [type_pattern_list.remove(x) for x in pattern_removed]
    if not type_pattern_list:
      return {}
    
    input_nodes = []
    for node in self._graph.get_ops():
      #if not node.in_nodes:
      if node.get_input_num() == 0:
        input_nodes.append(node)
        
    visited_nodes = set()
    node_sets = defaultdict(list)
    pattern_matcher = PatternMatcher(type_pattern_list)
    self._sliding_window = NodeTypeWindow(pattern_matcher.get_max_pattern_len())
    
    for node in input_nodes:
      self._find_node_from_type_and_apply_action(node, pattern_matcher, visited_nodes, node_sets)
        
    return node_sets