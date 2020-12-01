

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

from typing import List, Tuple, Union
    
    
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
    