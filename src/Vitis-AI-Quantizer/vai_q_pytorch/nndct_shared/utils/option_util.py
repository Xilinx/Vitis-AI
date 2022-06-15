

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

from typing import TypeVar, NoReturn, Optional, Iterator, List
from .option_list import NndctOption
from .option_def import Option, T

def set_option_value(option_name: str, option_value: T) -> NoReturn:
  NndctOption.__dict__[option_name].value = option_value

def get_all_options() ->Iterator:
  for _, option in NndctOption.__dict__.items():
    if isinstance(option, Option):
      yield option
      

def add_valid_nndct_option(argv: List[str], option: str, cmd_position: int, framework: str)-> List[str]:
  
  def _set_nndct_option(option_name: str, option_value: str) -> bool:
    
    def _get_option_by_name() -> Optional[Option]:
      return NndctOption.__dict__.get(option_name, None)
    
    option = _get_option_by_name()
    if option is None: return False
    if option.framework != framework and option.framework != 'all': return False
  
    if option.dtype == bool:
      if option_value is None and option.action is None:
        return False
      elif option_value:
        if option_value not in ["True", "False"]:
          return False
        option_value = True if option_value == "True" else False
        option.value = option_value
        return True
      else:
        option.value = option_value
        return True       
    else:
      try:
        option_value = option.dtype(option_value)
      except ValueError:
        return False
      else:
        option.value = option_value
        return True       
        
  def _is_valid_option():
    return option.startswith("--")
  
  remove_item = []
  if not _is_valid_option(): return remove_item
  
  try:
    equal_symbol_idx = option.index("=")
  except ValueError:
    remove_next_cmd = False
    option_name = option[2:]
    if cmd_position == len(argv)-1:
      option_value = None
    elif argv[cmd_position + 1].startswith("--") or argv[cmd_position + 1].startswith("-"):
      option_value = None
    else:
      option_value = argv[cmd_position + 1]
      remove_next_cmd = True
      
    if _set_nndct_option(option_name, option_value):
      remove_item.append(option)
      if remove_next_cmd: remove_item.append(option_value)
  else:
    if equal_symbol_idx == len(option)-1: return remove_item
    option_name = option[2:equal_symbol_idx]    
    option_value = option[equal_symbol_idx+1:]
    if _set_nndct_option(option_name, option_value):
      remove_item.append(option)
      
  return remove_item