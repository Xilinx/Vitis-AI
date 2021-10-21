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


import re





def _match_optional(match):
    typing_str = match[1]
    return convert_type_str(typing_str) + '?'

def _match_list(match):
    typing_str = match[1]
    return convert_type_str(typing_str) + '[]'

def _match_tuple(match):
    x_typing_str = match[1]
    y_typing_str = match[2]
    return '(' + convert_type_str(x_typing_str) + ', ' + convert_type_str(y_typing_str) + ')'

def _match_dict(match):
    x_typing_str = match[1]
    y_typing_str = match[2]
    return 'Dict(' + convert_type_str(x_typing_str) + ', ' + convert_type_str(y_typing_str) + ')'

def _match_brackets(match):
    outer_type_str = match[1]
    inner_type_str = match[2]
    return outer_type_str + '(' + convert_type_str(inner_type_str) + ')'

patterns_rlp_pair = {
    'Optional' : (re.compile(r'Optional\[(.*)\]'), _match_optional),
    'List' : (re.compile(r'List\[(.*)\]'), _match_list),
    'Tuple': (re.compile(r'Tuple\[(.*),\s*(.*)\]'), _match_tuple),
    'Dict': (re.compile(r'Dict\[(.*),\s*(.*)\]'), _match_dict),
    'Future': (re.compile(r'(\w+)\[(.*)\]'), _match_brackets)
}

def convert_type_str(typing_str):
    if '[' not in typing_str:
        return typing_str
    special_type = typing_str.split('[')[0]
    pattern, rlp_fn = patterns_rlp_pair[special_type]
    return pattern.sub(rlp_fn, typing_str)

class SchemaHelper(object):
  def __init__(self, func_schema):
    self._func_schema = func_schema
    
    
  def get_arguments(self):
    return self._func_schema.arguments
  
  def arg_name(self, arg):
    return arg.name
  
  def arg_type(self, arg):
    return str(arg.type)
  
  def toString(self):
   
    schema_str = "{}({}) -> {}".format(self.op_name,
                                       self._emit_args(self._func_schema.arguments),
                                       self._emit_rets(self._func_schema.returns))
    return schema_str
  

  def normalized_type(self, arg):
    s = self.arg_type(arg)
    s = s.replace("number", "Scalar")
    s = convert_type_str(s)
    return s
  
  def _emit_arg(self, arg):
      v = "{} {}".format(self.normalized_type(arg), arg.name)
      default = arg.default_value
      if default is not None:
        if not isinstance(default, str):
          v = "{}={}".format(v, str(default))
        else:
          v = "{}={}".format(v, f"'{default}'")

      return v

  def _emit_args(self, arguments):
      return ", ".join(self._emit_arg(arg) for i, arg in enumerate(arguments))

  def _emit_ret(self, ret):
      return self.normalized_type(ret)    

  def _emit_rets(self, returns):
      if len(returns) == 1:
          return self._emit_ret(returns[0])
      return "({})".format(", ".join(self._emit_ret(r) for r in returns))

  @property
  def op_name(self):
    return self._func_schema.name
  
  
  
    


    