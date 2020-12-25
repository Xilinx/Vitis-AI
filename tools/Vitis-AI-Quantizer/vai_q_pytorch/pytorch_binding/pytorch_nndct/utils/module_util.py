

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

from pytorch_nndct.parse.torch_graph import _NODE_NAME_SEPERATOR

def module_name_from_node(node):
  """Extract module name from parsed nndct node.

  Examples:
    A node's name is
    'ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/input.50',
    then extracted module name is 'layer4.1.conv1'
  """
  node_name = node if isinstance(node, str) else node.name
  if _NODE_NAME_SEPERATOR not in node_name:
    return None

  name_parts = []
  parts = node_name.split(_NODE_NAME_SEPERATOR)[1:-1]
  for part in parts:
    left_bracket = part.index('[')
    right_bracket = part.index(']')
    name_parts.append(part[left_bracket + 1 : right_bracket])
  return '.'.join(name_parts)

# Generalization of getattr
def get_module(model, submodule_key):
  tokens = submodule_key.split('.')
  cur_mod = model
  for s in tokens:
    cur_mod = getattr(cur_mod, s)
  return cur_mod

# Generalization of setattr
def set_module(model, submodule_key, module):
  tokens = submodule_key.split('.')
  sub_tokens = tokens[:-1]
  cur_mod = model
  for s in sub_tokens:
    cur_mod = getattr(cur_mod, s)

  setattr(cur_mod, tokens[-1], module)

