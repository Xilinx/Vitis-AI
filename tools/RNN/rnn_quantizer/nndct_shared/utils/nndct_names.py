

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

from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP, NNDCT_DEBUG_LVL
from .log import nndct_debug_print

def get_split_sym(model_type):
  if model_type == 'Nndct':
    return '_'
  elif model_type in ['tensorflow', 'tf-keras']:
    return '/'
  elif model_type == 'torch':
    return '.'
  raise Exception("can not find split symbol for model_type " + model_type)

def default_scoped_name(obj):
  if isinstance(obj, str):
    name = obj
  else:
    name = obj.name
  return '/'.join(name.split('/')[:-1]), name.split('/')[-1]

def get_default_name(obj):
  if isinstance(obj, str):
    return obj
  name = getattr(obj, 'name', None)
  if not name:
    raise Exception("{} has no attribute name, please check!".format(obj))
  return name.split(':')[0]

def default_legal_name(name):
  return name.replace('.', 'DOT').replace('/', 'SPL')

def derive_scope_name(name, scope, split_sym, offset=0):
  name_list = name.split(split_sym)
  for idx in range(len(name_list)):
    if name_list[idx].startswith(scope):
      base_idx = idx
      break
  return split_sym.join(name_list[:base_idx - offset])

def reverse_default_legal_name(name):
  return name.replace('DOT', '.').replace('SPL', '/')

def remove_prefix(obj, prefix):
  if obj is None:
    return obj
  if prefix is None or prefix == '':
    return obj
  if isinstance(prefix, str):
    if isinstance(obj, str) and len(prefix) > 0 and obj.startswith(prefix):
      obj = obj[len(prefix):]
    elif isinstance(obj, dict):
      obj = {k: remove_prefix(v, prefix) for k, v in obj.items()}
    elif isinstance(obj, list):
      obj = [remove_prefix(item, prefix) for item in obj]
    return obj
  elif isinstance(prefix, list):
    for pre in prefix:
      obj = remove_prefix(obj, pre)
    return obj
  else:
    raise Exception('prefix {} is not string or list!'.format(prefix))

def remove_suffix(obj, suffix):
  if obj is None:
    return obj
  if suffix is None or suffix == '':
    return obj
  if isinstance(suffix, str):
    if isinstance(obj, str) and len(suffix) > 0 and obj.endswith(suffix):
      obj = obj[:-len(suffix)]
    elif isinstance(obj, dict):
      obj = {k: remove_suffix(v, suffix) for k, v in obj.items()}
    elif isinstance(obj, list):
      obj = [remove_suffix(item, suffix) for item in obj]
    return obj
  elif isinstance(suffix, list):
    for suf in suffix:
      obj = remove_suffix(obj, suf)
    return obj
  else:
    raise Exception('suffix {} is not string or list!'.format(suffix))

def remove_trans_scp_prefix(name, scp=None):
  name = remove_prefix(name, scp)
  if name.startswith(NNDCT_KEYS.TRANS_SCOPE):
    name = '/'.join(name.split('/')[1:])
  return name

def scoped_untrans_name(name, scp):
  org_name = remove_trans_scp_prefix(name, scp)
  return scp + org_name

def scoped_trans_name(name, scp):
  org_name = remove_prefix(name, scp)
  if org_name.startswith(NNDCT_KEYS.TRANS_SCOPE):
    return scp + org_name
  else:
    return scp + NNDCT_KEYS.TRANS_SCOPE + '/' + org_name

def map_output_and_node(output, node_or_name, model_type):
  if node_or_name is None:
    return
  if isinstance(node_or_name, str):
    node_name = node_or_name
  else:
    node_name = node_or_name.name
  node_name = remove_trans_scp_prefix(node_name)

  def _do_map(output_name, node_name):
    if not output_name == node_name:
      if not GLOBAL_MAP.get_ele(NNDCT_KEYS.OUTPUT_TO_NODE_MAP):
        GLOBAL_MAP.set_map(NNDCT_KEYS.OUTPUT_TO_NODE_MAP, {})
      if not GLOBAL_MAP.get_ele(NNDCT_KEYS.NODE_TO_OUTPUT_MAP):
        GLOBAL_MAP.set_map(NNDCT_KEYS.NODE_TO_OUTPUT_MAP, {})
      #map output to node
      output_to_node_map = GLOBAL_MAP.get_ele(NNDCT_KEYS.OUTPUT_TO_NODE_MAP)
      if not output_name in output_to_node_map:
        nndct_debug_print(
            "<map_output_and_node> map out {} and node{}".format(
                output_name, node_name),
            level=NNDCT_DEBUG_LVL.BUILD_GRAPH)
        output_to_node_map[output_name] = node_name
      else:
        assert output_to_node_map[
            output_name] == node_name, "restored node name for output_name {} is {}, meet new node name {}".format(
                output_name, output_to_node_map[output_name], node_name)
      #add output to list keyed by node_name
      node_to_output_map = GLOBAL_MAP.get_ele(NNDCT_KEYS.NODE_TO_OUTPUT_MAP)
      if not node_name in node_to_output_map:
        node_to_output_map[node_name] = [output_name]
      else:
        node_to_output_map[node_name].append(output_name)

  if isinstance(output, str):
    _do_map(output, node_name)


def node_from_output(output_name, model_type):
  if model_type == 'Nndct':
    return output_name
  if model_type == 'tensorflow':
    output_name = output_name.split(':')[0]
  elif model_type == 'torch':
    if output_name.split('_')[-1] in ['backward', 'forward']:
      output_name = ''.join(output_name.split('_')[:-1])
  else:
    raise KeyError("node_from_output is not available for model type " +
                   str(model_type))
  output_map = GLOBAL_MAP.get_ele(NNDCT_KEYS.OUTPUT_TO_NODE_MAP)
  if output_map and output_name in output_map:
    return output_map[output_name]
  return output_name

def get_output_from_node(node_name, idx=-1):
  node_map = GLOBAL_MAP.get_ele(NNDCT_KEYS.NODE_TO_OUTPUT_MAP)
  if node_map and node_name in node_map:
    return node_map[node_name][idx]
  return node_name

def get_all_outputs_from_node(node_name):
  node_map = GLOBAL_MAP.get_ele(NNDCT_KEYS.NODE_TO_OUTPUT_MAP)
  if node_map and node_name in node_map:
    return node_map[node_name]
  return node_name
