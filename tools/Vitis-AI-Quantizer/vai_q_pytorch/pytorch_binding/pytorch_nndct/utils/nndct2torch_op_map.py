

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

from nndct_shared.base import NNDCT_OP

_NNDCT_OP_2_TORCH_OP = {}  # Dict[str, str]

def add_mapping_item(nndct_op_name: str, torch_op_name: str):
  global _NNDCT_OP_2_TORCH_OP
  if nndct_op_name not in _NNDCT_OP_2_TORCH_OP:
    _NNDCT_OP_2_TORCH_OP[nndct_op_name] = torch_op_name

def get_nndct_op_2_torch_op_map():
  global _NNDCT_OP_2_TORCH_OP
  if len(_NNDCT_OP_2_TORCH_OP) == 0:
    raise Exception('please build the nndct_op -> torch_op map')
  return _NNDCT_OP_2_TORCH_OP

def get_torch_op_type(nndct_op_type):

  # if nndct_op_type in TORCH_UNSUPPORTED_NNDCTOPS:
  #   return nndct_op_type

  if get_nndct_op_2_torch_op_map().get(nndct_op_type, None) is None:
    raise Exception('please register the operator:"{}"'.format(nndct_op_type))
  else:
    return get_nndct_op_2_torch_op_map()[nndct_op_type]
