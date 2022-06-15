

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

try:
  from xir import Graph
  from xir import Op
  from xir import Tensor
except:
  raise Exception('please install xir package')

import subprocess
from typing import Dict, List, Optional, Any, NoReturn, Sequence
from pathlib import Path
from collections import ChainMap
import re
from nndct_shared.base import NNDCT_KEYS
from nndct_shared.utils import NndctScreenLogger, ExportXmodelError

import numpy as np

NndctQuantInfo = Dict[str, Dict[str, List[int]]]

_XMODEL_NAME_PATTERN = re.compile(r'[^0-9A-Za-z]')

class XGraph(object):

  def __init__(self, name: str):
    self._graph = Graph(name)
    self._const_ops: Dict[str, Op] = {}
    self._ops: Dict[str, Op] = {}

  
  def _check_inputs(self, input_ops):
    if any([ip is None for ip in input_ops]):
      raise RuntimeError('The input op is `None`, please check graph.')
    
  def create_const_op(self, name: str, data: Optional[np.ndarray]) -> NoReturn:
    const_op = self._graph.create_const_op(name, data)
    if name in self._const_ops:
      raise RuntimeError('The const op {} has already in graph'.format(name))
    return const_op
  
  
  def create_input_transpose_ops(self, input_list: List[Op], input_tensors: 'List[base_tensor::Tensor]'):
    t_ops = []
    for i, (input, tensor) in enumerate(zip(input_list, input_tensors)):
      if tensor.ndim in [4, 5]:
        attrs: Dict[str, Any] = {}
        attrs['order'] = [0, 3, 1, 2] if tensor.ndim == 4 else [0, 4, 3, 1, 2]
        input_ops: Dict[str, List[Op]] = {}
        input_ops['input'] = [input]
        op_name = input.get_name() + NNDCT_KEYS.TRANSPOSE_OP_SUFFIX
        t_op = self.get_op_by_name(op_name)
        if t_op is None:
          t_op = self.create_normal_op(
              op_name,
              'transpose',
              attrs=attrs,
              input_ops=input_ops)

        t_ops.append(t_op)
      else:
        t_ops.append(input)
    return t_ops

  def create_normal_op(self,
                       name: str,
                       kind: str,
                       tensor: Optional[np.ndarray] = None,
                       attrs: Optional[Dict[str, Any]] = None,
                       input_ops: Optional[List[Op]] = None) -> Op:
    if input_ops is not None:
      self._check_inputs(input_ops['input'])
    op = self._graph.create_op(name, kind, attrs, input_ops)
    return op

  def create_fixed_const_op(self, name: str, data: np.ndarray,
                            quant_info: NndctQuantInfo) -> Op:

    formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", name)
    const_op = self.create_const_op(formal_name, data)
    # print(const_op.get_name(), "const", const_op.get_output_tensor().dims)    
    fixed_const_op = self.create_fix_op(const_op, name, quant_info)
    return fixed_const_op if fixed_const_op else const_op

  def create_fixed_normal_op(self,
                             name: str,
                             kind: str,
                             quant_info: NndctQuantInfo,
                             tensor: Optional[np.ndarray] = None,
                             attrs: Optional[Dict[str, Any]] = None,
                             input_ops: Optional[List[Op]] = None) -> Op:
    formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", name)
    op = self.create_normal_op(
        name=formal_name,
        kind=kind,
        tensor=tensor,
        attrs=attrs,
        input_ops=input_ops)
    # print(op.get_name(), kind, op.get_output_tensor().dims)
    post_fixed_op = self.create_fix_op(op, name, quant_info)
    return post_fixed_op if post_fixed_op else op

  def create_input_fix_ops(self, input_list: List[Op], key_name: str, quant_info: NndctQuantInfo):
    pre_fix_ops = []
    for i, op in enumerate(input_list):
      pre_fix_op = self.create_fix_op(op, key_name, quant_info, id=i, post_fix=False)
      if pre_fix_op:
        pre_fix_ops.append(pre_fix_op)
      else:
        pre_fix_ops.append(op)
    return pre_fix_ops

  
  def create_fix_op(self, input: Op, key_name: str,
                    quant_info: NndctQuantInfo, id: Optional[int] = None, post_fix: bool = True) -> Optional[Op]:

    def _get_fix_info(name: str, quant_info: NndctQuantInfo) -> Sequence[int]:
      if post_fix:
        combinded_fix_infos = ChainMap(dict(quant_info['param']), dict(quant_info['output']))
      else:
        combinded_fix_infos = quant_info['input']
      if name in combinded_fix_infos.keys():
        return combinded_fix_infos[name]
      else:
        return None, None

    # if NNDCT_KEYS.FIX_OP_SUFFIX in input.get_name():
    #   raise RuntimeError("The consecutive fix ops in graph is forbidden!")

    if not isinstance(quant_info, dict):
      return None

    bit_width, fix_point = _get_fix_info(key_name, quant_info)
    if bit_width is None or fix_point is None:
      return None

    attrs: Dict[str, Any] = {}
    attrs['fix_point'] = fix_point
    attrs['bit_width'] = bit_width
    attrs['round_mode'] = "DPU_ROUND"
    attrs['if_signed'] = True
    input_ops: Dict[str, List[Op]] = {}
    input_ops['input'] = [input]
    if post_fix:
      op_name = input.get_name() + NNDCT_KEYS.FIX_OP_SUFFIX
    else:
      formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", key_name)
      if id is not None:
        op_name = formal_name + NNDCT_KEYS.PRE_FIX_OP_SUFFIX + f"_i{id}"
      else:
        op_name = formal_name + NNDCT_KEYS.PRE_FIX_OP_SUFFIX

    fix_op = self.create_normal_op(
        op_name,
        'fix',
        attrs=attrs,
        input_ops=input_ops)
    return fix_op

  def get_op_by_name(self, name: str) -> Op:
    formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", name)
    op = self._graph.get_op(formal_name + NNDCT_KEYS.FIX_OP_SUFFIX)
    if op is None:
      op = self._graph.get_op(formal_name)
    return op
  
  def get_op_output_shape(self, name: str) -> List[int]:
    op = self.get_op_by_name(name)
    if op:
      return op.get_output_tensor().dims
    else:
      NndctScreenLogger().warning("{name} is not in xmodel. Please check it.")
      
  def export_to_xmodel(self, fname: str) -> NoReturn:
    fname += NNDCT_KEYS.XMODEL_SUFFIX
    try:
      self._graph.serialize(fname)
    except Exception:
      raise ExportXmodelError(self._graph.get_name())
    else:
      NndctScreenLogger().info(f"=>Successfully convert '{self._graph.get_name()}' to xmodel.({fname})")

  def export_to_img(self, fname: str) -> NoReturn:
    fname += NNDCT_KEYS.XMODEL_IMAGE_SUFFIX
    try:
      shell_command = "which dot"
      proc = subprocess.Popen(shell_command, stdout=subprocess.PIPE, shell=True)
      try:
        outs, errs = proc.communicate(timeout=2)
      except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        NndctScreenLogger().error(f"{errs}")
        raise
      if outs:
        self._graph.save_as_dot(fname)
      else:
        NndctScreenLogger().warning(("Can't find dot command in the system, please install it."
                                     " Otherwise, the xmodel image will not be generated."))
    except Exception as e:
      NndctScreenLogger().warning(f"Failed to generate xmodel image!({str(e)})")

  @property
  def graph(self):
    return self._graph
