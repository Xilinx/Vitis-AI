try:
  from xir.graph import Graph
  from xir.op import Op
  from xir.tensor import NUMPY_DTYPE_TO_XIR_DTYPE, Tensor
  from xir.wrapper import XirDataType
except:
  raise Exception('please install xir package')

from typing import Dict, List, Optional, Any, NoReturn, Sequence
from pathlib import Path
from collections import ChainMap
import re
from nndct_shared.base import NNDCT_KEYS

import numpy as np

NndctQuantInfo = Dict[str, Dict[str, List[int]]]

_XMODEL_NAME_PATTERN = re.compile(r'[^0-9A-Za-z]')

class XGraph(object):

  def __init__(self, name: str):
    self._graph = Graph(name)
    self._const_ops: Dict[str, Op] = {}
    self._ops: Dict[str, Op] = {}

  def create_const_op(self, name: str, data: Optional[np.ndarray]) -> NoReturn:
    const_op = self._graph.create_const_op(name=name, tensor=data)
    if name in self._const_ops:
      raise RuntimeError('The const op {} has already in graph'.format(name))
    return const_op

  def create_normal_op(self,
                       name: str,
                       kind: str,
                       tensor: Optional[np.ndarray] = None,
                       attrs: Optional[Dict[str, Any]] = None,
                       input_ops: Optional[List[Op]] = None) -> Op:
    op = self._graph.create_op(
        name=name, kind=kind, tensor=tensor, attrs=attrs, input_ops=input_ops)
    return op

  def create_fixed_const_op(self, name: str, data: np.ndarray,
                            quant_info: NndctQuantInfo) -> Op:

    formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", name)
    const_op = self.create_const_op(formal_name, data)
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
    fixed_op = self.create_fix_op(op, name, quant_info)
    return fixed_op if fixed_op else op

  def create_fix_op(self, input: Op, key_name: str,
                    quant_info: NndctQuantInfo) -> Optional[Op]:

    def _get_fix_info(name: str, quant_info: NndctQuantInfo) -> Sequence[int]:
      combinded_fix_infos = ChainMap(quant_info['params'], quant_info['blobs'])
      if name in combinded_fix_infos.keys():
        return combinded_fix_infos[name]
      else:
        return None, None

    if NNDCT_KEYS.FIX_OP_SUFFIX in input.get_name():
      raise RuntimeError("The consecutive fix ops in graph is forbidden!")

    if not isinstance(quant_info, dict):
      return None

    bit_width, fix_point = _get_fix_info(key_name, quant_info)
    if bit_width is None or fix_point is None:
      return None

    attrs: Dict[str, Any] = {}
    attrs['fix_point'] = fix_point
    attrs['bit_width'] = bit_width
    attrs['round_mode'] = "DPU_ROUND"
    input_ops: Dict[str, List[Op]] = {}
    input_ops['input'] = [input]
    fix_op = self.create_normal_op(
        input.get_name() + NNDCT_KEYS.FIX_OP_SUFFIX,
        'fix',
        attrs=attrs,
        input_ops=input_ops)
    return fix_op

  def get_op_by_name(self, name: str) -> Op:
    formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", name)
    try:
      op = self._graph.get_op(formal_name + NNDCT_KEYS.FIX_OP_SUFFIX)
    except AttributeError:
      op = self._graph.get_op(formal_name)
    return op

  def export_to_xmodel(self, fname: str) -> NoReturn:
    self._graph.serialize(Path(fname))

  def export_to_img(self, fname: str) -> NoReturn:
    self._graph.dump(Path(fname))

  @property
  def graph(self):
    return self._graph
