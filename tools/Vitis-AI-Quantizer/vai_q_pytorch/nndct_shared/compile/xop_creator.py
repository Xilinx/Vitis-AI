

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
import itertools
from typing import List, Dict, Any, NoReturn, Tuple
import numpy as np
from functools import partial
from nndct_shared.base import NNDCT_OP, NNDCT_KEYS
from nndct_shared.nndct_graph import Tensor, Node, transformed_axis
from .xgraph import XGraph
from xir import Op

NndctQuantInfo = Dict[str, Dict[str, List[int]]]


class _Converter:
  _nndct2xir_type = {np.float32: "FLOAT32", 
                np.float64: "FLOAT64", 
                np.int64: "INT64"
                }
  
  _nndct2numpy_type = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64
    
  }
  
  _pad_mode = {"pad_mode": {0: "FLOOR",
                            1: "CEIL",
                            2: "SAME",
                            3: "VALID"}
  }
  
  _nndct2xir_value = {NNDCT_OP.CONV2D: _pad_mode,
                      NNDCT_OP.DEPTHWISE_CONV2D: _pad_mode,
                      NNDCT_OP.CONVTRANSPOSE2D: _pad_mode,
                      NNDCT_OP.CONV3D: _pad_mode,
                      NNDCT_OP.CONVTRANSPOSE3D: _pad_mode,
                      NNDCT_OP.MAX_POOL: _pad_mode,
                      NNDCT_OP.AVG_POOL: _pad_mode,
                      NNDCT_OP.ADAPTIVEAVGPOOL2D: _pad_mode,
                      NNDCT_OP.PAD: {"mode": {0: "CONSTANT", 1: "REFLECT", 2: "SYMMETRIC"}}
                     
  }
                      
                      
  
  @classmethod
  def to_xir_dtype(cls, numpy_dtype):
    if not hasattr(numpy_dtype, "type"):
      raise TypeError("Only accept numpy dtype when convert to xir.")
    return cls._nndct2xir_type[numpy_dtype.type]
  
  @classmethod
  def to_xir_attr_value(cls, node_op_type, nndct_attr_name: str, nndct_attr_value: Any):
    if node_op_type not in cls._nndct2xir_value or nndct_attr_name not in cls._nndct2xir_value[node_op_type]:
      return nndct_attr_value
    else:
      return cls._nndct2xir_value[node_op_type][nndct_attr_name][nndct_attr_value]
    
  @classmethod
  def to_numpy_dtype(cls, nndct_dtype):
    return cls._nndct2numpy_type[nndct_dtype]
    


def _get_xir_attr_from_node(node: Node):
  attrs = None
  if len(node.op.attrs) > 0:
    attrs: Dict[str, Any] = {}
    for attr_name, attr_value in node.op.attrs.items():
      if node.op.is_xir_attr(attr_name):
        attrs[attr_name.value] = _Converter.to_xir_attr_value(node.op.type, attr_name.value, attr_value.value)
  return attrs

def _pack(xgraph: XGraph, node: Node, pack_name: str, packed_item: List[Any],
          quant_config: NndctQuantInfo) -> Tuple[Op, List[Op]]:
  """
  pack items into stack op
  """
  pack_list = []
  pack_input_ops: Dict[str, List[Op]] = {}
  for i, item in enumerate(packed_item):
    if isinstance(item, Tensor):
      pack_list.append(xgraph.get_op_by_name(item.node.name))
    else:
      # dtype = np.int64 if isinstance(item, int) else np.float64
      dtype = np.float32
      const_op = xgraph.create_fixed_const_op(
          name=node.name + f"_{pack_name}_attr[{i}]",
          data=np.array([item], dtype=dtype),
          quant_info=quant_config)
      pack_list.append(const_op)

  pack_input_ops["input"] = pack_list
  attrs: Dict[str, Any] = {}
  attrs["axis"] = 0
  sub_op_pack = xgraph.create_fixed_normal_op(
      node.name + f"_{pack_name}_i0",
      "stack",
      quant_config,
      attrs=attrs,
      input_ops=pack_input_ops)
  return sub_op_pack, pack_list


def _sub(xgraph, name, operand1, operand2, quant_config):
  if isinstance(operand1, Tensor) and (not isinstance(operand2, Tensor)):
    operand1 = xgraph.get_op_by_name(operand1.node.name)
    operand2 = np.ones(operand1.shape, dtype=type(operand2)) * operand2
    operand2 = xgraph.create_const_op(f"{name}_other", operand2)
  elif isinstance(operand2, Tensor) and (not isinstance(operand1, Tensor)):
    operand1 = np.ones(operand2.shape, dtype=type(operand1)) * operand1
    operand1 = xgraph.create_const_op(f"{name}_input", operand1)
    operand2 = xgraph.get_op_by_name(operand2.node.name)
  else:
    operand1 = xgraph.get_op_by_name(operand1.node.name)
    operand2 = xgraph.get_op_by_name(operand2.node.name)

  input_ops: Dict[str, List[Op]] = {}
  input_ops["input"] = [operand1, operand2]
  input_ops["input"] = xgraph.create_input_fix_ops(input_ops["input"], name, quant_config)
  xgraph.create_fixed_normal_op(name, "sub", quant_config, input_ops=input_ops)


def data_xop(xgraph: XGraph, node: Node,
             quant_config: NndctQuantInfo) -> NoReturn:
  
  shape = node.out_tensors[0].shape
  
  out_tensor = np.zeros(shape, dtype=np.float32)
  attrs: Dict[str, Any] = {}
  attrs["shape"] = shape
  attrs["data_type"] = _Converter.to_xir_dtype(out_tensor.dtype)
  xgraph.create_fixed_normal_op(
      node.name, "data", quant_config, tensor=out_tensor, attrs=attrs)


def const_xop(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  data = node.node_attr(node.op.AttrName.DATA)
  data_type = np.dtype(node.out_tensors[0].dtype)
  data_type = np.float32 if data_type == np.float64 else data_type
  if not isinstance(data, list):
    data = [data]
  
  data = np.array(data, dtype=data_type)
  xgraph.create_fixed_const_op(name=node.name, 
                               data=data, 
                               quant_info=quant_config)
  

def shape(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  r""" nndct shape is a macro operator, including shape, stridedslice 
      """
  # raise NotImplementedError("shape")
  input_list = []
  shape_input_ops: Dict[str, List[Op]] = {}
  for input in node.in_nodes:
    input_op = xgraph.get_op_by_name(input)
    input_list.append(input_op)
  shape_input_ops["input"] = input_list

  sub_op_shape = xgraph.create_fixed_normal_op(
      node.name + "_i0", "shape", quant_config, input_ops=shape_input_ops)

  attrs: Dict[str, Any] = {}
  strided_slice_input_ops: Dict[str, List[Op]] = {}
  strided_slice_input_ops["input"] = [sub_op_shape]
  dim = node.node_attr(node.op.AttrName.AXIS)
  attrs["begin"] = [dim]
  attrs["end"] = [dim + 1]
  xgraph.create_fixed_normal_op(
      node.name,
      "strided_slice",
      quant_config,
      attrs=attrs,
      input_ops=strided_slice_input_ops)


def reshape(xgraph: XGraph, node: Node,
            quant_config: NndctQuantInfo) -> NoReturn:
  r""" nndct reshape is a macro operator, including pack, reshape
      """
  # raise NotImplementedError("reshape")
  if node.in_tensors[0].layout == Tensor.Layout.NHWC:
    shape = node.node_attr(node.op.AttrName.SHAPE)
    sub_op_pack, pack_list = _pack(xgraph, node, "shape", shape, quant_config)
    input_ops: Dict[str, List[Op]] = {}
    input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name, "reshape", quant_config, input_ops=input_ops)
  elif node.out_tensors[0].ndim == 4:
    shape = node.node_attr(node.op.AttrName.SHAPE)
    sub_op_pack, pack_list = _pack(xgraph, node, "shape", shape, quant_config)
    input_ops: Dict[str, List[Op]] = {}
    input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "reshape", quant_config, input_ops=input_ops)

    attrs: Dict[str, Any] = {}
    # NCHW -> NHWC
    attrs["order"] = [0, 2, 3, 1]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]
    xgraph.create_fixed_normal_op(
        node.name, "transpose", quant_config, attrs=attrs, input_ops=input_ops)
  elif node.in_tensors[0].ndim == 4:
    shape = node.node_attr(node.op.AttrName.SHAPE)
    sub_op_pack, pack_list = _pack(xgraph, node, "shape", shape, quant_config)
    attrs: Dict[str, Any] = {}
    # NHWC -> NCHW
    attrs["order"] = [0, 3, 1, 2]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "transpose", quant_config, attrs=attrs, input_ops=input_ops)
    
    input_ops: Dict[str, List[Op]] = {}
    input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]
    xgraph.create_fixed_normal_op(
        node.name, "reshape", quant_config, input_ops=input_ops)
    
  else:
    shape = node.node_attr(node.op.AttrName.SHAPE)
    sub_op_pack, pack_list = _pack(xgraph, node, "shape", shape, quant_config)
    input_ops: Dict[str, List[Op]] = {}
    input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name, "reshape", quant_config, input_ops=input_ops)

   

def rsub(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  operand1, operand2 = node.node_attr(node.op.AttrName.INPUT), node.node_attr(
      node.op.AttrName.OTHER)

  _sub(xgraph, node.name, operand1, operand2, quant_config)


def sub(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  operand1, operand2 = node.node_attr(node.op.AttrName.INPUT), node.node_attr(
      node.op.AttrName.OTHER)
  _sub(xgraph, node.name, operand1, operand2, quant_config)


def default_xop(xop_type: str, xgraph: XGraph, node: Node,
                quant_config: NndctQuantInfo) -> NoReturn:

  attrs = _get_xir_attr_from_node(node)
  
  input_ops: Dict[str, List[Op]] = {}
  if node.has_bound_params():
    for param_name, param_tensor in node.op.params.items():
      param = xgraph.get_op_by_name(param_tensor.name)
      input_ops[param_name.name.lower()] = [param]
     
  input_list = []
  for input in node.in_tensors:
    if node.has_bound_params() and input.is_param_tensor():
      continue
    elif input.is_param_tensor():
      input_op = xgraph.get_op_by_name(input.name)
    else:
      input_op = xgraph.get_op_by_name(input.node.name)
    input_list.append(input_op)
    
  input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
  
  xgraph.create_fixed_normal_op(
      node.name, xop_type, quant_config, attrs=attrs, input_ops=input_ops)


def resize(xgraph: XGraph, node: Node,
           quant_config: NndctQuantInfo) -> NoReturn:
  """
  resize is a macro operator, including concat , resize
  """
  attrs: Dict[str, Any] = {}
  attrs["scale"] = node.node_attr(node.op.AttrName.SCALE)
  attrs["align_corners"] = node.node_attr(node.op.AttrName.ALIGN_CORNERS)
  attrs["half_pixel_centers"] = node.node_attr(
      node.op.AttrName.HALF_PIXEL_CENTERS)
  attrs["mode"] = node.node_attr(node.op.AttrName.MODE)
  attrs["mode"] = {0: "NEAREST", 3: "BILINEAR"}.get(attrs["mode"])
  size = node.node_attr(node.op.AttrName.SIZE)
  if size[0] == 0 and size[1] == 0:
    input_ops: Dict[str, List[Op]] = {}
    input_list = []
    for input in node.in_nodes:
      input_op = xgraph.get_op_by_name(input)
      input_list.append(input_op)
    input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
    xgraph.create_fixed_normal_op(
        node.name, "resize", quant_config, attrs=attrs, input_ops=input_ops)
  else:
    sub_pack_op, pack_list = _pack(xgraph, node, "size", size, quant_config)
    input_ops: Dict[str, List[Op]] = {}
    input_ops["size"] = [sub_pack_op]
    input_list = []
    for input in node.in_nodes:
      input_op = xgraph.get_op_by_name(input)
      input_list.append(input_op)
    input_ops["input"] = input_list
    input_ops["input"] = [
        op for op in input_ops["input"]
        if op.get_name() not in [i.get_name() for i in pack_list]
    ]
    input_ops["input"] = xgraph.create_input_fix_ops(input_ops["input"], node.name, quant_config)
    xgraph.create_fixed_normal_op(
        node.name, "resize", quant_config, attrs=attrs, input_ops=input_ops)


def dense(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:

  input_ops: Dict[str, List[Op]] = {}
  for param_name, param_tensor in node.op.params.items():
    if param_name == node.op.ParamName.WEIGHTS:
      weights = xgraph.get_op_by_name(param_tensor.name)
    else:
      bias = xgraph.get_op_by_name(param_tensor.name)
      input_ops["bias"] = [bias]

  input_list = []
  for input in node.in_nodes:
    input_op = xgraph.get_op_by_name(input)
    input_list.append(input_op)
  input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
  input_ops["input"].append(weights)

  attrs: Dict[str, Any] = {}
  attrs["transpose_a"] = False
  attrs["transpose_b"] = True

  xgraph.create_fixed_normal_op(
      node.name, "matmul", quant_config, attrs=attrs, input_ops=input_ops)


def matmul(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:

  input_ops: Dict[str, List[Op]] = {}
  input_list = []
  for input in node.in_nodes:
    input_op = xgraph.get_op_by_name(input)
    input_list.append(input_op)
  input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)

  attrs: Dict[str, Any] = {}
  attrs["transpose_a"] = False
  attrs["transpose_b"] = False

  xgraph.create_fixed_normal_op(
      node.name, "matmul", quant_config, attrs=attrs, input_ops=input_ops)

def permute_invar_op(xop_type, xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  if not node.node_attr(node.op.AttrName.KEEP_DIMS) \
    and node.in_tensors[0].ndim == 4 \
    and len(node.node_attr(node.op.AttrName.DIMS)) == 1 \
    and node.node_attr(node.op.AttrName.DIMS)[0] != 3:
    layout = ["N", "H", "W", "C"]
    del layout[node.node_attr(node.op.AttrName.DIMS)[0]]
    # create mean which keep_dim is True
    attrs: Dict[str, Any] = {}
    attrs["axis"] = node.node_attr(node.op.AttrName.DIMS)
    attrs["keep_dims"] = True
    input_ops: Dict[str, List[Op]] = {}
    input_list = []
    for input in node.in_nodes:
      input_op = xgraph.get_op_by_name(input)
      input_list.append(input_op)
    input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
    xgraph.create_fixed_normal_op(
        node.name + "_i0", xop_type, quant_config, attrs=attrs, input_ops=input_ops)
    
    attrs: Dict[str, Any] = {}
    if layout == ["N", "H", "C"]:
      attrs["order"] = [0, 3, 1, 2]
    else:
      attrs["order"] = [0, 3, 2, 1]

    # resume dimension to NCHW
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]
    xgraph.create_fixed_normal_op(
        node.name + "_i1", "transpose", quant_config, attrs=attrs, input_ops=input_ops)
    
    attrs: Dict[str, Any] = {}
    if layout == ["N", "H", "C"]:
      attrs["axis"] = [3]
    else:
      attrs["axis"] = [2]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i1")]
    xgraph.create_fixed_normal_op(
        node.name, "squeeze", quant_config, attrs=attrs, input_ops=input_ops)
  else:
    to_xir(xop_type)(xgraph, node, quant_config)
    

def flatten(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  
  if node.in_tensors[0].ndim != 4 or node.in_tensors[0].layout == Tensor.Layout.NHWC:
    to_xir("flatten")(xgraph, node, quant_config)
  else:
    attrs: Dict[str, Any] = {}
    # NHWC -> NCHW
    attrs["order"] = [0, 3, 1, 2]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "transpose", quant_config, attrs=attrs, input_ops=input_ops)
    
    # attrs = None
    # if len(node.op.attrs) > 0:
    #   attrs: Dict[str, Any] = {}
    #   for attr_name, attr_value in node.op.attrs.items():
    #     attrs[attr_name.value] = attr_value.value
    attrs = _get_xir_attr_from_node(node)

    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]

    xgraph.create_fixed_normal_op(
        node.name, "flatten", quant_config, attrs=attrs, input_ops=input_ops)
    
    

def avgpool(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  
  needScale = False
  scale = 1.0
  if node.node_attr(node.op.AttrName.KERNEL) == [3, 3]:
    needScale = True
    scale = 9.0 * 7.0 / 64.0
  elif node.node_attr(node.op.AttrName.KERNEL) == [5, 5]:
    needScale = True
    scale = 25.0 * 10.0 / 256.0
  elif node.node_attr(node.op.AttrName.KERNEL) in [[6, 6], [3, 6], [6, 3]]:
    needScale = True
    scale = 36.0 * 7.0 / 256.0
  elif node.node_attr(node.op.AttrName.KERNEL) == [7, 7]:
    needScale = True
    scale = 49.0 * 21.0 / 1024.0
  elif node.node_attr(node.op.AttrName.KERNEL) == [14, 14]:
    needScale = True
    scale = 196.0 * 21.0 / 4096.0
    
  if needScale:
    attrs = _get_xir_attr_from_node(node)
    # attrs: Dict[str, Any] = {}
    # for attr_name, attr_value in node.op.attrs.items():
    #   attrs[attr_name.value] = _Converter.to_xir_attr_value(attr_name.value, attr_value.value)

    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    input_ops["input"] = xgraph.create_input_fix_ops(input_ops["input"], node.name, quant_config)
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "avgpool2d", quant_config, attrs=attrs, input_ops=input_ops)
    
    scale = [scale]
    xgraph.create_fixed_const_op(name=node.name + "_i1", 
                                data=np.array(scale, dtype=np.float32), 
                                quant_info=quant_config)
    
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0"), xgraph.get_op_by_name(node.name + "_i1")]
    xgraph.create_fixed_normal_op(
        node.name, "mul", quant_config, input_ops=input_ops)
  else:
    to_xir("avgpool2d")(xgraph, node, quant_config)


def squeeze(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  if node.in_tensors[0].ndim == 4 and len(node.node_attr(node.op.AttrName.DIMS)) == 1:
    attrs: Dict[str, Any] = {}
    attrs["order"] = [0, 3, 1, 2]

    # resume dimension to NCHW
    input_ops: Dict[str, List[Op]] = {}
    input_list = []
    for input in node.in_nodes:
      input_op = xgraph.get_op_by_name(input)
      input_list.append(input_op)
    input_ops["input"] = input_list
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "transpose", quant_config, attrs=attrs, input_ops=input_ops)
    
    attrs: Dict[str, Any] = {}
    dim = node.node_attr(node.op.AttrName.DIMS)[0]
    dim = transformed_axis("NHWC", "NCHW", ndim=4, dim=dim)
    attrs["axis"] = [dim]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]
    xgraph.create_fixed_normal_op(
        node.name, "squeeze", quant_config, attrs=attrs, input_ops=input_ops)
  else:
    to_xir("squeeze")(xgraph, node, quant_config)
      
def zeros(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  shape = node.node_attr(node.op.AttrName.SHAPE)
  data = np.zeros(shape, dtype=_Converter.to_numpy_dtype(node.out_tensors[0].dtype))
  xgraph.create_fixed_const_op(name=node.name, 
                               data=data, 
                               quant_info=quant_config)
  

def conv3d(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  attrs = _get_xir_attr_from_node(node)
  attrs['kernel'] = attrs['kernel'][::-1]
  attrs['stride'] = attrs['stride'][::-1]
  attrs['dilation'] = attrs['dilation'][::-1]
  attrs['pad'] = list(itertools.chain.from_iterable([[pad]*2 for pad in attrs['pad'][::-1]]))
  print(attrs)
  input_ops: Dict[str, List[Op]] = {}
  if node.has_bound_params():
    for param_name, param_tensor in node.op.params.items():
      param = xgraph.get_op_by_name(param_tensor.name)
      input_ops[param_name.name.lower()] = [param]
     
  input_list = []
  for input in node.in_tensors:
    if node.has_bound_params() and input.is_param_tensor():
      continue
    elif input.is_param_tensor():
      input_op = xgraph.get_op_by_name(input.name)
    else:
      input_op = xgraph.get_op_by_name(input.node.name)
    input_list.append(input_op)
    
  input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
  
  xgraph.create_fixed_normal_op(
      node.name, "conv3d", quant_config, attrs=attrs, input_ops=input_ops)


def conv_transpose_3d(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  attrs = _get_xir_attr_from_node(node)
  attrs['kernel'] = attrs['kernel'][::-1]
  attrs['stride'] = attrs['stride'][::-1]
  attrs['dilation'] = attrs['dilation'][::-1]
  attrs['pad'] = list(itertools.chain.from_iterable([[pad]*2 for pad in attrs['pad'][::-1]]))
  print(attrs)
  input_ops: Dict[str, List[Op]] = {}
  if node.has_bound_params():
    for param_name, param_tensor in node.op.params.items():
      param = xgraph.get_op_by_name(param_tensor.name)
      input_ops[param_name.name.lower()] = [param]
     
  input_list = []
  for input in node.in_tensors:
    if node.has_bound_params() and input.is_param_tensor():
      continue
    elif input.is_param_tensor():
      input_op = xgraph.get_op_by_name(input.name)
    else:
      input_op = xgraph.get_op_by_name(input.node.name)
    input_list.append(input_op)
    
  input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
  
  xgraph.create_fixed_normal_op(
      node.name, "transposed_conv3d", quant_config, attrs=attrs, input_ops=input_ops)


def scale(xgraph, node, quant_config):
  attrs: Dict[str, Any] = {}
  input_ops: Dict[str, List[Op]] = {}
  if node.has_bound_params():
    for param_name, param_tensor in node.op.params.items():
      if param_name == node.op.ParamName.GAMMA:
          input_ops['scale'] = [xgraph.get_op_by_name(param_tensor.name)]
      if param_name == node.op.ParamName.BETA:
          input_ops['bias'] = [xgraph.get_op_by_name(param_tensor.name)]

  input_list = []
  for input in node.in_tensors:
    if node.has_bound_params() and input.is_param_tensor():
      continue
    elif input.is_param_tensor():
      input_op = xgraph.get_op_by_name(input.name)
    else:
      input_op = xgraph.get_op_by_name(input.node.name)
    input_list.append(input_op)

  input_ops["input"] = xgraph.create_input_fix_ops(input_list, node.name, quant_config)
  xgraph.create_fixed_normal_op(node.name, "scale", quant_config, attrs=attrs, input_ops=input_ops)

def hsigmoid(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  scale = 6.0 * 2731.0 / 16384.0
    
  attrs = _get_xir_attr_from_node(node)

  input_ops: Dict[str, List[Op]] = {}
  input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
  input_ops["input"] = xgraph.create_input_fix_ops(input_ops["input"], node.name, quant_config)
  xgraph.create_fixed_normal_op(
      node.name + "_i0", "hard-sigmoid", quant_config, attrs=attrs, input_ops=input_ops)
  
  scale = [scale]
  xgraph.create_fixed_const_op(name=node.name + "_i1", 
                              data=np.array(scale, dtype=np.float32), 
                              quant_info=quant_config)
  
  input_ops: Dict[str, List[Op]] = {}
  input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0"), xgraph.get_op_by_name(node.name + "_i1")]
  xgraph.create_fixed_normal_op(
      node.name, "mul", quant_config, input_ops=input_ops)

def hswish(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  scale = 6.0 * 2731.0 / 16384.0
    
  attrs = _get_xir_attr_from_node(node)
  node_input_op = xgraph.get_op_by_name(node.in_nodes[0])

  input_ops: Dict[str, List[Op]] = {}
  input_ops["input"] = [node_input_op]
  input_ops["input"] = xgraph.create_input_fix_ops(input_ops["input"], node.name, quant_config)
  hsigmoid_op = xgraph.create_fixed_normal_op(
      node.name + "_i0", "hard-sigmoid", quant_config, attrs=attrs, input_ops=input_ops)
  
  scale = [scale]
  const_op = xgraph.create_const_op(name=node.name + "_i1", data=np.array(scale, dtype=np.float32)) 
  
  input_ops["input"] = [hsigmoid_op, const_op]
  mul_op = xgraph.create_normal_op(node.name + '_mul', "mul", input_ops=input_ops)

  mul_fp = [8, None]
  mul_fp[0] = quant_config['output'][node.name][0] 
  mul_fp[1] = mul_fp[0] - 1
  attrs: Dict[str, Any] = {}
  attrs['fix_point'] = mul_fp[1]
  attrs['bit_width'] = mul_fp[0]
  attrs['round_mode'] = "DPU_ROUND"
  attrs['if_signed'] = True
  input_ops['input'] = [mul_op]
  op_name = mul_op.get_name() + NNDCT_KEYS.FIX_OP_SUFFIX
  mul_fixed_op = xgraph.create_normal_op(op_name, 'fix', attrs=attrs, input_ops=input_ops)

  input_ops["input"] = [mul_fixed_op, node_input_op]
  hswish_fixed_op = xgraph.create_fixed_normal_op(
      node.name, "mul", quant_config, input_ops=input_ops)


def unsupported_xop(xgraph: XGraph, node: Node,
                    quant_config: NndctQuantInfo) -> NoReturn:
  raise NotImplementedError(f"Please add op type:{node.op.type} for xmodel.")


def to_xir(xop_type):
  return partial(default_xop, xop_type)


def to_permute_invar_op(xop_type):
  return partial(permute_invar_op, xop_type)


NNDCTIR2XIR_CONVERTOR = {
    NNDCT_OP.INPUT: data_xop,
    NNDCT_OP.CONV2D: to_xir("conv2d"),
    NNDCT_OP.DEPTHWISE_CONV2D: to_xir("depthwise-conv2d"),
    NNDCT_OP.CONVTRANSPOSE2D: to_xir("transposed-conv2d"),
    NNDCT_OP.AVG_POOL: avgpool,
    NNDCT_OP.ADAPTIVEAVGPOOL2D: avgpool,
    NNDCT_OP.MAX_POOL: to_xir("maxpool2d"),
    NNDCT_OP.RELU: to_xir("relu"),
    NNDCT_OP.LEAKY_RELU: to_xir("leaky-relu"),
    NNDCT_OP.TANH: to_xir("tanh"),
    NNDCT_OP.SIGMOID: to_xir("sigmoid"),
    NNDCT_OP.DENSE: dense,
    NNDCT_OP.MATMUL: matmul,
    NNDCT_OP.RESHAPE: reshape,
    NNDCT_OP.ADD: to_xir("add"),
    NNDCT_OP.SCALAR_ADD: to_xir("add"),
    NNDCT_OP.FLATTEN: flatten,
    NNDCT_OP.CONCAT: to_xir("concat"),
    NNDCT_OP.MULTIPLY: to_xir("mul"),
    NNDCT_OP.SCALAR_MUL: to_xir("mul"),
    NNDCT_OP.STRIDED_SLICE: to_xir("strided_slice"),
    NNDCT_OP.RSUB: rsub,
    NNDCT_OP.SUB: sub,
    NNDCT_OP.PAD: to_xir("pad"),
    # NNDCT_OP.RESIZE: to_xir("resize"),
    NNDCT_OP.RESIZE: resize,
    NNDCT_OP.SOFTMAX: to_xir("softmax"),
    NNDCT_OP.PERMUTE: to_xir("transpose"),
    NNDCT_OP.CONST: const_xop,
    NNDCT_OP.RELU6: to_xir("relu6"),
    NNDCT_OP.MEAN: to_permute_invar_op("reduction_mean"),
    NNDCT_OP.BATCH_NORM: scale,
    NNDCT_OP.BATCH_NORM1D: scale,
    NNDCT_OP.BATCH_NORM3D: scale,
    NNDCT_OP.QUANT_STUB: data_xop,
    NNDCT_OP.MAX: to_permute_invar_op("reduction_max"),
    NNDCT_OP.TRANSPOSE: to_xir("transpose"),
    NNDCT_OP.SQUEEZE: squeeze,
    NNDCT_OP.ZEROS: zeros,
    NNDCT_OP.NEG: to_xir("neg"),
    NNDCT_OP.DIV: to_xir("div"),
    NNDCT_OP.SUM: to_permute_invar_op("reduction_sum"),
    #NNDCT_OP.HSIGMOID: to_xir("hard-sigmoid"),
    NNDCT_OP.HSIGMOID: hsigmoid,
    NNDCT_OP.HSWISH: hswish,
    NNDCT_OP.PIXEL_SHUFFLE: to_xir("reorg"),
    # NNDCT_OP.CONV3D: conv3d,
    # NNDCT_OP.CONVTRANSPOSE3D: conv_transpose_3d,
    # NNDCT_OP.RESIZE_3D: to_xir("resize3d")
    # NNDCT_OP.SHAPE: to_xir("shape"),
}
