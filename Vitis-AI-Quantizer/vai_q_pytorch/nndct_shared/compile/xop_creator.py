from typing import List, Dict, Any, NoReturn, Tuple
import numpy as np
from functools import partial
from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Tensor, Node
from .xgraph import XGraph
from xir.op import Op

NndctQuantInfo = Dict[str, Dict[str, List[int]]]

  
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
  xgraph.create_fixed_normal_op(name, "sub", quant_config, input_ops=input_ops)


def data_xop(xgraph: XGraph, node: Node,
             quant_config: NndctQuantInfo) -> NoReturn:
  
  shape = node.out_tensors[0].shape
  out_tensor = np.zeros(shape, dtype=np.float32)
  xgraph.create_fixed_normal_op(
      node.name, "data", quant_config, tensor=out_tensor)


def const_xop(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  data = node.node_attr(node.op.AttrName.DATA)
  data_type = np.dtype(node.out_tensors[0].dtype)
  
  if not isinstance(data, list):
    data = [data]
    
  xgraph.create_fixed_const_op(name=node.name, 
                               data=np.array(data, dtype=data_type), 
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
  
  if node.in_tensors[0].ndim != 4 or node.in_tensors[0].layout == Tensor.Layout.NHWC:
    shape = node.node_attr(node.op.AttrName.SHAPE)
    sub_op_pack, pack_list = _pack(xgraph, node, "shape", shape, quant_config)
    input_ops: Dict[str, List[Op]] = {}
    input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name, "reshape", quant_config, input_ops=input_ops)

  else:
    shape = node.node_attr(node.op.AttrName.SHAPE)
    sub_op_pack, pack_list = _pack(xgraph, node, "shape", shape, quant_config)
    attrs: Dict[str, Any] = {}
    # NHWC -> NCHW
    attrs["order"] = [0, 3, 1, 2]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "permute", quant_config, attrs=attrs, input_ops=input_ops)
    
    input_ops: Dict[str, List[Op]] = {}
    input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]
    xgraph.create_fixed_normal_op(
        node.name, "reshape", quant_config, input_ops=input_ops)
    
"""
def reshape(xgraph: XGraph, node: Node,
            quant_config: NndctQuantInfo) -> NoReturn:
 
  if node.in_tensors[0].ndim != 4: 
    to_xir("reshape")(xgraph, node, quant_config)
    
  else:
    attrs: Dict[str, Any] = {}
    # NHWC -> NCHW
    attrs["order"] = [0, 3, 1, 2]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "permute", quant_config, attrs=attrs, input_ops=input_ops)
    
    attrs: Dict[str, Any] = {}
    attrs["shape"] = node.node_attr(node.op.AttrName.SHAPE)
    input_ops: Dict[str, List[Op]] = {}
    # input_ops["shape"] = [sub_op_pack]
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]
    xgraph.create_fixed_normal_op(
        node.name, "reshape", quant_config, attrs=attrs, input_ops=input_ops)
"""   


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
  attrs = None
  if len(node.op.attrs) > 0:
    attrs: Dict[str, Any] = {}
    for attr_name, attr_value in node.op.attrs.items():
      attrs[attr_name.value] = attr_value.value

  input_ops: Dict[str, List[Op]] = {}
  if len(node.op.params) > 0:
    for param_name, param_tensor in node.op.params.items():
      param = xgraph.get_op_by_name(param_tensor.name)
      input_ops[param_name.name.lower()] = [param]
      # if param_name == node.op.ParamName.WEIGHTS:
      #   input_ops["weights"] = [param]
      # else:
      #   input_ops[param_name.value] = [param]

  input_list = []
  for input in node.in_nodes:
    input_op = xgraph.get_op_by_name(input)
    input_list.append(input_op)
  input_ops["input"] = input_list

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
  size = node.node_attr(node.op.AttrName.SIZE)
  if size[0] == 0 and size[1] == 0:
    input_ops: Dict[str, List[Op]] = {}
    input_list = []
    for input in node.in_nodes:
      input_op = xgraph.get_op_by_name(input)
      input_list.append(input_op)
    input_ops["input"] = input_list
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
    xgraph.create_fixed_normal_op(
        node.name, "resize", quant_config, attrs=attrs, input_ops=input_ops)


def dense(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:

  # step2: specify input operators
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
  input_ops["input"] = input_list
  input_ops["input"].append(weights)

  attrs: Dict[str, Any] = {}
  attrs["transpose_a"] = False
  attrs["transpose_b"] = True

  xgraph.create_fixed_normal_op(
      node.name, "matmul", quant_config, attrs=attrs, input_ops=input_ops)


def mean(xgraph: XGraph, node: Node, quant_config: NndctQuantInfo) -> NoReturn:
  if not node.node_attr(node.op.AttrName.KEEP_DIMS) \
    and node.in_tensors[0].ndim == 4 \
    and len(node.node_attr(node.op.AttrName.DIMS)) == 1 \
    and node.node_attr(node.op.AttrName.DIMS)[0] != 3:
    layout = ["N", "H", "W", "C"]
    del layout[node.node_attr(node.op.AttrName.DIMS)[0]]
    # create mean which keep_dim is True
    attrs: Dict[str, Any] = {}
    attrs["dims"] = node.node_attr(node.op.AttrName.DIMS)
    attrs["keep_dims"] = True
    input_ops: Dict[str, List[Op]] = {}
    input_list = []
    for input in node.in_nodes:
      input_op = xgraph.get_op_by_name(input)
      input_list.append(input_op)
    input_ops["input"] = input_list
    xgraph.create_fixed_normal_op(
        node.name + "_i0", "mean", quant_config, attrs=attrs, input_ops=input_ops)
    
    attrs: Dict[str, Any] = {}
    if layout == ["N", "H", "C"]:
      attrs["order"] = [0, 3, 1, 2]
    else:
      attrs["order"] = [0, 3, 2, 1]

    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0")]
    xgraph.create_fixed_normal_op(
        node.name + "_i1", "permute", quant_config, attrs=attrs, input_ops=input_ops)
    
    attrs: Dict[str, Any] = {}
    if layout == ["N", "H", "C"]:
      attrs["dims"] = [3]
    else:
      attrs["dims"] = [2]
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i1")]
    xgraph.create_fixed_normal_op(
        node.name, "squeeze", quant_config, attrs=attrs, input_ops=input_ops)
  else:
    to_xir("mean")(xgraph, node, quant_config)

  
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
        node.name + "_i0", "permute", quant_config, attrs=attrs, input_ops=input_ops)
    
    attrs = None
    if len(node.op.attrs) > 0:
      attrs: Dict[str, Any] = {}
      for attr_name, attr_value in node.op.attrs.items():
        attrs[attr_name.value] = attr_value.value

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
  elif node.node_attr(node.op.AttrName.KERNEL) == [6, 6]:
    needScale = True
    scale = 36.0 * 7.0 / 256.0
  elif node.node_attr(node.op.AttrName.KERNEL) == [7, 7]:
    needScale = True
    scale = 49.0 * 21.0 / 1024.0
  elif node.node_attr(node.op.AttrName.KERNEL) == [14, 14]:
    needScale = True
    scale = 196.0 * 21.0 / 4096.0
    
  if needScale:
    attrs: Dict[str, Any] = {}
    for attr_name, attr_value in node.op.attrs.items():
      attrs[attr_name.value] = attr_value.value

    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.in_nodes[0])]

    xgraph.create_fixed_normal_op(
        node.name + "_i0", "avgpool", quant_config, attrs=attrs, input_ops=input_ops)
    
    scale = [scale]
    xgraph.create_fixed_const_op(name=node.name + "_i1", 
                                data=np.array(scale, dtype=np.float32), 
                                quant_info=quant_config)
    
    input_ops: Dict[str, List[Op]] = {}
    input_ops["input"] = [xgraph.get_op_by_name(node.name + "_i0"), xgraph.get_op_by_name(node.name + "_i1")]
    xgraph.create_fixed_normal_op(
        node.name, "mul", quant_config, input_ops=input_ops)
  else:
    to_xir("avgpool")(xgraph, node, quant_config)

  
def unsupported_xop(xgraph: XGraph, node: Node,
                    quant_config: NndctQuantInfo) -> NoReturn:
  raise NotImplementedError(f"Please add op type:{node.op.type} for xmodel.")


def to_xir(xop_type):
  return partial(default_xop, xop_type)


NNDCTIR2XIR_CONVERTOR = {
    NNDCT_OP.INPUT: data_xop,
    NNDCT_OP.CONV2D: to_xir("conv2d"),
    NNDCT_OP.DEPTHWISE_CONV2D: to_xir("depthwise-conv2d"),
    NNDCT_OP.CONVTRANSPOSE2D: to_xir("transposed-conv2d"),
    NNDCT_OP.AVG_POOL: avgpool,
    NNDCT_OP.ADAPTIVEAVGPOOL2D: avgpool,
    NNDCT_OP.MAX_POOL: to_xir("maxpool"),
    NNDCT_OP.RELU: to_xir("relu"),
    NNDCT_OP.LEAKY_RELU: to_xir("leaky-relu"),
    NNDCT_OP.TANH: to_xir("tanh"),
    NNDCT_OP.SIGMOID: to_xir("sigmoid"),
    NNDCT_OP.DENSE: dense,
    NNDCT_OP.RESHAPE: reshape,
    NNDCT_OP.ADD: to_xir("eltwise"),
    NNDCT_OP.SCALAR_ADD: to_xir("eltwise"),
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
    NNDCT_OP.PERMUTE: to_xir("permute"),
    NNDCT_OP.CONST: const_xop,
    NNDCT_OP.RELU6: to_xir("relu6"),
    NNDCT_OP.MEAN: mean,
    NNDCT_OP.BATCH_NORM: to_xir("batchnorm"),
    NNDCT_OP.BATCH_NORM1D: to_xir("batchnorm")
}
