

from nndct_shared.base import NNDCT_CONSTANT

def shape_attr_transform_fn(node, transpose_order):
  shape = node.node_attr(node.op.AttrName.SHAPE)
  new_shape = len(shape) * [None]
  for i, dim in enumerate(transpose_order):
    new_shape[i] = shape[dim] 
  node.set_node_attr(node.op.AttrName.SHAPE, new_shape)   
  


def axis_attr_transform_fn(node, transpose_order):
  dim = node.node_attr(node.op.AttrName.AXIS)  
  new_dim = transpose_order.index(dim)
  node.set_node_attr(node.op.AttrName.AXIS, new_dim)
  
def slice_attr_transform_fn(node, transpose_order):
  begin = node.node_attr(node.op.AttrName.BEGIN)
  new_begin = [None] * len(begin)
  for dim, pos in enumerate(begin):
    new_dim = transpose_order.index(dim)
    new_begin[new_dim] = pos
                  
  begin_mask = 0
  for dim, pos in enumerate(new_begin):
    if pos == 0:
      begin_mask |= 1 << dim   
                    
  node.set_node_attr(node.op.AttrName.BEGIN_MASK, begin_mask)
  node.set_node_attr(node.op.AttrName.BEGIN, new_begin)
                  
  end = node.node_attr(node.op.AttrName.END)
  new_end = [None] * len(end)
  end_mask = 0
  for dim, pos in enumerate(end):
    new_dim = transpose_order.index(dim)
    new_end[new_dim] = pos

  for dim, pos in enumerate(new_end):
    if isinstance(pos, int) and pos >= NNDCT_CONSTANT.INT_MAX:
      end_mask |= 1 << dim

  node.set_node_attr(node.op.AttrName.END_MASK, end_mask)
  node.set_node_attr(node.op.AttrName.END, new_end)
  
  strides = node.node_attr(node.op.AttrName.STRIDES)
  new_strides = [1] * len(strides)
  
  for dim, step in enumerate(strides):
    new_dim = transpose_order.index(dim)
    new_strides[new_dim] = step
  
  node.set_node_attr(node.op.AttrName.STRIDES, new_strides)  
  
  
def reduce_op_attr_transform_fn(node, transpose_order):
  dims = node.node_attr(node.op.AttrName.DIMS)
  new_dims = [None] * len(dims)
  for i, dim in enumerate(dims):
    new_dim = transpose_order.index(dim)
    new_dims[i] = new_dim
    
  node.set_node_attr(node.op.AttrName.DIMS, new_dims)

