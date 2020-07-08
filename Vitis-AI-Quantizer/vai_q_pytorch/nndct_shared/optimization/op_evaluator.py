
import math


class Evaluator(object):
  @staticmethod
  def shape(node):
    shape = node.in_tensors[0].shape
    dim = node.node_attr(node.op.AttrName.AXIS)
    node.out_tensors[0].data = shape[dim]

  @staticmethod
  def tensor(node):
    assert node.in_tensors[0].ndim == 0
    node.out_tensors[0].data = float(node.in_tensors[0].data)
    
  @staticmethod
  def mul(node): 
    node.out_tensors[0].data = node.in_tensors[0].data * node.in_tensors[1].data
    
  @staticmethod
  def cast(node):
    node.out_tensors[0].data = node.in_tensors[0].data
  
  @staticmethod
  def floor(node):
    node.out_tensors[0].data = math.floor(node.in_tensors[0].data)
    
  @staticmethod
  def int(node):
    node.out_tensors[0].data = int(node.in_tensors[0].data)
    
  