

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

import math


class Evaluator(object):
  @staticmethod
  def shape(node):
    shape = node.in_tensors[0].shape
    dim = node.node_attr(node.op.AttrName.AXIS)
    node.out_tensors[0].data = shape[dim]

  @staticmethod
  def tensor(node):
    # assert node.in_tensors[0].ndim == 0
    # node.out_tensors[0].data = float(node.in_tensors[0].data)
    node.out_tensors[0].data = float(node.get_attr_val(node.op.AttrName.DATA))
    
  @staticmethod
  def mul(node): 
    node.out_tensors[0].data = node.get_attr_val(node.op.AttrName.INPUT) * node.get_attr_val(node.op.AttrName.OTHER) 
    
  @staticmethod
  def cast(node):
    # node.out_tensors[0].data = node.in_tensors[0].data
    node.out_tensors[0].data = node.get_attr_val(node.op.AttrName.INPUT)
  
  @staticmethod
  def floor(node):
    # node.out_tensors[0].data = math.floor(node.in_tensors[0].data)
    node.out_tensors[0].data = math.floor(node.get_attr_val(node.op.AttrName.INPUT))
    
  @staticmethod
  def int(node):
    # node.out_tensors[0].data = int(node.in_tensors[0].data)
    node.out_tensors[0].data = int(node.get_attr_val(node.op.AttrName.INPUT))
    
  @staticmethod
  def sub(node):
    node.out_tensors[0].data = node.get_attr_val(node.op.AttrName.INPUT) - node.get_attr_val(node.op.AttrName.OTHER) 

  @staticmethod
  def elemwise_div(node):
    node.out_tensors[0].data = float(node.get_attr_val(node.op.AttrName.INPUT) / node.get_attr_val(node.op.AttrName.OTHER))
    
  @staticmethod
  def floor_div(node):
    node.out_tensors[0].data = int(node.get_attr_val(node.op.AttrName.INPUT) // node.get_attr_val(node.op.AttrName.OTHER))

  @staticmethod
  def add(node):
    node.out_tensors[0].data = node.get_attr_val(node.op.AttrName.INPUT) + node.get_attr_val(node.op.AttrName.OTHER) 
    