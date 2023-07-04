

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

import numpy as np


class PermuteMergeHandler(object):
  def __call__(self, *args, **kwargs):
    _, node_set = args
    permute_node = node_set[0]
    reshape_like_node = node_set[1]
    linear_node = node_set[2]
    if len(permute_node.out_nodes) > 1 or len(reshape_like_node.out_nodes) > 1:
      permute_node.merged = False
      return
    
    order = permute_node.node_attr(permute_node.op.AttrName.ORDER)
    if order != [0, 3, 1, 2]:
      permute_node.merged = False
      return 
    
    batch = permute_node.out_tensors[0].shape[0]
    reshape_shape = reshape_like_node.out_tensors[0].shape

    if reshape_shape[len(reshape_shape) - 2] != batch:
      permute_node.merged = False
      return 

    for dim_i in range(len(reshape_shape) - 2):
      if reshape_shape[dim_i] > 1:
        permute_node.merged = False
        return 
    
    weight_shape = linear_node.op.params[linear_node.op.ParamName.WEIGHTS].shape
    weight_data = linear_node.op.params[linear_node.op.ParamName.WEIGHTS].data.flatten()
    new_weights_array = []
    ori_shape = permute_node.out_tensors[0].shape
    for b in range(weight_shape[0]):
      for h in range(ori_shape[2]):
        for w in range(ori_shape[3]):
          for c in range(ori_shape[1]):
            idx = ((b * ori_shape[1] + c) * ori_shape[2] + h) * ori_shape[3] + w
            new_weights_array.append(weight_data[idx])

    new_weight_data = np.array(new_weights_array).reshape(weight_shape)
    linear_node.op.params[linear_node.op.ParamName.WEIGHTS].from_ndarray(new_weight_data)
    permute_node.merged = True


    


