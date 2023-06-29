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

from .op_dispatcher import *
from .parse_utils import *

def change_addmm_to_linear(raw_graph):
  for node in raw_graph.nodes:
    if node.op.type in [NNDCT_OP.ADDMM]:
      weight = node.op.get_config('mat2')
      bias = node.op.get_config('input')
      if (weight and weight.node == None) and (bias and bias.node == None):
        linear_op = TorchLinear()
        weight_size = weight.shape
        linear_op.set_param(linear_op.ParamName.WEIGHTS, weight)
        if bias is None:
          linear_op.set_config("bias", False)
        else:
          linear_op.set_config("bias", True)
          linear_op.set_param(linear_op.ParamName.BIAS, bias)

        linear_op.set_config('out_features', weight_size[1])
        linear_op.set_config('in_features', weight_size[0])
        
        addmm_weights = linear_op.params[linear_op.ParamName.WEIGHTS].data
        addmm_weights = addmm_weights.transpose(1,0)
        linear_op.set_param_from_data(
          linear_op.ParamName.WEIGHTS,
          addmm_weights)
        node.op = linear_op