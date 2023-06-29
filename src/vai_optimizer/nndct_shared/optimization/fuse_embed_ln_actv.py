

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
from nndct_shared.base import NNDCT_OP
from nndct_shared.utils import NndctScreenLogger

BN_MERGED_TYPES = [NNDCT_OP.CONV2D, 
                   NNDCT_OP.DEPTHWISE_CONV2D, 
                   NNDCT_OP.CONVTRANSPOSE2D,
                   NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,
                   NNDCT_OP.CONV3D, 
                   NNDCT_OP.DEPTHWISE_CONV3D,
                   NNDCT_OP.CONVTRANSPOSE3D,
                   NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]

class EmbedLnActvHandler(object):
  def __call__(self, *args, **kwargs):
    _, node_set = args
    embed_node = node_set[0]
    ln_node = node_set[1]
    actv_node = node_set[2]
    
    embed_weight = embed_node.op.params[embed_node.op.ParamName.WEIGHT].data
    embed_weight_shape = embed_weight.shape
    ln_beta = ln_node.op.params[ln_node.op.ParamName.BETA].data
    ln_gamma = ln_node.op.params[ln_node.op.ParamName.GAMMA].data
    ln_eps = ln_node.op.eps
    ln_norm_shape = ln_node.op.normalized_shape
    
    # @staticmethod
    # def _compute_ln_dim(ln_norm_shape, embed_weight_shape):
    #   assert len(ln_norm_shape) <= len(embed_weight_shape)
    #   embed_shape_str = [str(x) for x in embed_weight_shape]
    #   ln_norm_str = [str(x) for x in ln_norm_shape]
    #   embed_str = '#'.join(embed_shape_str) + '#'
    #   ln_str = '#'.join(ln_norm_str) + '#'
    #   index = embed_str.rfind(ln_str)
    #   if index == -1:
    #     NndctScreenLogger().error(f"The normalized shape of LayerNorm is not match the shape of input")
    #     exit(2)
    #   embed_str_cut = embed_str[:index]
    #   start_index = embed_str_cut.count('#')
    #   ln_axis = [start_index]
    #   for i in range(len(ln_norm_shape) - 1):
    #     ln_axis.append(start_index+i+1)
    #   return tuple(ln_axis)
    
    def _compute_ln_dim(ln_norm_shape, input_shape):
      assert len(ln_norm_shape) <= len(input_shape)
      ln_axis = []
      for i in range(len(ln_norm_shape)):
        ln_dim = len(ln_norm_shape) - i - 1
        input_dim = len(input_shape) - i - 1
        if ln_norm_shape[ln_dim] != input_shape[input_dim]:
          NndctScreenLogger().error(f"The normalized shape of LayerNorm is not match the shape of input")
          exit(2)
        ln_axis.append(input_dim)
      return tuple(ln_axis[::-1])
    
    ln_axis = _compute_ln_dim(ln_norm_shape, embed_weight_shape)
    
    embed_mean = embed_weight.mean(axis=(ln_axis))
    embed_var = embed_weight.var(axis=(ln_axis))
    for axis in ln_axis:
      embed_mean = np.expand_dims(embed_mean, axis=axis)
      embed_var = np.expand_dims(embed_var, axis=axis)
    
    #embed_weight_std = (embed_weight-embed_mean)/np.sqrt(embed_var+1e-05)
    #embed_weight_std = ((embed_weight.transpose()-embed_mean)/np.sqrt(embed_var+ln_eps)).transpose()
    embed_weight_std = (embed_weight-embed_mean)/np.sqrt(embed_var+ln_eps)
    embed_ln = ln_gamma*embed_weight_std+ln_beta
    
    embed_ln_actv = None
    if actv_node.op.type == NNDCT_OP.SIGMOID:
      embed_ln_actv = 1./(1.+np.exp(-embed_ln))
    elif actv_node.op.type == NNDCT_OP.TANH:
      embed_ln_actv = np.tanh(embed_ln)
      
    embed_node.op.set_param_from_data(
          embed_node.op.ParamName.WEIGHT,
          embed_ln_actv)
    ln_node.merged = True
    actv_node.merged = True
