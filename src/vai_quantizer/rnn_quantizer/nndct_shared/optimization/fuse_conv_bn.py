

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

BN_MERGED_TYPES = [NNDCT_OP.CONV2D, 
                   NNDCT_OP.DEPTHWISE_CONV2D, 
                   NNDCT_OP.CONVTRANSPOSE2D,
                   NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,
                   NNDCT_OP.CONV3D, 
                   NNDCT_OP.DEPTHWISE_CONV3D,
                   NNDCT_OP.CONVTRANSPOSE3D,
                   NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]


class ConvBnHandler(object):
  def __call__(self, *args, **kwargs):
    _, node_set = args
    conv_node = node_set[0]
    bn_node = node_set[-1]
    concat_node = None
    if len(node_set) == 3:
      concat_node = node_set[1]
      if len(concat_node.out_nodes) > 1:
        return
      for in_node in kwargs['graph'].parents(concat_node):
        if in_node.op.type not in BN_MERGED_TYPES:
          return
    bn_node.merged = True

    # print(f"fuse conv {conv_node.name} with bn {bn_node.name}")
    # print(f"fuse conv {conv_node.name} with bn {bn_node.name}")
    if conv_node.node_attr(conv_node.op.AttrName.BIAS_TERM):
      bias_param_name = conv_node.op.params[
          conv_node.op.ParamName.BIAS].name
      bias_data = conv_node.op.params[conv_node.op.ParamName.BIAS].data
    else:
      # TODO: need to infer split_sym
      split_sym = "."
      # bias_param_name = '.'.join(
      #     conv_node.op.params[conv_node.op.ParamName.WEIGHTS].name.split(
      #         split_sym)[:-1]) + split_sym + 'bias'
      
      end_str = conv_node.op.params[conv_node.op.ParamName.WEIGHTS].name.split(split_sym)[-1]
      if end_str.isdigit():
        bias_param_name = '.'.join(
            conv_node.op.params[conv_node.op.ParamName.WEIGHTS].name.split(
                split_sym)[:-2]) + split_sym + 'bias' + split_sym + end_str
      else:
        bias_param_name = '.'.join(
            conv_node.op.params[conv_node.op.ParamName.WEIGHTS].name.split(
                split_sym)[:-1]) + split_sym + 'bias'

      bias_data = 0
      conv_node.set_node_attr(conv_node.op.AttrName.BIAS_TERM, True)

      conv_node.set_node_attr(conv_node.op.AttrName.BIAS_TERM, True)

    conv_weights = conv_node.op.params[conv_node.op.ParamName.WEIGHTS].data
    bn_gamma = bn_node.op.params[bn_node.op.ParamName.GAMMA].data
    bn_beta = bn_node.op.params[bn_node.op.ParamName.BETA].data
    bn_mean = bn_node.op.params[bn_node.op.ParamName.MOVING_MEAN].data
    bn_var = bn_node.op.params[bn_node.op.ParamName.MOVING_VAR].data
    # epsilon = bn_node.node_attr('epsilon')
    epsilon = bn_node.node_attr(bn_node.op.AttrName.EPSILON)
    if all(data is not None for data in
            [conv_weights, bias_data, bn_beta, bn_gamma, bn_mean, bn_var]):
      scale = bn_gamma / np.sqrt(bn_var + epsilon)
      offset = bn_beta - bn_mean * scale
      if concat_node is not None:
        cat_axis = concat_node.node_attr(concat_node.op.AttrName.AXIS)
        cat_begin = 0
        cat_end = 0
        for idx in range(len(concat_node.in_nodes)):
          if conv_node.name == concat_node.in_nodes[idx]:
            cat_end = cat_begin + concat_node.in_tensors[idx].shape[cat_axis]
            break
          else:
            cat_begin += concat_node.in_tensors[idx].shape[cat_axis]
        scale = scale[cat_begin : cat_end]
        offset = offset[cat_begin : cat_end]

      if conv_node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
        # [channel_multiplier, h, w, in_channels] -> [channel_multiplier, in_channles, h, w]
        conv_weights = conv_weights.transpose(0, 3, 1, 2)
        in_dim = conv_node.node_attr(conv_node.op.AttrName.IN_DIM)
        out_dim = conv_node.node_attr(conv_node.op.AttrName.OUT_DIM)
        kernel_size = conv_node.node_attr(conv_node.op.AttrName.KERNEL)
        channel_multiplier = int(out_dim / in_dim)
        # [channel_multiplier, in_channles, h, w] -> [channel_multiplier * in_channles,  1, h, w]
        conv_weights = conv_weights.reshape((channel_multiplier * in_dim, 1, *kernel_size))
        # [channel_multiplier * in_channles,  1, h, w] -> [h, w, 1, channel_multiplier * in_channles]
        new_conv_weights = conv_weights.transpose(2, 3, 1, 0) * scale
        # [h, w, 1, channel_multiplier * in_channles] -> [channel_multiplier * in_channles, 1, h, w]
        new_conv_weights = new_conv_weights.transpose(3, 2, 0, 1)
        # [channel_multiplier * in_channles, 1, h, w] -> [channel_multiplier, in_channles, h, w]
        new_conv_weights = new_conv_weights.reshape((channel_multiplier, in_dim, *kernel_size))
        # [channel_multiplier, in_channles, h, w] -> [channel_multiplier, h, w, in_channles]
        new_conv_weights = new_conv_weights.transpose(0, 2, 3, 1)
      elif conv_node.op.type in [NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]:
        # [channel_multiplier, h, w, d, in_channels] -> [channel_multiplier, in_channles, h, w, d]
        conv_weights = conv_weights.transpose(0, 4, 1, 2, 3)
        in_dim = conv_node.node_attr(conv_node.op.AttrName.IN_DIM)
        out_dim = conv_node.node_attr(conv_node.op.AttrName.OUT_DIM)
        *kernel_wh, d = conv_node.node_attr(conv_node.op.AttrName.KERNEL)
        channel_multiplier = int(out_dim / in_dim)
        # [channel_multiplier, in_channles, h, w] -> [channel_multiplier * in_channles,  1, h, w, d]
        conv_weights = conv_weights.reshape((channel_multiplier * in_dim, 1, *kernel_wh[::-1], d))
        # [channel_multiplier * in_channles,  1, h, w, d] -> [h, w, d, 1, channel_multiplier * in_channles]
        new_conv_weights = conv_weights.transpose(2, 3, 4, 1, 0) * scale
        # [h, w, d, 1, channel_multiplier * in_channles] -> [channel_multiplier * in_channles, 1, h, w, d]
        new_conv_weights = new_conv_weights.transpose(4, 3, 0, 1, 2)
        # [channel_multiplier * in_channles, 1, h, w, d] -> [channel_multiplier, in_channles, h, w, d]
        new_conv_weights = new_conv_weights.reshape((channel_multiplier, in_dim, *kernel_wh[::-1], d))
        # [channel_multiplier, in_channles, h, w, d] -> [channel_multiplier, h, w, d, in_channles]
        new_conv_weights = new_conv_weights.transpose(0, 2, 3, 4, 1)
      else:
        new_conv_weights = conv_weights.swapaxes(0, conv_weights.ndim - 1) * scale
        new_conv_weights = new_conv_weights.swapaxes(0, conv_weights.ndim - 1)

      conv_node.op.set_param_from_data(
          conv_node.op.ParamName.WEIGHTS,
          new_conv_weights)
      conv_node.op.set_param_from_data(conv_node.op.ParamName.BIAS,
                                       bias_data * scale + offset,
                                       bias_param_name)

    else:
      kernel_tensor = conv_node.op.params[conv_node.op.ParamName.WEIGHTS]
      if conv_node.op.type == NNDCT_OP.CONV2D:
        shape = [kernel_tensor.shape[0]]
      elif conv_node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
        shape = [kernel_tensor.shape[1]]
      else:
        raise KeyError(
            "unexpected conv type {} found during FuseBnToConv".foramt(
                conv_node.op.type))
      conv_node.op.set_param_from_des(conv_node.op.ParamName.BIAS, {
          'shape': shape,
          'dtype': kernel_tensor.dtype
      }, bias_param_name)

