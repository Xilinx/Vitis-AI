

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


class ConvBnHandler(object):
  def __call__(self, *args, **kwargs):
    _, node_set = args
    conv_node, bn_node = node_set
    # print(f"fuse conv {conv_node.name} with bn {bn_node.name}")
    if conv_node.node_attr(conv_node.op.AttrName.BIAS_TERM):
      bias_param_name = conv_node.op.params[
          conv_node.op.ParamName.BIAS].name
      bias_data = conv_node.op.params[conv_node.op.ParamName.BIAS].data
    else:
      # TODO: need to infer split_sym
      split_sym = "."
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

      if conv_node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
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
      elif conv_node.op.type == NNDCT_OP.CONV3D:
        new_conv_weights = conv_weights.transpose(1, 2, 3, 4, 0) * scale
        new_conv_weights = new_conv_weights.transpose(4, 0, 1, 2, 3)
      elif conv_node.op.type == NNDCT_OP.CONVTRANSPOSE3D:
        new_conv_weights = conv_weights.transpose(2, 3, 4, 0, 1) * scale
        new_conv_weights = new_conv_weights.transpose(3, 4, 0, 1, 2)
      else:
        new_conv_weights = conv_weights.transpose(2, 3, 1, 0) * scale
        new_conv_weights = new_conv_weights.transpose(3, 2, 0, 1)

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

