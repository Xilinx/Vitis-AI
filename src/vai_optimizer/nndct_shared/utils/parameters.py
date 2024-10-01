

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

def get_batchnorm_params(param_list, param_getter, center=True, scale=True):
  #order: gamma,beta,mean,var
  if all(param_getter(p) is not None for p in param_list):
    param_shape = param_getter(param_list[-1]).shape
    bn_params = []
    if center and scale:
      bn_params = [param_getter(p) for p in param_list]
    elif center:
      bn_params = [np.ones(param_shape),
                   param_getter(param_list[0])
                  ] + [param_getter(p) for p in param_list[-2:]]
    elif scale:
      bn_params = [param_getter(node.op.params[0]),
                   np.zeros(param_shape)
                  ] + [param_getter(p) for p in param_list[-2:]]
    if len(bn_params) == 2:
      #no mean and var
      bn_params.extend([np.zeros(param_shape), np.ones(param_shape)])
  else:
    bn_params = [None] * 4
  assert len(
      bn_params
  ) == 4, "batch norm should has 4 variables: gamma, beta, mean, var, please check!"
  return bn_params

def get_batchnorm_param_names(param_list, center=True, scale=True):
  if center and scale:
    assert len(
        param_list) == 4, "expect 4 parameters names, got " + str(param_list)
    return {
        'gamma': param_list[0],
        'beta': param_list[1],
        'mean': param_list[2],
        'var': param_list[3]
    }
  elif center:
    assert len(
        param_list) == 3, "expect 3 parameters names, got " + str(param_list)
    return {'beta': param_list[0], 'mean': param_list[1], 'var': param_list[2]}
  elif scale:
    assert len(
        param_list) == 3, "expect 3 parameters names, got " + str(param_list)
    return {'gamma': param_list[0], 'mean': param_list[1], 'var': param_list[2]}

def get_in_out_channel_idx(ndim, optype, data_formats):
  #TODO: same shape with different format, is this possible?
  if ndim == 1:
    return 0, 0
  if optype == NNDCT_OP.CONV2D:
    if data_formats[optype] == 'HWIO':
      in_idx, out_idx = 2, 3
    elif data_formats[optype] == 'OIHW':
      in_idx, out_idx = 1, 0
    else:
      raise Exception("data format of conv2d kernel {} is not supported".format(
          data_formats[NNDCT_OP.CONV2D]))
  elif optype == NNDCT_OP.DEPTHWISE_CONV2D:
    if data_formats[optype] == 'HWIO':
      in_idx, out_idx = 2, 2
    elif data_formats[optype] == 'OIHW':
      in_idx, out_idx = 1, 1
    else:
      raise Exception(
          "data format of depthwise_conv2d kernel {} is not supported".format(
              data_formats[NNDCT_OP.CONV2D]))
  elif optype in [NNDCT_OP.DENSE, NNDCT_OP.BASIC_LSTM]:
    if data_formats[optype] == 'IO':
      in_idx, out_idx = 0, 1
    elif data_formats[optype] == 'OI':
      in_idx, out_idx = 1, 0
    else:
      raise Exception("data format of 2 dim mat {} is not supported".format(
          data_formats[NNDCT_OP.CONV2D]))
  else:
    raise Exception("unexpected optype: " + str(optype))
  return in_idx, out_idx

def get_tensor_out_dim(tensor, optype, data_formats):
  _, out_idx = get_in_out_channel_idx(tensor.ndim, optype, data_formats)
  return tensor.shape[out_idx]

def get_tensor_in_dim(tensor, optype, data_formats):
  in_idx, _ = get_in_out_channel_idx(tensor.ndim, optype, data_formats)
  return tensor.shape[in_idx]

def delete_in_out_channel_indexs(data,
                                 in_idx=None,
                                 out_idx=None,
                                 in_channel_array=None,
                                 out_channel_array=None):
  if in_idx is not None and in_channel_array is not None and not (
      in_idx == out_idx and out_channel_array is not None):
    data = np.delete(data, in_channel_array, axis=in_idx)
  if out_idx is not None and out_channel_array is not None:
    data = np.delete(data, out_channel_array, axis=out_idx)
  return data

def insert_in_out_channel_indexs(data,
                                 in_idx=None,
                                 out_idx=None,
                                 in_channel_array=None,
                                 out_channel_array=None):
  if in_idx is not None and in_channel_array is not None and not (
      in_idx == out_idx and out_channel_array is not None):
    for pos in sorted(in_channel_array.tolist()):
      data = np.insert(data, pos, 0, axis=in_idx)
  if out_idx is not None and out_channel_array is not None:
    for pos in sorted(out_channel_array.tolist()):
      data = np.insert(data, pos, 0, axis=out_idx)
  return data

def expand_in_out_channel_indexs(data,
                                 in_idx=None,
                                 out_idx=None,
                                 in_channel_array=None,
                                 out_channel_array=None):
  # assert len(data.shape) in [1,2,4], 'unexpected param data shape'
  in_dim = None
  out_dim = None

  if in_channel_array is not None and in_idx is not None and not (
      in_idx == out_idx and out_channel_array is not None):
    in_dim = data.shape[in_idx] + len(in_channel_array)
  if out_idx is not None and out_channel_array is not None:
    out_dim = data.shape[out_idx] + len(out_channel_array)

  assert in_dim is not None or out_dim is not None

  expand_shape = [0] * len(data.shape)
  expand_idxs = [0] * len(data.shape)
  for idx, dim in enumerate(data.shape):
    if in_dim is not None and idx == in_idx:
      expand_shape[idx] = in_dim
      idx_in_channel = sorted(
          np.array(list(set(range(in_dim)) - set(in_channel_array))))
      expand_idxs[idx] = idx_in_channel
    elif out_dim is not None and idx == out_idx:
      expand_shape[idx] = out_dim
      idx_out_channel = sorted(
          np.array(list(set(range(out_dim)) - set(out_channel_array))))
      expand_idxs[idx] = idx_out_channel

    else:
      expand_shape[idx] = dim
      expand_idxs[idx] = np.array(range(dim))

  expand_data = np.zeros(expand_shape, dtype=data.dtype)
  expand_data[np.ix_(*expand_idxs)] = data
  return expand_data
