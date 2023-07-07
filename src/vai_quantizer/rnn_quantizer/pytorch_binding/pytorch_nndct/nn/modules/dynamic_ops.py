# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

__all__ = [
    'DynamicSeparableConv2d', 'DynamicConv2d', 'DynamicGroupConv2d',
    'DynamicBatchNorm2d', 'DynamicGroupNorm', 'DynamicLinear'
]

def get_same_padding(kernel_size):
  if isinstance(kernel_size, tuple):
    assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
    p1 = get_same_padding(kernel_size[0])
    p2 = get_same_padding(kernel_size[1])
    return p1, p2
  assert isinstance(kernel_size,
                    int), 'kernel size should be either `int` or `tuple`'
  assert kernel_size % 2 > 0, 'kernel size should be odd number'
  return kernel_size // 2

def sub_filter_start_end(kernel_size, sub_kernel_size):
  center = kernel_size // 2
  dev = sub_kernel_size // 2
  start, end = center - dev, center + dev + 1
  assert end - start == sub_kernel_size
  return start, end

class DynamicSeparableConv2d(nn.Module):

  def __init__(self,
               candidate_in_channels_list,
               candidate_out_channels_list,
               candidate_kernel_size_list,
               stride=1,
               dilation=1,
               groups=1,
               bias=False):
    super(DynamicSeparableConv2d, self).__init__()

    self.candidate_in_channels_list = candidate_in_channels_list
    self.candidate_out_channels_list = candidate_out_channels_list
    self.candidate_kernel_size_list = candidate_kernel_size_list
    self.stride = stride
    self.dilation = dilation
    self.bias = bias

    self.conv = nn.Conv2d(
        max(self.candidate_in_channels_list),
        max(self.candidate_in_channels_list),
        max(self.candidate_kernel_size_list),
        self.stride,
        dilation=self.dilation,
        groups=max(self.candidate_in_channels_list),
        bias=self.bias,
    )

    self._ks_set = list(set(self.candidate_kernel_size_list))
    self._ks_set.sort()  # e.g., [3, 5, 7]
    self.active_kernel_size = max(self.candidate_kernel_size_list)

  def get_active_filter(self, in_channel, kernel_size):
    out_channel = in_channel
    max_kernel_size = max(self.candidate_kernel_size_list)

    start, end = sub_filter_start_end(max_kernel_size, kernel_size)
    filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
    return filters

  def get_active_bias(self, out_channel):
    return self.conv.bias[:out_channel] if self.bias else None

  def forward(self, x, kernel_size=None):
    if kernel_size is None:
      kernel_size = self.active_kernel_size

    in_channel = x.size(1)

    filters = self.get_active_filter(in_channel, kernel_size).contiguous()
    bias = self.get_active_bias(in_channel)

    padding = get_same_padding(kernel_size)

    y = F.conv2d(x, filters, bias, self.stride, padding, self.dilation,
                 in_channel)
    return y

class DynamicConv2d(nn.Module):

  def __init__(self,
               candidate_in_channels_list,
               candidate_out_channels_list,
               candidate_kernel_size_list,
               stride=1,
               dilation=1,
               groups=1,
               bias=False):
    super(DynamicConv2d, self).__init__()

    self.candidate_in_channels_list = candidate_in_channels_list
    self.candidate_out_channels_list = candidate_out_channels_list
    self.candidate_kernel_size_list = candidate_kernel_size_list
    self.stride = stride
    self.padding = 1  #dy
    self.dilation = dilation
    self.groups = groups  #dy
    self.bias = bias

    self.conv = nn.Conv2d(
        max(self.candidate_in_channels_list),
        max(self.candidate_out_channels_list),
        max(self.candidate_kernel_size_list),
        self.stride,
        dilation=self.dilation,
        groups=self.groups,
        bias=self.bias,
    )

    self.is_depthwsie_conv = True if self.groups == max(
        self.candidate_in_channels_list) else False

    self._ks_set = list(set(self.candidate_kernel_size_list))
    self._ks_set.sort()  # e.g., [3, 5, 7]

    self.active_in_channel = max(
        self.candidate_out_channels_list)  # for export subet
    self.active_out_channel = max(self.candidate_out_channels_list)
    self.active_kernel_size = max(self.candidate_kernel_size_list)

  def get_active_filter(self, in_channel, out_channel, kernel_size):
    max_kernel_size = max(self.candidate_kernel_size_list)

    start, end = sub_filter_start_end(max_kernel_size, kernel_size)
    filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]

    return filters

  def get_active_bias(self, out_channel):
    return self.conv.bias[:out_channel] if self.bias else None

  def forward(self, x, out_channel=None, kernel_size=None):
    if kernel_size is None:
      kernel_size = self.active_kernel_size
    if out_channel is None:
      out_channel = self.active_out_channel

    in_channel = x.size(1)
    self.active_in_channel = in_channel

    if self.is_depthwsie_conv == True:
      self.groups = in_channel
      out_channel = in_channel

    filters = self.get_active_filter(in_channel, out_channel,
                                     kernel_size).contiguous()
    bias = self.get_active_bias(out_channel)

    self.padding = get_same_padding(kernel_size)

    y = F.conv2d(x, filters, bias, self.stride, self.padding, self.dilation,
                 self.groups)

    return y

class DynamicPointConv2d(nn.Module):

  def __init__(self,
               candidate_in_channels_list,
               candidate_out_channels_list,
               candidate_kernel_size_list,
               stride=1,
               dilation=1,
               groups=1,
               bias=False):
    super(DynamicPointConv2d, self).__init__()

    self.candidate_in_channels_list = candidate_in_channels_list
    self.candidate_out_channels_list = candidate_out_channels_list
    self.candidate_kernel_size_list = candidate_kernel_size_list
    self.stride = stride
    self.dilation = dilation
    self.groups = groups
    self.bias = bias

    self.conv = nn.Conv2d(
        max(self.candidate_in_channels_list),
        max(self.candidate_out_channels_list),
        max(self.candidate_kernel_size_list),
        stride=self.stride,
        dilation=self.dilation,
        groups=self.groups,
        bias=self.bias,
    )

    self.active_out_channel = max(self.candidate_out_channels_list)

  def get_active_filter(self, out_channel, in_channel):
    return self.conv.weight[:out_channel, :in_channel, :, :]

  def get_active_bias(self, out_channel):
    return self.conv.bias[:out_channel] if self.bias else None

  def forward(self, x, out_channel=None):
    if out_channel is None:
      out_channel = self.active_out_channel
    in_channel = x.size(1)
    filters = self.get_active_filter(out_channel, in_channel).contiguous()
    bias = self.get_active_bias(out_channel)

    padding = get_same_padding(self.candidate_kernel_size_list)

    y = F.conv2d(x, filters, bias, self.stride, padding, self.dilation, 1)
    return y

class DynamicGroupConv2d(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size_list,
               stride=1,
               dilation=1,
               groups_list=[1],
               bias=False):
    super(DynamicGroupConv2d, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size_list = kernel_size_list
    self.groups_list = groups_list
    self.stride = stride
    self.dilation = dilation

    self.conv = nn.Conv2d(
        self.in_channels,
        self.out_channels,
        max(self.kernel_size_list),
        self.stride,
        groups=min(self.groups_list),
        bias=False,
    )

    self.active_kernel_size = max(self.kernel_size_list)
    self.active_groups = min(self.groups_list)

  def get_active_filter(self, kernel_size, groups):
    start, end = sub_filter_start_end(max(self.kernel_size_list), kernel_size)
    filters = self.conv.weight[:, :, start:end, start:end]

    sub_filters = torch.chunk(filters, groups, dim=0)
    sub_in_channels = self.in_channels // groups
    sub_ratio = filters.size(1) // sub_in_channels

    filter_crops = []
    for i, sub_filter in enumerate(sub_filters):
      part_id = i % sub_ratio
      start = part_id * sub_in_channels
      filter_crops.append(sub_filter[:, start:start + sub_in_channels, :, :])
    filters = torch.cat(filter_crops, dim=0)
    return filters

  def get_active_bias(self, out_channel):
    return self.conv.bias[:out_channel] if self.bias else None

  def forward(self, x, kernel_size=None, groups=None):
    if kernel_size is None:
      kernel_size = self.active_kernel_size
    if groups is None:
      groups = self.active_groups

    filters = self.get_active_filter(kernel_size, groups).contiguous()
    bias = self.get_active_bias(self.out_channels)
    padding = get_same_padding(kernel_size)

    y = F.conv2d(
        x,
        filters,
        bias,
        self.stride,
        padding,
        self.dilation,
        groups,
    )
    return y

class DynamicBatchNorm2d(nn.Module):
  SET_RUNNING_STATISTICS = False

  def __init__(self,
               num_features,
               eps=1e-5,
               momentum=0.1,
               affine=True,
               track_running_stats=True):
    super(DynamicBatchNorm2d, self).__init__()

    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats

    self.bn = nn.BatchNorm2d(self.num_features, self.eps, self.momentum,
                             self.affine, self.track_running_stats)

  @staticmethod
  def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
    if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
      return bn(x)
    else:
      exponential_average_factor = 0.0

      if bn.training and bn.track_running_stats:
        if bn.num_batches_tracked is not None:
          bn.num_batches_tracked += 1
          if bn.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
          else:  # use exponential moving average
            exponential_average_factor = bn.momentum
      return F.batch_norm(
          x,
          bn.running_mean[:feature_dim],
          bn.running_var[:feature_dim],
          bn.weight[:feature_dim],
          bn.bias[:feature_dim],
          bn.training or not bn.track_running_stats,
          exponential_average_factor,
          bn.eps,
      )

  def forward(self, x):
    feature_dim = x.size(1)
    self.num_features = feature_dim
    y = self.bn_forward(x, self.bn, feature_dim)
    return y

class DynamicGroupNorm(nn.GroupNorm):

  def __init__(self,
               num_groups,
               num_channels,
               eps=1e-5,
               affine=True,
               channel_per_group=None):
    super(DynamicGroupNorm, self).__init__(num_groups, num_channels, eps,
                                           affine)
    self.channel_per_group = channel_per_group

  def forward(self, x):
    n_channels = x.size(1)
    n_groups = n_channels // self.channel_per_group
    return F.group_norm(x, n_groups, self.weight[:n_channels],
                        self.bias[:n_channels], self.eps)

  @property
  def bn(self):
    return self

class DynamicLinear(nn.Module):

  def __init__(self, max_in_features, max_out_features, bias=True):
    super(DynamicLinear, self).__init__()

    self.max_in_features = max_in_features
    self.max_out_features = max_out_features
    self.bias = bias

    self.linear = nn.Linear(self.max_in_features, self.max_out_features,
                            self.bias)

    self.active_out_features = self.max_out_features

  def get_active_weight(self, out_features, in_features):
    return self.linear.weight[:out_features, :in_features]

  def get_active_bias(self, out_features):
    return self.linear.bias[:out_features] if self.bias else None

  def forward(self, x, out_features=None):
    if out_features is None:
      out_features = self.active_out_features

    in_features = x.size(1)
    weight = self.get_active_weight(out_features, in_features).contiguous()
    bias = self.get_active_bias(out_features)
    y = F.linear(x, weight, bias)
    return y
