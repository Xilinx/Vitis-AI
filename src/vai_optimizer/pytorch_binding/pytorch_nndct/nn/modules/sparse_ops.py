import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['SparseConv2d', 'SparseLinear']

def reshape_tensor(matrix, m):
  if matrix.shape[1] % m > 0:
    # padding 0
    mat = torch.cuda.FloatTensor(matrix.shape[0], matrix.shape[1] +
                                 (m - matrix.shape[1] % m)).fill_(0)
    mat[:, :matrix.shape[1]] = matrix
    return mat.view(-1, m)
  else:
    return matrix.view(-1, m)

def get_mask(matrix, m, n):
  mat = reshape_tensor(matrix, m)

  topk, indices = mat.abs().topk(k=n, dim=1, sorted=False)
  mask_matrix = torch.zeros_like(mat)
  mask_min = torch.min(topk, dim=-1).values
  mask_min = mask_min.unsqueeze(-1).repeat(1, m)
  mask = torch.where(
      torch.ge(mat.abs(), mask_min) & torch.ne(mat.abs(), mask_matrix), mat,
      mask_matrix)
  return mask

""" returns a sparse mask """

def compute_sparse_tensor(tensor, sparsity, block_size):
  if sparsity != 0:

    shape = tensor.shape
    t = tensor.float().contiguous()

    if len(tensor.shape) == 2:
      # fc
      t = t.view(shape[0], shape[1])
      mask = get_mask(t, block_size, int(block_size * (1 - sparsity)))
    elif len(tensor.shape) == 4:
      # conv
      t = t.permute(2, 3, 0,
                    1).contiguous().view(shape[2] * shape[3] * shape[0],
                                         shape[1])
      mask = get_mask(t, block_size, int(block_size * (1 - sparsity)))
      mask = mask.view(shape[2], shape[3], shape[0],
                       shape[1]).permute(2, 3, 0, 1).contiguous()

    return mask.view(shape).type(tensor.type())

  else:
    return tensor

class SparseConv2d(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               **kwargs):

    super(SparseConv2d, self).__init__()

    self.w_sparsity = kwargs['w_sparsity']
    self.a_sparsity = kwargs['a_sparsity']
    self.block_size = kwargs['block_size']
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.bias = bias

    self.conv = nn.Conv2d(
        in_channels,
        out_channels,
        self.kernel_size,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
        self.bias,
    )

  def forward(self, input):
    w_shape = self.conv.weight.shape

    if input.shape[1] % self.block_size != 0:
      in_channels_to_pad = self.block_size - input.shape[1] % self.block_size

      padded_input = torch.cuda.FloatTensor(input.shape[0],
                                            input.shape[1] + in_channels_to_pad,
                                            input.shape[2],
                                            input.shape[3]).fill_(0)
      padded_input[:, 0:input.shape[1], :, :] = input

      padded_weights = torch.cuda.FloatTensor(w_shape[0],
                                              w_shape[1] + in_channels_to_pad,
                                              w_shape[2], w_shape[3]).fill_(0)
      padded_weights[:, 0:w_shape[1], :, :] = self.conv.weight
    else:
      padded_input = input
      padded_weights = self.conv.weight

    sparse_input = compute_sparse_tensor(padded_input, self.a_sparsity,
                                         self.block_size)
    sparse_weight = compute_sparse_tensor(padded_weights, self.w_sparsity,
                                          self.block_size)

    out = F.conv2d(sparse_input, sparse_weight, self.conv.bias, self.stride,
                   self.padding, self.dilation, self.groups)
    return out

class SparseLinear(nn.Module):

  def __init__(self, in_features, out_features, bias=True, **kwargs):

    super(SparseLinear, self).__init__()

    self.w_sparsity = kwargs['w_sparsity']
    self.a_sparsity = kwargs['a_sparsity']
    self.block_size = kwargs['block_size']

    self.linear = nn.Linear(in_features, out_features, bias)

  def forward(self, input):

    sparse_input = compute_sparse_tensor(input, self.a_sparsity,
                                         self.block_size)
    sparse_weight = compute_sparse_tensor(self.linear.weight, self.w_sparsity,
                                          self.block_size)

    return F.linear(sparse_input, sparse_weight, self.linear.bias)
