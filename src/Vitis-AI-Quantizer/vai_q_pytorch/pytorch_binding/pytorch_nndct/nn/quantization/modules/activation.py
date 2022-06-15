import torch

from functools import partial
from torch import nn
from torch.nn import functional as F

from pytorch_nndct.nn.quantization.ops import quantize_ops

class DPULeakyReLU(nn.LeakyReLU):

  def __init__(self, *args, **kwargs):
    # only support the specified slope and inplace operation
    super().__init__(*args, **kwargs)
    self.negative_slope = 0.1015625

  def forward(self, inputs):
    return super().forward(inputs)
