import torch

class DPULeakyReLU(torch.nn.LeakyReLU):
  def __init__(self, *args, **kwargs):
    # only support the specified slope and inplace operation
    super().__init__(*args, **kwargs)
    self.negative_slope = 0.1015625

  def forward(self, inputs):
    return super().forward(inputs)
