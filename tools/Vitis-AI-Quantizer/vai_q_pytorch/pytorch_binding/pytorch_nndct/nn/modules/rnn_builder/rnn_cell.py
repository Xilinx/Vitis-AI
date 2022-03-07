from typing import Optional, Tuple

import torch
from torch import Tensor


class LSTMCell(torch.nn.Module):
  __constants__ = ['input_size', 'hidden_size', 'bias']
  
  def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    num_chunks = 4
    self.weight_ih = torch.nn.Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
    self.weight_hh = torch.nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
    if bias:
      self.bias_ih = torch.nn.Parameter(torch.Tensor(num_chunks * hidden_size))
      self.bias_hh = torch.nn.Parameter(torch.Tensor(num_chunks * hidden_size))
    else:
      self.register_parameter('bias_ih', None)
      self.register_parameter('bias_hh', None)

  def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    assert isinstance(input, Tensor)
    assert isinstance(state, tuple)
    
    hx, cx = state
    igates = torch.mm(input, self.weight_ih.t())
    hgates = torch.mm(hx, self.weight_hh.t())
    gates = igates + hgates
    if self.bias:
      gates += self.bias_ih + self.bias_hh
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * torch.tanh(cy)
    return hy, cy
      