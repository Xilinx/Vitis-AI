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
      self.bias_ih = torch.nn.Parameter(torch.zeros(num_chunks * hidden_size))
      self.bias_hh = torch.nn.Parameter(torch.zeros(num_chunks * hidden_size))

    # self.igates_linear = torch.nn.Linear(input_size, num_chunks * hidden_size, bias=True)
    # self.hgates_linear = torch.nn.Linear(hidden_size, num_chunks * hidden_size, bias=True)
    
    self.iigates_linear = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.ifgates_linear = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.icgates_linear = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.iogates_linear = torch.nn.Linear(input_size, hidden_size, bias=True)
    
    self.higates_linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    self.hfgates_linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    self.hcgates_linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    self.hogates_linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
  
  def init_weight(self):
    iigates_weight, ifgates_weight, icgates_weight, iogates_weight = self.weight_ih.chunk(4)
    higates_weight, hfgates_weight, hcgates_weight, hogates_weight = self.weight_hh.chunk(4)
    iigates_bias, ifgates_bias, icgates_bias, iogates_bias = self.bias_ih.chunk(4)
    higates_bias, hfgates_bias, hcgates_bias, hogates_bias = self.bias_hh.chunk(4)
    
    self.iigates_linear.weight.data.copy_(iigates_weight.data)
    self.ifgates_linear.weight.data.copy_(ifgates_weight.data)
    self.icgates_linear.weight.data.copy_(icgates_weight.data)
    self.iogates_linear.weight.data.copy_(iogates_weight.data)
    
    self.higates_linear.weight.data.copy_(higates_weight.data)
    self.hfgates_linear.weight.data.copy_(hfgates_weight.data)
    self.hcgates_linear.weight.data.copy_(hcgates_weight.data)
    self.hogates_linear.weight.data.copy_(hogates_weight.data)
    
    self.iigates_linear.bias.data.copy_(iigates_bias.data)
    self.ifgates_linear.bias.data.copy_(ifgates_bias.data)
    self.icgates_linear.bias.data.copy_(icgates_bias.data)
    self.iogates_linear.bias.data.copy_(iogates_bias.data)
    
    self.higates_linear.bias.data.copy_(higates_bias.data)
    self.hfgates_linear.bias.data.copy_(hfgates_bias.data)
    self.hcgates_linear.bias.data.copy_(hcgates_bias.data)
    self.hogates_linear.bias.data.copy_(hogates_bias.data)
    
    # self.igates_linear.weight.data.copy_(self.weight_ih.data)
    # self.hgates_linear.weight.data.copy_(self.weight_hh.data)
    # self.igates_linear.bias.data.copy_(self.bias_ih.data)
    # self.hgates_linear.bias.data.copy_(self.bias_hh.data)



  def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    assert isinstance(input, Tensor)
    assert isinstance(state, tuple)
    
    hx, cx = state
    # igates = self.igates_linear(input)
    # hgates = self.hgates_linear(hx)
    # gates = igates + hgates
    # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    
    iigates = self.iigates_linear(input)
    ifgates = self.ifgates_linear(input)
    icgates = self.icgates_linear(input)
    iogates = self.iogates_linear(input)
    
    higates = self.higates_linear(hx)
    hfgates = self.hfgates_linear(hx)
    hcgates = self.hcgates_linear(hx)
    hogates = self.hogates_linear(hx)
    
    ingate = iigates + higates
    forgetgate = ifgates + hfgates
    cellgate = icgates + hcgates
    outgate = iogates + hogates
    
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * torch.tanh(cy)
    return hy, (hy, cy)
      