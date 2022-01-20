import torch
from torch import Tensor
from typing import Tuple
# Quantizable LSTM layer Module
class QuantLstmLayer(torch.nn.Module):

  def __init__(self,
               input_size: int,
               hidden_size: int,
               memory_size: int,
               lstm_cell: torch.nn.Module,
               go_forward: bool = True) -> None:
  
    
    r"""Applies a layer of long short-term memory(LSTM) to an input sequence. 

      Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        memory_size:The number of features in the hidden state `c`
        lstm_cell: The LSTM cell in the layer
        go_forward: If False, process the input sequence backwards and return the reversed sequence
      
      
      Call Args: 
        input: tensor containing the features of input sequence with shape(batch, timesteps, input_size)
                or The input can also be a packed variable length sequence
        initial_state(optional): a tuple of h_0 and c_0. Both are tensors. The shape of h_0 is  (batch, hidden_size) and the shape
                      of the c_0 is (batch, memory_size).
        batch_length(optional): a Tensor containing the list of lengths of each sequence in the batch
      
      Returns:
        output: tensor containing the features of input sequence with shape(batch, timesteps, hidden_size).
                If input is a packed sequence, then output  also be a packed variable length sequence.
                
        finial_state: a tuple of h_n and c_n.
    """
    
    
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.memory_size = memory_size
    self.lstm_cell_module = lstm_cell
    self.go_forward = go_forward
    self.input = None

  
  def forward(self, input, initial_state=None, batch_lengths=None):
    if self.input is None:
      self.input = input[:]
      self.initial_state = initial_state[:] if initial_state is not None else None
      self.batch_lengths = batch_lengths[:] if batch_lengths is not None else None
      
    batch_size = input.size()[0]
    total_timesteps = input.size()[1]
    output = input.new_zeros(batch_size, total_timesteps,
                                          self.hidden_size)
    
    if initial_state is None:
      full_batch_h = input.new_zeros(batch_size, self.hidden_size)
      full_batch_c = input.new_zeros(batch_size, self.memory_size)
    else:
      full_batch_h = initial_state[0].squeeze(0)
      full_batch_c = initial_state[1].squeeze(0)
      
    full_batch_h = full_batch_h.clone()
    full_batch_c = full_batch_c.clone()  
    if batch_lengths is None:
      current_length_index = batch_size - 1 
    else:
      current_length_index = batch_size - 1 if self.go_forward else 0
      
    for timestep in range(total_timesteps):
      index = timestep if self.go_forward else total_timesteps - timestep - 1
      if batch_lengths is not None:
        if self.go_forward:
          while batch_lengths[current_length_index] <= index:
            current_length_index -= 1
        else:
          while (current_length_index < (len(batch_lengths) - 1) and
                batch_lengths[current_length_index + 1] > index):
            current_length_index += 1
            
      c_prev = full_batch_c[0:current_length_index + 1]
      h_prev = full_batch_h[0:current_length_index + 1]
      #print('---- Forwarding of timestep {}'.format(timestep), flush=True)
      timestep_output, memory = self.lstm_cell_module(input[:current_length_index+1, index], h_prev, c_prev)
      full_batch_c[0:current_length_index + 1] = memory
      full_batch_h[0:current_length_index + 1] = timestep_output
      output[:current_length_index+1, index] = timestep_output

    final_state = (
        full_batch_h.unsqueeze(0),
        full_batch_c.unsqueeze(0),
    )
    return output, final_state
  

class QuantGruLayer(torch.nn.Module):

  def __init__(self,
               input_size: int,
               hidden_size: int,
               gru_cell: torch.nn.Module,
               go_forward: bool = True) -> None:
  
    
    r"""Applies a layer of Gate Recurrent Unit(GRU) to an input sequence. 

      Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        gru_cell: The gru cell in the layer
        go_forward: If False, process the input sequence backwards and return the reversed sequence
      
      
      Call Args: 
        input: tensor containing the features of input sequence with shape(batch, timesteps, input_size)
                or The input can also be a packed variable length sequence
        initial_state(optional): h_0 . Both are tensors. The shape of h_0 is  (batch, hidden_size)
        batch_length(optional): a Tensor containing the list of lengths of each sequence in the batch
      
      Returns:
        output: tensor containing the features of input sequence with shape(batch, timesteps, hidden_size).
                If input is a packed sequence, then output  also be a packed variable length sequence.
                
        finial_state: a tuple of h_n.
    """
    
    
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.lstm_cell_module = gru_cell
    self.go_forward = go_forward
    self.input = None

  
  def forward(self, input, initial_state=None, batch_lengths=None):
    if self.input is None:
      self.input = input[:1]
      self.initial_state = initial_state[0] if initial_state is not None else None
      self.batch_lengths = batch_lengths[:1] if batch_lengths is not None else None
      
    batch_size = input.size()[0]
    total_timesteps = input.size()[1]
    output = input.new_zeros(batch_size, total_timesteps,
                                          self.hidden_size)
    
    if initial_state is None:
      full_batch_h = input.new_zeros(batch_size, self.hidden_size)
    else:
      full_batch_h = initial_state[0].squeeze(0)
      
      
    if batch_lengths is None:
      current_length_index = batch_size - 1 
    else:
      current_length_index = batch_size - 1 if self.go_forward else 0
      
    for timestep in range(total_timesteps):
      index = timestep if self.go_forward else total_timesteps - timestep - 1
      if batch_lengths is not None:
        if self.go_forward:
          while batch_lengths[current_length_index] <= index:
            current_length_index -= 1
        else:
          while (current_length_index < (len(batch_lengths) - 1) and
                batch_lengths[current_length_index + 1] > index):
            current_length_index += 1
            
      h_prev = full_batch_h[0:current_length_index + 1]
      timestep_output = self.lstm_cell_module(input[:current_length_index+1, index], h_prev)
      full_batch_h[0:current_length_index + 1] = timestep_output
      output[:current_length_index+1, index] = timestep_output

    final_state = (
        full_batch_h.unsqueeze(0),
    )
    return output, final_state

  
class LSTMLayer(torch.nn.Module):
    def __init__(self, cell, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        cell_dummy_input = torch.randn(1, input_size)
        cell_dummy_state = (torch.randn(1, hidden_size), torch.randn(1, hidden_size))
        self.cell = torch.jit.trace(cell(input_size, hidden_size, bias), (cell_dummy_input, cell_dummy_state))

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        outputs = []
        for i in range(input.size(0)):
            state = self.cell(input[i], state)
            outputs += [state[0]]
        return torch.stack(outputs), state
      