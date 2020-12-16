

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

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from .rnn_layer import QuantLstmLayer, QuantGruLayer
class StackedLstm(torch.nn.Module):

  def __init__(self, input_sizes, hidden_sizes, memory_sizes, layers, stack_mode=None, batch_first=None):
    super(StackedLstm, self).__init__()
    self.stack_mode = stack_mode
    self.num_layers = len(layers)
    self.lstm_layers = []
    self.batch_first = batch_first
    for layer_index in range(len(layers)):
      lstm_cell_pair = {}
      for direction, layer in layers[layer_index].items():
        layer = Lstm(input_sizes[layer_index], hidden_sizes[layer_index], memory_sizes[layer_index], layer, direction,
                      self.batch_first)
        self.add_module(f"{direction}_layer_{layer_index}", layer)
        lstm_cell_pair[direction] = layer
      self.lstm_layers.append(lstm_cell_pair)

  def forward(self, inputs, initial_state=None, **kwargs):
    is_packed = isinstance(inputs, PackedSequence)
    if self.stack_mode in ['bidirectional']:
      if not initial_state:
        hidden_states = [None] * len(self.lstm_layers) * len(
            self.lstm_layers[0])
      elif initial_state[0].size()[0] != len(self.lstm_layers) * len(
          self.lstm_layers[0]):
        raise ValueError(f"initial states does not match the number of layers.")
      else:
        hidden_states = list(
            zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))
    else:
      if not initial_state:
        hidden_states = [None] * len(self.lstm_layers)
      elif initial_state[0].size()[0] != len(self.lstm_layers):
        raise ValueError(f"initial states does not match the number of layers.")
      else:
        hidden_states = list(
            zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))
    # print(f"nndct_inputs:{inputs}")
    output_sequence = inputs

    if self.stack_mode in ['bidirectional']:
      final_h = []
      final_c = []

      for i in range(len(self.lstm_layers)):
        forward_layer = getattr(self, "forward_layer_{}".format(i))
        backward_layer = getattr(self, "backward_layer_{}".format(i))
        forward_output, final_forward_state = forward_layer(
            output_sequence, hidden_states[i * 2])
        if self.batch_first is not True:
          output_sequence.transpose_(0, 1)
        backward_output, final_backward_state = backward_layer(
            output_sequence, hidden_states[i * 2 + 1])
        if is_packed:
          forward_output, lengths = pad_packed_sequence(
              forward_output, batch_first=self.batch_first)
          backward_output, _ = pad_packed_sequence(
              backward_output, batch_first=self.batch_first)

        # output_sequence = output_sequence.flip(1)
        # backward_output = backward_output.flip(1)
        output_sequence = torch.cat([forward_output, backward_output], -1)
        if is_packed:
          output_sequence = pack_padded_sequence(
              output_sequence, lengths, batch_first=self.batch_first)
        final_h.extend([final_forward_state[0], final_backward_state[0]])
        final_c.extend([final_forward_state[1], final_backward_state[1]])
      final_hidden_state = torch.cat(final_h, dim=0)
      final_cell_state = torch.cat(final_c, dim=0)
    else:
      final_states = []
      for i, state in enumerate(hidden_states):
        if self.stack_mode == 'alternating':
          layer = getattr(self,
                          f"forward_layer_{i}") if i % 2 == 0 else getattr(
                              self, f"backward_layer_{i}")
        else:
          layer = getattr(self, f"forward_layer_{i}")

        output_sequence, final_state = layer(output_sequence, state)

        # print(f"nndct_layer{i} output:{output_sequence}")
        final_states.append(final_state)

      final_hidden_state, final_cell_state = tuple(
          torch.cat(state_list, 0) for state_list in zip(*final_states))
    # print(f"nndct_final_output:{output_sequence}")
    return output_sequence, (final_hidden_state, final_cell_state)


class Lstm(torch.nn.Module):
 

  def __init__(self, input_size, hidden_size, memory_size, cell_module, direction, batch_first):
    super(Lstm, self).__init__()
    # self.cell_module = cell_module
    # self.hidden_size = hidden_size
  
    self.batch_first = batch_first
    go_forward = True if direction == "forward" else False
    self.layer = QuantLstmLayer(input_size,hidden_size, memory_size, cell_module, go_forward)

  def _forward_packed(self, inputs, initial_state=None):
    sequence_tensor, batch_lengths = pad_packed_sequence(
        inputs, batch_first=self.batch_first)

    if self.batch_first is not True:
      sequence_tensor.transpose_(0, 1)
      
    output, final_state = self.layer(sequence_tensor, initial_state, batch_lengths)

    if self.batch_first is not True:
      output.transpose_(0, 1)
    output = pack_padded_sequence(
        output, batch_lengths, batch_first=self.batch_first)

    return output, final_state

  
    
  def forward(self, inputs, initial_state=None):
    if isinstance(inputs, PackedSequence):
      return self._forward_packed(inputs, initial_state)
    
    if self.batch_first is not True:
      inputs.transpose_(0, 1)
    output, final_state = self.layer(inputs, initial_state)
    
    if self.batch_first is not True:
      output.transpose_(0, 1)
    return output, final_state
  

class StackedGru(torch.nn.Module):
  def __init__(self,input_sizes, hidden_sizes, layers, stack_mode=None, batch_first=None):
    super(StackedGru, self).__init__()
    self.stack_mode = stack_mode
    self.num_layers = len(layers)
    self.lstm_layers = []
    self.batch_first = batch_first
    for layer_index in range(len(layers)):
      lstm_cell_pair = {}
      for direction, layer in layers[layer_index].items():
        layer = Gru(input_sizes[layer_index], hidden_sizes[layer_index], layer, direction, self.batch_first)
        self.add_module(f"{direction}_layer_{layer_index}", layer)
        lstm_cell_pair[direction] = layer
      self.lstm_layers.append(lstm_cell_pair)
  
  def forward(self, inputs, initial_state=None, **kwargs): 
    is_packed = isinstance(inputs, PackedSequence)
    if self.stack_mode in ['bidirectional']: 
      if initial_state is None:
        hidden_states = [None] * len(self.lstm_layers) * len(self.lstm_layers[0])
      elif initial_state.size()[0] != len(self.lstm_layers) * len(self.lstm_layers[0]):
        raise ValueError(f"initial states does not match the number of layers.")
      else:
        hidden_states = list(zip(initial_state.split(1, 0), initial_state.split(1, 0)))
    else: 
      if initial_state is None:
        hidden_states = [None] * len(self.lstm_layers)
      elif initial_state.size()[0] != len(self.lstm_layers):
        raise ValueError(f"initial states does not match the number of layers.")
      else:
        hidden_states = list(zip(initial_state.split(1, 0), initial_state.split(1, 0)))
   # print(f"nndct_inputs:{inputs}") 
    output_sequence = inputs
    
    
    if self.stack_mode in ['bidirectional']:
      final_h = []
      
      for i in range(len(self.lstm_layers)):
        forward_layer = getattr(self, "forward_layer_{}".format(i))
        backward_layer = getattr(self, "backward_layer_{}".format(i))
        forward_output, final_forward_state = forward_layer(output_sequence, hidden_states[2 * i])
        if self.batch_first is not True:
          output_sequence.transpose_(0,1)
        backward_output, final_backward_state = backward_layer(output_sequence, hidden_states[2 * i + 1])
        if is_packed:
          forward_output, lengths = pad_packed_sequence(forward_output, batch_first=self.batch_first)
          backward_output, _ = pad_packed_sequence(backward_output, batch_first=self.batch_first)
          
        output_sequence = torch.cat([forward_output, backward_output], -1)
        if is_packed:
          output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=self.batch_first)
        final_h.extend([final_forward_state[0], final_backward_state[0]])
      final_hidden_state = torch.cat(final_h, dim=0)
    else:
      final_states = []  
      output_reversed = False    
      for i, state in enumerate(hidden_states):
        if self.stack_mode == 'alternating':
          layer = getattr(self, f"forward_layer_{i}") if i % 2 == 0 else getattr(self, f"backward_layer_{i}")
        else: 
          layer = getattr(self, f"forward_layer_{i}")

        output_sequence, final_state = layer(output_sequence, state)
        
        
     #   print(f"nndct_layer{i} output:{output_sequence}")
        final_states.append(final_state)
        
      final_hidden_state = tuple(
            torch.cat(state_list, 0) for state_list in zip(*final_states)
        )
  #  print(f"nndct_final_output:{output_sequence}")
    return output_sequence, (final_hidden_state) 
  
class Gru(torch.nn.Module):
  def __init__(self, input_size, hidden_size, cell_module, direction, batch_first):
    super(Gru, self).__init__()
    # self.cell_module = cell_module
    # self.hidden_size = hidden_size
  
    self.batch_first = batch_first
    go_forward = True if direction == "forward" else False
    self.layer = QuantGruLayer(input_size,hidden_size, cell_module, go_forward)
  
  def _forward_packed(self, inputs, initial_state=None): 
    sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=self.batch_first)  
    
    if self.batch_first is not True:
      sequence_tensor.transpose_(0, 1)
      
    output, final_state = self.layer(sequence_tensor, initial_state, batch_lengths)

    if self.batch_first is not True:
      output.transpose_(0, 1)
    output = pack_padded_sequence(
        output, batch_lengths, batch_first=self.batch_first)

    return output, final_state
  
  def forward(self, inputs, initial_state=None):
    
    if isinstance(inputs, PackedSequence):
      return self._forward_packed(inputs, initial_state)
    
    if self.batch_first is not True:
      inputs.transpose_(0, 1)
    output, final_state = self.layer(inputs, initial_state)
    
    if self.batch_first is not True:
      output.transpose_(0, 1)
    return output, final_state
