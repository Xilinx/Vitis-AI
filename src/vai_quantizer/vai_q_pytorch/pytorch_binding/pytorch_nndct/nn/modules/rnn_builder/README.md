## Introduction to Customized LSTM Quantization

  1. A customized stacked LSTM can be rebuild using NNDCT LSTM layer as a basic block. 
  2. The front-end of NNDCT can capture computation graphs through these layers. Then, the compuation graphs converted to a quantizable stacked LSTM model.

## NNDCT LSTM Layer API
- *NNDCT LSTM layer only support single layer definition. Therefore, NNDCT loses the connection between the layers. Currently, only support alternating connection mode in NNDCT.*
- *NNDCT LSTM layer can only be used during inference phase*
```python
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
      self.input = input[:1]
      self.initial_state = initial_state[:1] if initial_state is not None else None
      self.batch_lengths = batch_lengths[:1] if batch_lengths is not None else None
      
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
      timestep_output, memory = self.lstm_cell_module(input[:current_length_index+1, index], h_prev, c_prev)
      full_batch_c[0:current_length_index + 1] = memory
      full_batch_h[0:current_length_index + 1] = timestep_output
      output[:current_length_index+1, index] = timestep_output

    final_state = (
        full_batch_h.unsqueeze(0),
        full_batch_c.unsqueeze(0),
    )
    return output, final_state
```
## Rewrite a Customized LSTM
Let's examine a example, to learn how to rewrite your own LSTM:
```python
# Augmented LSTM in allennlp package
import pytorch_nndct.nn as nndct_nn

class AugmentedLstm(torch.nn.Module):
  def __init__(
    self,
    input_size: int,
    hidden_size: int,
    go_forward: bool = True,
    recurrent_dropout_probability: float = 0.0,
    use_highway: bool = True,
    use_input_projection_bias: bool = True,
      ) -> None:

    super().__init__()
    # Original initialization in AugmentedLstm class
    """
   
    self.input_size = input_size
    self.hidden_size = hidden_size

    self.go_forward = go_forward
    self.use_highway = use_highway
    self.recurrent_dropout_probability = recurrent_dropout_probability

    if use_highway:
        self.input_linearity = torch.nn.Linear(
            input_size, 6 * hidden_size, bias=use_input_projection_bias
        )
        self.state_linearity = torch.nn.Linear(hidden_size, 5 * hidden_size, bias=True)
    else:
        self.input_linearity = torch.nn.Linear(
            input_size, 4 * hidden_size, bias=use_input_projection_bias
        )
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=True)
    """
    # Using NNDCT LSTM layer to redefine a customized LSTM.

    self.input_size = input_size
    self.hidden_size = hidden_size

    # 1. Extract a LSTM cell(without loop) from forward function.
    cell_module = AugmentedLstmCell(input_size, hidden_size, use_input_projection_bias, use_highway)

    # 2. Using the cell module to define a NNDCT LSTM layer.
    self.layer = nndct_nn.QuantLstmLayer(input_size, hidden_size, hidden_size, cell_module, go_forward)

    # 3. Mapping parameter layers to facilitate loading parameters.
    self.input_linearity = self.layer.lstm_cell_module.input_linearity
        self.state_linearity = self.layer.lstm_cell_module.state_linearity
   
  def forward(
      self,
      inputs: PackedSequence,
      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  ):
    # Original forward in AugmentedLstm class
    """
    if not isinstance(inputs, PackedSequence):
        raise ConfigurationError("inputs must be PackedSequence but got %s" % (type(inputs)))

    sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
    batch_size = sequence_tensor.size()[0]
    total_timesteps = sequence_tensor.size()[1]

    output_accumulator = sequence_tensor.new_zeros(
        batch_size, total_timesteps, self.hidden_size
    )
    if initial_state is None:
        full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.hidden_size)
        full_batch_previous_state = sequence_tensor.new_zeros(batch_size, self.hidden_size)
    else:
        full_batch_previous_state = initial_state[0].squeeze(0)
        full_batch_previous_memory = initial_state[1].squeeze(0)

    current_length_index = batch_size - 1 if self.go_forward else 0
    if self.recurrent_dropout_probability > 0.0:
        dropout_mask = get_dropout_mask(
            self.recurrent_dropout_probability, full_batch_previous_memory
        )
    else:
        dropout_mask = None

    # The time step iteration can be replaced by NNDCT LSTM layer.
    for timestep in range(total_timesteps):
      index = timestep if self.go_forward else total_timesteps - timestep - 1
      if self.go_forward:
        while batch_lengths[current_length_index] <= index:
            current_length_index -= 1
      
      else:
        while (
            current_length_index < (len(batch_lengths) - 1)
            and batch_lengths[current_length_index + 1] > index
        ):
            current_length_index += 1

      previous_memory = full_batch_previous_memory[0 : current_length_index + 1].clone()
      previous_state = full_batch_previous_state[0 : current_length_index + 1].clone()
     
      # NNDCT LSTM layer is designed only for inference. So droput_mask is invalid in NNDCT LSTM layer.
      if dropout_mask is not None and self.training:
        previous_state = previous_state * dropout_mask[0 : current_length_index + 1]
      timestep_input = sequence_tensor[0 : current_length_index + 1, index]

  
      projected_input = self.input_linearity(timestep_input)
      projected_state = self.state_linearity(previous_state)


      input_gate = torch.sigmoid(
          projected_input[:, 0 * self.hidden_size : 1 * self.hidden_size]
          + projected_state[:, 0 * self.hidden_size : 1 * self.hidden_size]
      )
      forget_gate = torch.sigmoid(
          projected_input[:, 1 * self.hidden_size : 2 * self.hidden_size]
          + projected_state[:, 1 * self.hidden_size : 2 * self.hidden_size]
      )
      memory_init = torch.tanh(
          projected_input[:, 2 * self.hidden_size : 3 * self.hidden_size]
          + projected_state[:, 2 * self.hidden_size : 3 * self.hidden_size]
      )
      output_gate = torch.sigmoid(
          projected_input[:, 3 * self.hidden_size : 4 * self.hidden_size]
          + projected_state[:, 3 * self.hidden_size : 4 * self.hidden_size]
      )
      memory = input_gate * memory_init + forget_gate * previous_memory
      timestep_output = output_gate * torch.tanh(memory)

      if self.use_highway:
          highway_gate = torch.sigmoid(
              projected_input[:, 4 * self.hidden_size : 5 * self.hidden_size]
              + projected_state[:, 4 * self.hidden_size : 5 * self.hidden_size]
          )
          highway_input_projection = projected_input[
              :, 5 * self.hidden_size : 6 * self.hidden_size
          ]
          timestep_output = (
              highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection
          )

      full_batch_previous_memory = full_batch_previous_memory.clone()
      full_batch_previous_state = full_batch_previous_state.clone()
      full_batch_previous_memory[0 : current_length_index + 1] = memory
      full_batch_previous_state[0 : current_length_index + 1] = timestep_output
      output_accumulator[0 : current_length_index + 1, index] = timestep_output

    output_accumulator = pack_padded_sequence(
        output_accumulator, batch_lengths, batch_first=True
    )

    final_state = (
        full_batch_previous_state.unsqueeze(0),
        full_batch_previous_memory.unsqueeze(0),
    )

    return output_accumulator, final_state
    """
    # Rewrite forward with NNDCT LSTM layer.
    if not isinstance(inputs, PackedSequence):
      raise ConfigurationError("inputs must be PackedSequence but got %s" % (type(inputs)))

      sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
      # Replace time step loop with NNNDCT layer module
      output_accumulator, final_state = self.layer(sequence_tensor, initial_state, batch_lengths)
      
      output_accumulator = pack_padded_sequence(
          output_accumulator, batch_lengths, batch_first=True
      )

      return output_accumulator, final_state



# Extract LSTM cell as a module.
class AugmentedLstmCell(torch.nn.Module):
  def __init__(
      self,
      input_size: int,
      hidden_size: int,
      bias: bool, 
      use_highway: bool = True,
  ) -> None:
    super().__init__()
    self.use_highway = use_highway
    self.hidden_size = hidden_size
    if use_highway:
        self.input_linearity = torch.nn.Linear(
            input_size, 6 * hidden_size, bias=bias
        )
        self.state_linearity = torch.nn.Linear(hidden_size, 5 * hidden_size, bias=True)
    else:
        self.input_linearity = torch.nn.Linear(
            input_size, 4 * hidden_size, bias=bias
        )
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        
    
  def forward(self, input, previous_state=None, previous_memory=None):
    projected_input = self.input_linearity(input)
    projected_state = self.state_linearity(previous_state)

    input_gate = torch.sigmoid(
        projected_input[:, 0 * self.hidden_size : 1 * self.hidden_size]
        + projected_state[:, 0 * self.hidden_size : 1 * self.hidden_size]
    )
    forget_gate = torch.sigmoid(
        projected_input[:, 1 * self.hidden_size : 2 * self.hidden_size]
        + projected_state[:, 1 * self.hidden_size : 2 * self.hidden_size]
    )
    memory_init = torch.tanh(
        projected_input[:, 2 * self.hidden_size : 3 * self.hidden_size]
        + projected_state[:, 2 * self.hidden_size : 3 * self.hidden_size]
    )
    output_gate = torch.sigmoid(
        projected_input[:, 3 * self.hidden_size : 4 * self.hidden_size]
        + projected_state[:, 3 * self.hidden_size : 4 * self.hidden_size]
    )
    memory = input_gate * memory_init + forget_gate * previous_memory
    timestep_output = output_gate * torch.tanh(memory)

    if self.use_highway:
        highway_gate = torch.sigmoid(
            projected_input[:, 4 * self.hidden_size : 5 * self.hidden_size]
            + projected_state[:, 4 * self.hidden_size : 5 * self.hidden_size]
        )
        highway_input_projection = projected_input[
            :, 5 * self.hidden_size : 6 * self.hidden_size
        ]
        timestep_output = (
            highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection
        )
    return timestep_output, memory
```