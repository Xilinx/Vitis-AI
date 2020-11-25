"""
A stacked LSTM with LSTM layers which alternate between going forwards over
the sequence and going backwards.
"""

from typing import Optional, Tuple, Union, List
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.common.checks import ConfigurationError

#import ctypes
import numpy as np
import time
import json
import dpu4rnn_py

TensorPair = Tuple[torch.Tensor, torch.Tensor]


class StackedAlternatingLstm(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards. This implementation is based on the
    description in `Deep Semantic Role Labelling - What works and what's next
    <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    use_input_projection_bias : bool, optional (default = True)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.

    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """


    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True,
    ) -> None:
        super().__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_lstm = dpu4rnn_py.dpu4rnn.create("openie")

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            go_forward = layer_index % 2 == 0
            layer = AugmentedLstm(
                lstm_input_size,
                hidden_size,
                go_forward,
                recurrent_dropout_probability=recurrent_dropout_probability,
                use_highway=use_highway,
                use_input_projection_bias=use_input_projection_bias,
            )
            lstm_input_size = hidden_size
            self.add_module("layer_{}".format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(
        self, inputs: PackedSequence, initial_state: Optional[TensorPair] = None
    ) -> Tuple[Union[torch.Tensor, PackedSequence], TensorPair]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: Tuple[torch.Tensor, torch.Tensor]
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        if not initial_state:
            hidden_states: List[Optional[TensorPair]] = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        t1 = time.time()
        dpu_time = 0
        with open("model/openie.json",'r') as load_f:
            load_dict = json.load(load_f)
        in_pos = load_dict[0]['lstm_in_float2fix']
        out_pos = load_dict[0]['lstm_out_fix2float']

        input_scale = 2.0**in_pos
        output_scale = 2.0**out_pos
        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        frame_num = len(sequence_tensor[0])
        zeros_cat = np.zeros((frame_num, 24), dtype=np.int16)
        frame_size = len(sequence_tensor[0]) * 224 * 2
        layer_output = []
        for i in range(sequence_tensor.shape[0]):
            hw_input = torch.floor(sequence_tensor.data[i] * input_scale).cpu().short().numpy()
            # padding the input as 224
            hw_input = np.concatenate((hw_input, zeros_cat), axis = 1)

            output = np.zeros(frame_num*300, dtype=np.int16)
            t3 = time.time()
            self.model_lstm.run(hw_input.flatten(), frame_size, output, frame_num);
            t4 = time.time()
            layer_output.append(output.reshape(frame_num, 300))
            dpu_time+=(t4-t3)
        torch_output = torch.from_numpy(np.array(layer_output, dtype=np.float32)/output_scale)
        output_sequence = pack_padded_sequence(torch_output, batch_lengths, batch_first=True)

        new_shape = (len(hidden_states), len(batch_lengths), len(output_sequence.data[0]))
        final_hidden_state = torch.zeros(new_shape)
        final_cell_state = torch.zeros(new_shape)
        t2 = time.time()
        print()
        print("DPU LSTM time: ", t2 - t1)
        print("DPU time (without memcpy): ", dpu_time)

        return output_sequence, (final_hidden_state, final_cell_state)
