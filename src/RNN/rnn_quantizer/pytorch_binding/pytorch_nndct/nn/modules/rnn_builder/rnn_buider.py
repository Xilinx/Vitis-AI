from .rnn import *

class DynamicRnnBuilder(object):

  def __call__(self,
               rnn_type,
               input_sizes,
               hidden_sizes,
               memory_sizes,
               layers,
               stack_mode=None,
               batch_first=None):
    if rnn_type == "LSTM":
      return {
        "LSTM": StackedLstm
      }.get(rnn_type, None)(
        input_sizes, memory_sizes, hidden_sizes, layers, stack_mode, batch_first)
    elif  rnn_type == "GRU":
      return {
        "GRU": StackedGru
      }.get(rnn_type, None)(
        input_sizes,  hidden_sizes, layers, stack_mode, batch_first)