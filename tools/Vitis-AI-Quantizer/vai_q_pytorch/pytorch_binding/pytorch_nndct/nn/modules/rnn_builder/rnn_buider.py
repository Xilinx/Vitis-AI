

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