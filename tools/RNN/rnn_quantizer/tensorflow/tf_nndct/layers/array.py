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

import tensorflow as tf

from tf_nndct.graph import OpTypes
from tf_nndct.quantization.utils import QuantizedModule
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.utils import NndctOption

@QuantizedModule(OpTypes.INPUT)
class Identity(tf.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    if self.quant_mode and self.quant_mode > 0:
      self._quant_vars = {}
    self.valid_inputs = None
    self.valid_output = None
    self.quant_info = None
    self.params_name = None
    self.node = None
    self.quant_vars_initialized = False
    self.params_quantized = False

  def _quant_vars_init(self):
    self._quant_vars['output_pos'] = self.quantizer.create_fp_tensor(
        name='output_pos',
        fp_name=self.valid_output[0])

  def __call__(self, input):
    if NndctOption.nndct_quant_off.value:
      output = tf.identity(input)
      return output

    if self.valid_output is not None and not self.quant_vars_initialized:
      self._quant_vars_init()
      self.quant_vars_initialized = True

    output = tf.identity(input)
    if self.valid_output is not None:
      output = self.quantizer.get_fp_and_quantize(
                   output,
                   self.valid_output[0],
                   self._quant_vars['output_pos'],
                   tensor_type='output')

    return output
