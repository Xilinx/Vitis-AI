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

import sys

from tensorflow.python.eager import context
from tensorflow.python.util import nest

# Inheritance chain:
# tf.Module -> keras.Layer -> keras.Dense -> tf_nndct.Dense
#     -> Inspectable -> QuantizedDense

# TODO(yuwang):
# 1. Use names without ambiguity.
# 2. Design a better machinism for Inspectable.
def make_quantized(base):

  class Inspectable(base):

    class Attr:
      SavingOutputs = '_saving_outputs'
      SavedOutputs = '_saved_outputs'

    def __call__(self, *args, **kwargs):
      outputs = super(Inspectable, self).__call__(*args, **kwargs)
      if getattr(self, self.Attr.SavingOutputs, False) and context.executing_eagerly():
        self._save(outputs)
      return outputs

    def enable_saving_outputs(self):
      setattr(self, self.Attr.SavingOutputs, True)

    def disable_saving_outputs(self):
      setattr(self, self.Attr.SavingOutputs, False)

    def _save(self, tensors):
      if not hasattr(self, self.Attr.SavedOutputs):
        setattr(self, self.Attr.SavedOutputs, [])

      batch_outputs = []
      for tensor in nest.flatten(tensors):
        batch_outputs.append(tensor.numpy())
      saved_outputs = getattr(self, self.Attr.SavedOutputs)
      saved_outputs.append(batch_outputs)

    def saved_outputs(self):
      return getattr(self, self.Attr.SavedOutputs, [])

  cls_name = base.__name__
  quantized_cls = type(cls_name, (Inspectable,), {})
  # Use base class __init__ directly so that the `KerasWriter` can inspect init args
  # automatically and write them out correctly.
  quantized_cls.__init__ = base.__init__
  setattr(sys.modules[__name__], cls_name, quantized_cls)
  return quantized_cls
