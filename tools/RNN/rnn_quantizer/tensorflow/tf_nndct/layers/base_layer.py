

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

import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.util import nest

class Layer(keras.layers.Layer):
  def __init__(self, trainable=True, name=None, dtype=None, dynamic=False,
               **kwargs):
    super(Layer, self).__init__(trainable, name, dtype, dynamic, **kwargs)
    self._dumping_outputs = False

  #def __call__(self, *args, **kwargs):
  #  outputs = super(Layer, self).__call__(*args, **kwargs)
  #  print('context.executing_eagerly:', context.executing_eagerly())
  #  #if self.dump_outputs and context.executing_eagerly():
  #  if self.dump_outputs:
  #    #tf.print(*outputs, output_stream='file:///tmp/dump_data')
  #    for output in nest.flatten(outputs):
  #      print(output.numpy())
  #  return outputs

  def enable_dumping_outputs(self):
    self._dumping_outputs = True

  def disable_dumping_outputs(self):
    self._dumping_outputs = False

  def call(self, *args, **kwargs):
    outputs = self._internal_call(*args, **kwargs)

    if self._dumping_outputs and context.executing_eagerly():
      self._dump(outputs)

    return outputs

  def _dump(self, tensors):
    for tensor in nest.flatten(tensors):
      #print(tensor.numpy())
      pass
