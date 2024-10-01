# Copyright 2021 Xilinx Inc.
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

from tensorflow.python.ops import array_ops

# Remove the deprecated argument 'validate_indices' from orginal signature
# of tf.gather so that the generated code by writer can match the correct
# argument. When we call
#   tf_nndct.ops.gather(t0, t1, t2),
# actually we will call:
#   tf.gather(t0, t1, axis=t2)
# See https://www.tensorflow.org/api_docs/python/tf/gather
def gather(params, indices, axis=None, batch_dims=0, name=None):
  return array_ops.gather_v2(
      params, indices, axis=axis, batch_dims=batch_dims, name=name)
