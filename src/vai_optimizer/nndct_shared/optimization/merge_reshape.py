

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

from nndct_shared.base import NNDCT_OP


class ReshapeMergeHandler(object):
  def __init__(self):
    self.visited = set()
  
  def __call__(self, *args, **kwargs):
    _, node_set = args
    reshape_0 = node_set[0]
    reshape_1 = node_set[1]

    if reshape_0 in self.visited or reshape_1 in self.visited:
      return 
    
    self.visited.add(reshape_0)
    self.visited.add(reshape_1)
  
    uses = list(reshape_0.out_tensors[0].uses)
    for use in uses:
      if use.user.op.type == NNDCT_OP.RESHAPE:
        use.user.replace_input_at(use.offset, reshape_0.in_tensors[0])
  