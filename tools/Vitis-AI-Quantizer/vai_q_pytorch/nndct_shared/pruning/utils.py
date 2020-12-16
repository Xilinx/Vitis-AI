

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

def tensor_out_in_axis(tensor):
  nndct_layouts = {2: 'OI', 4: 'OHWI'}
  data_format = nndct_layouts[tensor.ndim]
  out_axis = data_format.index('O')
  in_axis = data_format.index('I')
  return out_axis, in_axis
