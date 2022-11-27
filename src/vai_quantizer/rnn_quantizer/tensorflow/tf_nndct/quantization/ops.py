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

from tf_nndct import kernels

def fix_neuron(input, fp_tensor, bit_width, method=4):
  #return kernels.nndct_fix_neuron_v2(input, valmax, valamp, method)
  return kernels.nndct_fix_neuron(input, fp_tensor, bit_width, method)

def diffs_fix_pos(input, bit_width=8, range=5, method=4):
  return kernels.nndct_diff_s(input, bit_width, range, method)

def stat_act_pos(fp_tensor, fp_stat_tensor):
  return kernels.nndct_stat_act_pos(fp_tensor, fp_stat_tensor)

def scaleop(input, scale):
  return kernels.nndct_scale_op(input, float(scale))

def table_lookup(input, table, fragpos, type):
  return kernels.nndct_table_lookup(input, table, fragpos, type)

def simulation(input, type):
  return kernels.nndct_simulation(input, type)