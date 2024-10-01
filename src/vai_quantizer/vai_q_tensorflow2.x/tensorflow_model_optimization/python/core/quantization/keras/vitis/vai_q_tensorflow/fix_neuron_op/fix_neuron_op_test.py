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

"""Python wrapper for the fix neuron operators"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.decent_q.ops import gen_fix_neuron_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


_fix_neuron_ops_cu = load_library.load_op_library(
    resource_loader.get_path_to_datafile("../gen_files/libvaifncuda.so"))
_fix_neuron_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile("../gen_files/libvaifn.so"))

import pdb; pdb.set_trace()
# fix_neuron = gen_fix_neuron_ops.fix_neuron

@ops.RegisterGradient("FixNeuron")
def _FixNeuronGrad(_, grad):
    return grad
