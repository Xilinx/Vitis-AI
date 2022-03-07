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

import numpy as np
import nndct_shared.utils.tensor_util as tu

from nndct_shared.base import FrameworkType

def param_to_nndct_format(tensor):
  tu.convert_parameter_tensor_format(tensor, FrameworkType.TORCH,
                                     FrameworkType.NNDCT)

def param_to_torch_format(tensor):
  tu.convert_parameter_tensor_format(tensor, FrameworkType.NNDCT,
                                     FrameworkType.TORCH)

# def blob_to_nndct_format(tensor):
#   tu.convert_blob_tensor_format(tensor, FrameworkType.TORCH,
#                                 FrameworkType.NNDCT)

# def blob_to_torch_format(tensor):
#   tu.convert_blob_tensor_format(tensor, FrameworkType.NNDCT,
#                                 FrameworkType.TORCH)


    