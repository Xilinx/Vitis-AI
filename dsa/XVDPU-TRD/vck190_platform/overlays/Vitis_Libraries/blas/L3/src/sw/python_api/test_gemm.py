# Copyright 2019 Xilinx, Inc.
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


import numpy as np
import xfblas_L3 as xfblas
from test import Test

if __name__ == '__main__':
  args, xclbin_opts = xfblas.processCommandLine()
  xfblas.createGemm(args,xclbin_opts,1,0) # number of CUs, ith of device
  test = Test()
  test.test_basic_gemm(128,128,128,xclbin_opts,0,0) # ith of CU, ith of device
  