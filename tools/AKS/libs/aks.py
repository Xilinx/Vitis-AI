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

import numpy as np
import ctypes
import os
import sys

# New Python Extensions
default_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

import libaks

SysManager         = libaks.SysManager
NodeParams         = libaks.NodeParams
DynamicParamValues = libaks.DynamicParamValues

sys.setdlopenflags(default_flags)
