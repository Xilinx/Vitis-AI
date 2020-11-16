

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

import imp
import os
import sys
import torch
from torch.utils.cpp_extension import load, _import_module_from_library
from nndct_shared.utils import create_work_dir, NndctScreenLogger

_cur_dir = os.path.dirname(os.path.realpath(__file__))
_aot = False

for name in os.listdir(_cur_dir):
  if name.split(".")[-1] == "so":
    _aot = True
    break

if _aot:
  try:
    from pytorch_nndct.nn import _kernels
    nndct_kernels = _kernels
  except ImportError as e:
    NndctScreenLogger().error(f"{str(e)}")
    sys.exit(1)
  else:
    NndctScreenLogger().info(f"Loading NNDCT kernels...")
    
else:    
  if os.path.exists(os.path.join(_cur_dir, "kernel")):
    from .kernel import NN_PATH
  else:
    NN_PATH = _cur_dir
  try:
    cwd = NN_PATH
    lib_path = os.path.join(cwd, "lib")
    create_work_dir(lib_path)
    cpu_src_path = os.path.join(cwd, "../../../csrc/cpu")
    source_files = []
    for name in os.listdir(cpu_src_path):
      if name.split(".")[-1] in ["cpp", "cc", "c"]:
        source_files.append(os.path.join(cpu_src_path, name))

    extra_include_paths = [
        os.path.join(cwd, "../../../include/cpu"),
        os.path.join(cwd, "include")
    ]
    
    with_cuda = False
    #if torch.cuda.is_available() and "CUDA_HOME" in os.environ:
    if "CUDA_HOME" in os.environ:
      cuda_src_path = os.path.join(cwd, "../../../csrc/cuda")
      for name in os.listdir(cuda_src_path):
        if name.split(".")[-1] in ["cu", "cpp", "cc", "c"]:
          source_files.append(os.path.join(cuda_src_path, name))
          
      cpp_src_path = os.path.join(cwd, "src/cuda")
      for name in os.listdir(cpp_src_path):
        if name.split(".")[-1] in ["cpp", "cc", "c"]:
          source_files.append(os.path.join(cpp_src_path, name))
      
      extra_include_paths.append(os.path.join(cwd, "../../../include/cuda"))
      with_cuda = None
    else:
      print("CUDA is not available, or CUDA_HOME not found in the environment " 
          "so building without GPU support.")
      cpp_src_path = os.path.join(cwd, "src/cpu")
      for name in os.listdir(cpp_src_path):
        if name.split(".")[-1] in ["cpp", "cc", "c"]:
          source_files.append(os.path.join(cpp_src_path, name))
    
    nndct_kernels = load(
        name="nndct_kernels",
        sources=source_files,
        verbose=False,
        build_directory=lib_path,
        extra_include_paths=extra_include_paths,
        with_cuda=with_cuda)
    
  except ImportError as e:
    NndctScreenLogger().error(f"{str(e)}")
    sys.exit(1)
  else:
    NndctScreenLogger().info(f"Loading NNDCT kernels...")
