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
import torch

from pytorch_nndct.utils.torch_utils import CmpFlag, compare_torch_version

def get_opset_version():
  if "_onnx_stable_opsets" in torch.onnx.symbolic_helper.__dict__:
    return  torch.onnx.symbolic_helper._onnx_stable_opsets[-1]
  elif "onnx_stable_opsets" in torch.onnx._constants.__dict__:
    return  torch.onnx._constants.onnx_stable_opsets[-1]
  elif "ONNX_MAX_OPSET" in torch.onnx._constants.__dict__:
    return  torch.onnx._constants.ONNX_MAX_OPSET
  else:
    raise RuntimeError("Onnx stable opset version is not found. Please check pytorch version (1.4.0 ~ 1.13.0)")

def support_onnx_export():
  return compare_torch_version(CmpFlag.GREATER_EQUAL, "1.7")
