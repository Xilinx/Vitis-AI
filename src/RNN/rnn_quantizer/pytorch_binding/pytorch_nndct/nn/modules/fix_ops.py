

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


# import os
# import sys
import torch
from ..load_kernels import nndct_kernels
import copy
import numpy as np
# from torch.utils.cpp_extension import load
__all__ = ["NndctScale", \
           "NndctFixNeuron",
           "NndctDiffsFixPos",\
           "NndctSigmoidTableLookup",\
           "NndctSigmoidSimulation",\
           "NndctTanhTableLookup",\
           "NndctTanhSimulation"]
# try:
#   dir_path = os.path.dirname(os.path.realpath(__file__))
#   extra_include_paths=[os.path.join(dir_path,"../include")]
#   fix_kernels=load(name="fix_kernels",
#               sources=[os.path.join(dir_path,"../src/nndct_fixneuron_op.cpp"),
#                        os.path.join(dir_path,"../src/nndct_diffs_op.cpp"),
#                        os.path.join(dir_path,"../src/register_kernels.cpp"),
#                        os.path.join(dir_path,"../src/nndct_fix_kernels.cu"),
#                        os.path.join(dir_path,"../src/nndct_cuda_math.cu"),
#                        os.path.join(dir_path,"../src/nndct_cu_utils.cc")],

#               verbose=True,
#               extra_include_paths=extra_include_paths
#               )
# except ImportError as e:
#   print(f"Import 'fix_kernels module' failed.({str(e)})")
#   sys.exit(1)
# else:
#   print("Import 'fix_kernels module' successfully.")


def NndctScale(Tinput, scale):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  nndct_kernels.Scale(Tinput, scale, device_id)
  '''
  if Tinput.device == torch.device("cpu"):
    Tinput.mul_(scale)
  else:
    nndct_kernels.Scale(Tinput, scale)
  '''


def NndctFixNeuron(Tinput, Toutput, maxamp, method=2):
  valmax, valamp = maxamp[0], maxamp[1]
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  nndct_kernels.FixNeuronV2(Tinput, Toutput, valmax, valamp, method, device_id)
  return Toutput
  '''
  if Tinput.device == torch.device("cpu"):
    output = Tinput.cuda()
    nndct_kernels.FixNeuronV2(output, output, valmax,
                              valamp, method)
    Tinput.copy_(output.cpu())
    return Tinput

    # cpu fix neuron
    """
    # output = Tinput.cpu().detach().numpy()
    # output = output * valamp
    # if method == 2:
    #   output = np.where(output > valmax - 1, (valmax - 1), output)
    #   output = np.where(output < (-valmax), -valmax, output)
    #   output = np.where(np.logical_and(output > 0, np.logical_and(np.floor(output) % 2 == 0, output - np.floor(output) == 0.5)), np.ceil(output), output)
    #   output = np.where(output >= 0, np.round(output), output)
    #   output = np.where(np.logical_and(output < 0, output - np.floor(output) == 0.5), np.ceil(output), output)
    #   output = np.where(output < 0, np.round(output), output)

    # elif method == 3:
    #   output = np.where(output > valmax - 1, (valmax - 1), output)
    #   output = np.where(output < (-valmax), -valmax, output)
    #   output = np.where(np.logical_and(output > 0, np.logical_and(np.floor(output) % 2 == 0, output - np.floor(output) == 0.5)), np.ceil(output), output)
    #   output = np.where(output >= 0, np.round(output), output)
    #   output = np.where(np.logical_and(output < 0, np.logical_and(np.ceil(output) % 2 == 0, output - np.floor(output) == 0.5)), np.floor(output), output)
    #   output = np.where(output < 0, np.round(output), output)

    # Tinput.copy_(torch.from_numpy(output))
    # Tinput.div_(valamp)
    # return Tinput
    """
  else:
    nndct_kernels.FixNeuronV2(Tinput, Toutput, valmax,
                              valamp, method)
  return Toutput
  '''


def NndctDiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width=8, range=5, method=2):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  nndct_kernels.DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)


def NndctSigmoidTableLookup(Tinput, Ttable, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  nndct_kernels.SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)

def NndctSigmoidSimulation(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Sigmoid simulation dose not support CPU")
  else:
    nndct_kernels.SigmoidSimulation(Tinput, Toutput, device_id)


def NndctTanhTableLookup(Tinput, Ttable, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  nndct_kernels.TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)

def NndctTanhSimulation(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Tanh simulation dose not support CPU")
  else:
    nndct_kernels.TanhSimulation(Tinput, Toutput, device_id)

