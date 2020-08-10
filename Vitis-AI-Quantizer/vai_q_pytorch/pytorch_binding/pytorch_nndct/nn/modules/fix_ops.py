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
           "NndctTanhTableLookup"]
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
  if Tinput.device == torch.torch.device("cpu"):
    Tinput.mul_(scale)
  else:
    nndct_kernels.Scale(Tinput, scale)


def NndctFixNeuron(Tinput, Toutput, maxamp, method=2):
  valmax, valamp = maxamp[0], maxamp[1]
  if Tinput.device == torch.torch.device("cpu"):
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


def NndctDiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width=8, range=5, method=2):  
  nndct_kernels.DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method)


def NndctSigmoidTableLookup(Tinput, Ttable, Toutput, fragpos):
  nndct_kernels.SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos) 


def NndctTanhTableLookup(Tinput, Ttable, Toutput, fragpos):
  nndct_kernels.TanhTableLookup(Tinput, Ttable, Toutput, fragpos)


