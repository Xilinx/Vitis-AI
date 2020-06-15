import nndct_shared.utils.tensor_util as tensor_util
from nndct_shared.base import FrameworkType

def param_to_nndct_format(tensor):
  tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.TORCH,
              FrameworkType.NNDCT)



def param_to_torch_format(tensor):
  tensor_util.convert_parameter_tensor_format(
              tensor, FrameworkType.NNDCT,
              FrameworkType.TORCH)


def blob_to_nndct_format(tensor):
  tensor_util.convert_blob_tensor_format(tensor,
              FrameworkType.TORCH,
              FrameworkType.NNDCT)
  
  

def blob_to_torch_format(tensor):
  tensor_util.convert_blob_tensor_format(tensor,
              FrameworkType.NNDCT,
              FrameworkType.TORCH)