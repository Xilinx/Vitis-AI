
from functools import wraps
import torch

def pre_and_post_process_f16_tensor(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    tensor_type_list = []
    tensor_type_dict = {}
    out_need_convert = False
    for arg in args:
      if isinstance(arg, torch.Tensor):
        tensor_type_list.append(arg.dtype)
        if arg.dtype == torch.float16:
          arg.data = arg.data.to(torch.float32)
          out_need_convert = True
    for key, value in kwargs.items():
      if isinstance(value, torch.Tensor):
        tensor_type_dict[key] = value.dtype
        if value.dtype == torch.float16:
          value.data = value.data.to(torch.float32)
          out_need_convert = True
    out = func(*args, **kwargs)
    if out_need_convert:
      if isinstance(out, torch.Tensor):
        if out.dtype == torch.float32:
          out.data = out.data.to(torch.float16)
      
      #for i in range(len(args)):
      #  if isinstance(args[i], torch.Tensor):
      #    if args[i].dtype != tensor_type_list[i]:
      #      args[i].data = args[i].data.to(tensor_type_list[i])
      
      index = 0
      for arg in args:
        if isinstance(arg, torch.Tensor):
          if arg.dtype != tensor_type_list[index]:
            arg.data = arg.data.to(tensor_type_list[index])
            index = index + 1

      for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
          if value.dtype != tensor_type_dict[key]:
            value.data = value.data.to(tensor_type_dict[key])
    return out
      
  return wrapper
