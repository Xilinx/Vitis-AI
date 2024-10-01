# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch
import math
import functools
#
from nndct_shared.utils import (AddXopError, NndctOption, NndctScreenLogger,
                                option_util, QError, QWarning)
def logging_warn(message):
  NndctScreenLogger().warning2user(QWarning.FLOAT_OP, message)
#-----------------------------------------------------------------------------
# base structure

HANDLED_FUNCTIONS = {}
def implements(torch_function_list):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function_list)
    def decorator(func):
        for torch_fn in torch_function_list:
            HANDLED_FUNCTIONS[torch_fn] = func
        return func
    return decorator

class TraceTensor(torch.Tensor):
  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
      if kwargs is None:
          kwargs = {}
      if func not in HANDLED_FUNCTIONS:
          if hasattr(torch.Tensor, '__torch_function__'):
              return super().__torch_function__(func, types, args, kwargs)
          return torch_function_old_version(func, types, args, kwargs)
      return HANDLED_FUNCTIONS[func](*args, **kwargs)

def torch_function_old_version(func, types, args, kwargs):
    args_t = convert_tracetensor_to_tensor(args)
    ret = func(*args_t, **kwargs)
    ret = convert_tensor_to_tracetensor(ret)
    return ret

def convert_tensor_to_tracetensor(data):
    if isinstance(data, torch.Tensor):
        return data.as_subclass(TraceTensor)
    if not isinstance(data,(tuple, list)):
        return data
    new_data = []
    for item in data:
        new_data.append(convert_tensor_to_tracetensor(item))
    if isinstance(data, tuple):
        new_data = tuple(new_data)
    return new_data

def convert_tracetensor_to_tensor(data):
    if isinstance(data, TraceTensor):
        return data.as_subclass(torch.Tensor) 
    if not isinstance(data,(tuple, list)):
        return data
    new_data = []
    for item in data:
        new_data.append(convert_tracetensor_to_tensor(item))
    if isinstance(data, tuple):
        new_data = tuple(new_data)
    return new_data

def change2TwoNumberMultiplication(integer):
    start = int(math.sqrt(integer))
    factor = integer / start
    while not (int(factor) == factor):
        start += 1
        factor = integer / start
    return int(factor), start

class TraceTensorMode():
    def __init__(self, data, org_module=None):
        self.data = data
        self.org_module = org_module

    def __enter__(self):
        if hasattr(torch.Tensor, '__torch_function__'):
            self.trace_data = convert_tensor_to_tracetensor(self.data)
            return self.trace_data
        return self.data
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in dir(self.org_module):
            if isinstance(getattr(self.org_module, k), TraceTensor):
                setattr(self.org_module, k, convert_tracetensor_to_tensor(getattr(self.org_module, k)))
 
#-----------------------------------------------------------------------------
# over pytorch functions
# regist custom function replace python function

@implements([torch.split, torch.Tensor.split])
def split(tensor:TraceTensor, split_size_or_sections:'int|list[int]|tuple[int]', dim:int=0):
    def aten_split_with_sizes(tensor, split_size_or_sections, dim=0):
        if isinstance(split_size_or_sections, (list,tuple)):
            sum = 0
            for x in split_size_or_sections:
                sum = sum + x
            assert(sum == tensor.shape[dim])

        tensor_list_return = []
        last_end_index = 0
        for k in range(len(split_size_or_sections)):
            tmp = [slice(None, None) for x in tensor.shape]
            start = last_end_index
            end = start + split_size_or_sections[k]
            last_end_index = end
            tmp[dim] = slice(start,end)
            tensor_list_return.append(tensor[tmp])
        return tuple(tensor_list_return)

    def aten_split(tensor, split_size_or_sections, dim=0):
        if not isinstance(split_size_or_sections, int):
            assert(False)
        group_num = tensor.shape[dim] / split_size_or_sections
        group_num = math.ceil(group_num)
        if group_num == 0:
            group_num = 1
        tensor_list_return = []
        for k in range(group_num):
            tmp = [slice(None, None) for x in tensor.shape]
            start = k * split_size_or_sections
            if k == (group_num - 1):  
                end = tensor.shape[dim]
            else:
                end = (k + 1) * split_size_or_sections
            tmp[dim] = slice(start,end)
            tensor_list_return.append(tensor[tmp])
        return tuple(tensor_list_return)

    if isinstance(split_size_or_sections, (list, tuple)):
        return aten_split_with_sizes(tensor, split_size_or_sections, dim)
    if isinstance(split_size_or_sections, int):
        return aten_split(tensor, split_size_or_sections, dim)
    assert(False)
        
@implements([torch.Tensor.chunk, torch.chunk])
def chunk(input:TraceTensor, chunks:int, dim:int=0):
    assert(chunks != 0)
    split_size = math.ceil(input.shape[dim] / chunks)
    if split_size != 0:
       chunks = math.ceil(input.shape[dim] / split_size)
    tensor_list_return = []
    for k in range(chunks):
        tmp = [slice(None, None) for x in input.shape]
        start = k * split_size
        if k == (chunks - 1):  
            end = input.shape[dim]
        else:
            end = (k + 1) * split_size
        tmp[dim] = slice(start,end)
        tensor_list_return.append(input[tmp])
    return tuple(tensor_list_return)

def check_big_pooling(kernel_size, stride, padding):
    if not NndctOption.nndct_pooling_split_mode.value:
        return False
        
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding)  
    #------------------------
    # only support pooling == 0
    check_padding_flag = True
    if padding[0] != 0 or padding[1] != 0:
        check_padding_flag = False
    #------------------------
    # only support without overlap
    check_stride_flag = True
    if stride != None:
        if stride[0] != kernel_size[0] or stride[1] != kernel_size[1]:
            check_stride_flag = False
    #------------------------
    # only support kernel_size < 512
    check_kernel_size_flag = False
    if kernel_size[0] * kernel_size[1] >= 512:
        check_kernel_size_flag = True
    #------------------------
    base_check_result = check_stride_flag and check_kernel_size_flag and check_padding_flag
    #------------------------
    if base_check_result:
        # pre-check new kernel_size is equal to orgin kernel_size, prevent infinite recursion
        new_kernel_size1 = []
        new_kernel_size2 = []
        for size in kernel_size:
            a, b = change2TwoNumberMultiplication(size)
            new_kernel_size1.append(a)
            new_kernel_size2.append(b)
        new_kernel_size1 = tuple(new_kernel_size1)
        new_kernel_size2 = tuple(new_kernel_size2)
        if new_kernel_size1[0] == kernel_size[0] and new_kernel_size1[1] == kernel_size[1]:
            return False    
        if new_kernel_size2[0] == kernel_size[0] and new_kernel_size2[1] == kernel_size[1]:
            return False
    #------------------------
    return base_check_result
        
@implements([torch.nn.functional.avg_pool2d, torch.nn.AvgPool2d])
def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    
    # kernel_size (Union[int, Tuple[int, int]]) – the size of the window
    # stride (Union[int, Tuple[int, int]]) – the stride of the window. Default value is kernel_size
    # padding (Union[int, Tuple[int, int]]) – implicit zero padding to be added on both sides
    # ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape
    # count_include_pad (bool) – when True, will include the zero-padding in the averaging calculation
    # divisor_override (Optional[int]) – if specified, it will be used as divisor, otherwise size of the pooling region will be used.
    #-------------------------------------------------------
    # big kernel size split into two smaller kernels

    if check_big_pooling(kernel_size, stride, padding):
        logging_warn('big pooling split')
        new_kernel_size1 = []
        new_kernel_size2 = []
        for size in kernel_size:
            a, b = change2TwoNumberMultiplication(size)
            new_kernel_size1.append(a)
            new_kernel_size2.append(b)
        new_kernel_size1 = tuple(new_kernel_size1)
        new_kernel_size2 = tuple(new_kernel_size2)
        new_stride1 = new_kernel_size1
        new_stride2 = new_kernel_size2
        input = torch.nn.functional.avg_pool2d(input, new_kernel_size1, new_stride1, padding, ceil_mode, count_include_pad, divisor_override)
        input = torch.nn.functional.avg_pool2d(input, new_kernel_size2, new_stride2, padding, ceil_mode, count_include_pad, divisor_override)
        return input
    else:
        normal_data = input.as_subclass(torch.Tensor)
        normal_data = torch.nn.functional.avg_pool2d(normal_data, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        return normal_data.as_subclass(TraceTensor)

@implements([torch.nn.functional.max_pool2d, torch.nn.MaxPool2d])
def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    
    # kernel_size (Union[int, Tuple[int, int]]) – the size of the window to take a max over
    # stride (Union[int, Tuple[int, int]]) – the stride of the window. Default value is kernel_size
    # padding (Union[int, Tuple[int, int]]) – implicit zero padding to be added on both sides
    # dilation (Union[int, Tuple[int, int]]) – a parameter that controls the stride of elements in the window
    # return_indices (bool) – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
    # ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape
    #-----------------------------------------------
    # big kernel size split into two smaller kernels
    if check_big_pooling(kernel_size, stride, padding):
        logging_warn('big pooling split')
        new_kernel_size1 = []
        new_kernel_size2 = []
        for size in kernel_size:
            a, b = change2TwoNumberMultiplication(size)
            new_kernel_size1.append(a)
            new_kernel_size2.append(b)
        new_kernel_size1 = tuple(new_kernel_size1)
        new_kernel_size2 = tuple(new_kernel_size2)
        new_stride1 = new_kernel_size1
        new_stride2 = new_kernel_size2

        input = torch.nn.functional.max_pool2d(input, new_kernel_size1, new_stride1, padding, dilation, ceil_mode, return_indices)
        input = torch.nn.functional.max_pool2d(input, new_kernel_size2, new_stride2, padding, dilation, ceil_mode, return_indices)
        return input
    else:
        normal_data = input.as_subclass(torch.Tensor)
        normal_data = torch.nn.functional.max_pool2d(normal_data, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        return normal_data.as_subclass(TraceTensor)
    
@implements([torch.nn.functional.adaptive_avg_pool2d, torch.nn.AdaptiveAvgPool2d])
def adaptive_avg_pool2d(input, output_size):
    # output_size (Union[int, None, Tuple[Optional[int], Optional[int]]]) – the target output size of the image of the form H x W. Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int, or None which means the size will be the same as that of the input.
    if output_size == 1:
        kernel_size = [int(input.shape[-2]), int(input.shape[-1])]
        if check_big_pooling(kernel_size, kernel_size, 0):
            logging_warn('replace adapt pooling to pooling')
            return torch.nn.functional.avg_pool2d(input, kernel_size)
       
    normal_data = input.as_subclass(torch.Tensor)
    normal_data = torch.nn.functional.adaptive_avg_pool2d(normal_data, output_size)
    return normal_data.as_subclass(TraceTensor)

@implements([torch.nn.functional.adaptive_max_pool2d, torch.nn.AdaptiveMaxPool2d])
def adaptive_max_pool2d(input, output_size, return_indices=False):
    # output_size (Union[int, None, Tuple[Optional[int], Optional[int]]]) – the target output size of the image of the form  can be either a int, or None which means the size will be the same as that of the input.
    # return_indices (bool) – if True, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool2d. Default: False
    
    if output_size == 1:
        kernel_size = [int(input.shape[-2]), int(input.shape[-1])]
        if check_big_pooling(kernel_size, kernel_size, 0):
            logging_warn('replace adapt pooling to pooling')
            return torch.nn.functional.max_pool2d(input, kernel_size)
        
    normal_data = input.as_subclass(torch.Tensor)
    normal_data = torch.nn.functional.adaptive_max_pool2d(normal_data, output_size, return_indices)
    return normal_data.as_subclass(TraceTensor)

@implements([torch.mean, torch.Tensor.mean])
def mean(input, dim, keepdim=False, *, dtype=None, out=None):
    # Tensor.mean(dim=None, keepdim=False, *, dtype=None)
    if isinstance(dim, (list, tuple)) and len(dim) == 2:
        new_dim = []
        for k in dim:
            if k < 0:
                k = k + input.dim()
            new_dim.append(k)
        dim = new_dim

        if (input.dim() == 3 and tuple(dim) == (1, 2)) or (input.dim() == 4 and tuple(dim) == (2, 3)):
            kernel_size = [int(input.shape[-2]), int(input.shape[-1])]
            if check_big_pooling(kernel_size, kernel_size, 0):
                logging_warn('replace mean to avg pooling')
                data = torch.nn.functional.avg_pool2d(input, kernel_size)
                if not keepdim:
                    data = torch.squeeze(data, dim=-1)
                    data = torch.squeeze(data, dim=-1)
                return data
    
    normal_data = input.as_subclass(torch.Tensor)
    normal_data = torch.mean(normal_data, dim, keepdim=keepdim, dtype=dtype, out=out)
    return normal_data.as_subclass(TraceTensor)
    
#-----------------------------------------------------------------------------
# clear redundant op create by override

def clear_override_import_redundant_op(graph):
    # remove alias
    alias_node_list = graph.findAllNodes("aten::alias")
    for alias_node in alias_node_list:
      out_val = alias_node.output()
      inp_val = alias_node.inputsAt(0)
      out_val.replaceAllUsesWith(inp_val)
      alias_node.destroy()
