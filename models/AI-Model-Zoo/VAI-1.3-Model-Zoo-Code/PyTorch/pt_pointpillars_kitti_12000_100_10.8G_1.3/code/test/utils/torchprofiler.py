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

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from collections import OrderedDict

    
class Profiler(object):
    def __init__(self, model, device = None, caffe_style = False):
        super(Profiler, self).__init__()
        if device == None:
            self.device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        else:
            if not device.lower() in (None, 'cuda', 'cpu'):
                raise ValueError('Unsupported device: {}'.format(device))
            self.device = device.lower()

        self.model = getattr(model, self.device)().eval()
        self.dtype = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor          
        self.caffe_style = caffe_style

        self.hooks = []
        self.initialize_hooks()
        
        self._summary = OrderedDict()
        self._completed = False

    def initialize_hooks(self):

        def register_hook(module):

            def hook(module, input, output):
                class_name = module.__class__.__name__
                module_idx = len(self._summary)

                m_key = '%s-%i' % (class_name, module_idx + 1)
                self._summary[m_key] = OrderedDict()

                # shape
                self._summary[m_key]['input_shape'] = get_data_shape(input[0])
                if isinstance(output, (list, tuple)):
                    self._summary[m_key]['output_shape'] = [
                        list(o.size()) for o in output
                    ]
                else:
                    self._summary[m_key]['output_shape'] = [list(output.size())]

                # params
                '''
                params = 0
                if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    self._summary[m_key]['trainable'] = module.weight.requires_grad
                if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                self._summary[m_key]['nb_params'] = params
                '''
                self._summary[m_key]['nb_params'] = get_params_general(module, self.caffe_style)

                # ops
                ops = 0
                if class_name in get_ops_functions:
                    ops = get_ops_functions[class_name](module, input[0], output)
                elif module.__class__.__bases__[0].__name__ in get_ops_functions:
                    base_class_name = module.__class__.__bases__[0].__name__
                    print('Assuming {} is a {} layer'.format(class_name, base_class_name))
                    ops = get_ops_functions[base_class_name](module, input[0], output)
                elif len(list(module.children())) == 0:
                    print('Unknown module type for flops calculation: {0}'.format(class_name))
                    ops = 0
                else:
                    # This is module sequentials, modulelists or user-defined modules
                    # such as bottlenecks / resblocks
                    # The total flops will be calculated by recursively calling the submodules
                    # So just return 0 here 
                    ops = 0
                self._summary[m_key]['flops'] = ops

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (len(list(self.model.children())) > 0 and module == self.model)
            ):
                self.hooks.append(module.register_forward_hook(hook))
        self.model.apply(register_hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def run_with_input(self, *inputs):
        self._summary = OrderedDict()
        self._completed = False
        # Assume that the main input is the first one
        self.input_shape = [get_data_shape(x) for x in inputs]
        self.model(*inputs)
        self._completed = True

    def run(self, *input_shape):
        self._summary = OrderedDict()
        self._completed = False
        self.input_shape = input_shape
        x = [Variable(torch.rand(*in_shape).type(self.dtype)) for in_shape in input_shape]
        self.model(*x)
        self._completed = True
    
    @property
    def total_input(self):
        if not self._completed:
            raise RuntimeError('Profiling not completed. Please use .run() first.')
        total_input = 0
        for shape in self.input_shape:
            total_input += abs(np.prod(shape))
        return total_input

    @property
    def total_output(self):
        if not self._completed:
            raise RuntimeError('Profiling not completed. Please use .run() first.')
        total_output = 0
        for layer in self._summary:
            for out in self._summary[layer]['output_shape']:
                total_output += abs(np.prod(out))
        return abs(int(total_output))

    @property
    def total_params(self):
        if not self._completed:
            raise RuntimeError('Profiling not completed. Please use .run() first.')
        total_params = 0
        for layer in self._summary:
            total_params += self._summary[layer]['nb_params']['total']
        return abs(int(total_params))

    @property
    def total_flops(self):
        if not self._completed:
            raise RuntimeError('Profiling not completed. Please use .run() first.')
        total_flops = 0
        for layer in self._summary:
            total_flops += self._summary[layer]['flops']
        return abs(int(total_flops))

    @property
    def trainable_params(self):
        if not self._completed:
            raise RuntimeError('Profiling not completed. Please use .run() first.')
        trainable_params = 0
        for layer in self._summary:
            trainable_params += self._summary[layer]['nb_params']['trainable']
        return abs(int(trainable_params))
        

    def print_summary(self):
        total_params = 0
        total_output = 0
        total_flops = 0
        trainable_params = 0
        print('-' * 80)
        line_new = '{:>20}  {:>25} {:>15} {:>15}'.format(
            'Layer (type)', 
            'Output Shape', 
            'Params #',
            'Flops #')
        print(line_new)
        print('=' * 80)
        for layer in self._summary:
            for i, output_shape in enumerate(self._summary[layer]['output_shape']):
                if i == 0:
                    line_new = '{:>20} {:>25} {:>15} {:>15}'.format(
                        layer,
                        str(output_shape),
                        '{0:,}'.format(self._summary[layer]['nb_params']['total']),
                        '{0:,}'.format(self._summary[layer]['flops']),
                    )
                else:
                    line_new = '{:>20} {:>25} {:>15} {:>15}'.format(
                        '',
                        str(output_shape),
                        '',
                        '',
                    )
                print(line_new)

        total_input = self.total_input
        total_output = self.total_output
        total_flops = self.total_flops
        total_params = self.total_params
        trainable_params = self.trainable_params

        itemsize = np.array(self.dtype(0)).itemsize

        total_input_size = total_input * itemsize / (1024 ** 2.)
        total_output_size = total_output * itemsize  / (1024 ** 2.)  
        total_params_size = total_params * itemsize  / (1024 ** 2.)
        total_size = total_params_size + total_output_size * 2 + total_input_size
        
        print('-' * 80)
        print('Total flops: {0:,}'.format(total_flops))
        print('Total params: {0:,}'.format(total_params))
        print('Trainable params: {0:,}'.format(trainable_params))
        print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        print('-' * 80)
        print('Input size (MB): {:.2f}'.format(total_input_size))
        print('Forward/backward pass size (MB): {:.2f}'.format(total_output_size * 2)) # x2 for gradients
        print('Params size (MB): {:.2f}'.format(total_params_size))
        print('Estimated Total Size (MB): {:.2f}'.format(total_size))
        print('-' * 80)

def get_params_general(module, caffe_style = False):
    # params
    params = 0
    trainable = 0
    param_list = ('weight', 'bias', 'running_mean', 'running_var')
    for param_type in param_list:
        if hasattr( module, param_type) and hasattr(getattr(module, param_type), 'numel'):
            params += getattr(module, param_type).numel()
            if hasattr(getattr(module, param_type),'requires_grad') \
                and getattr(module, param_type).requires_grad:
                trainable += getattr(module, param_type).numel()

    # Note: 1 more param for BatchNorm caffe implementation
    if 'BatchNorm' in module.__class__.__name__ and caffe_style:
        params += 1

    result = {
        'total': params,
        'trainable': trainable,
        }
    return result


def get_ops_conv2d(module, input_data, output_data):
    # module.weight.shape = (out, in, kh, kw)
    ops_each_output = module.weight[0].numel() * 2
    if isinstance(module.bias, Variable):
        ops_each_output += 1
    return ops_each_output * output_data.numel()

def get_ops_convtransposed2d(module, input_data, output_data):
    # module.weight.shape = (in, out, kh, kw)
    ops_each_input = module.weight[0].numel() * 2
    result = ops_each_input * input_data.numel()
    if isinstance(module.bias, Variable):
        result += output_data.numel()
    return result


def get_ops_linear(module, input_data, output_data):
    # module.weight.shape = (out, in)
    ops_each_output = module.weight[0].numel() * 2
    if isinstance(module.bias, Variable):
        ops_each_output += 1
    return ops_each_output * output_data.numel()

def get_data_shape(data):
    if isinstance(data, np.ndarray):
        return data.shape
    elif isinstance(data, (torch.Tensor, torch.autograd.Variable)):
        return tuple(data.size())
    elif isinstance(data, (float, int)):
        return 1,
    else:
        raise TypeError('Unknown data type for shape')

return_zero = lambda x,y,z:0
get_ops_functions = {
    'Conv2d': get_ops_conv2d,
    'ConvTranspose2d': get_ops_convtransposed2d,
    'Conv3d': get_ops_conv2d,
    'ConvTranspose3d': get_ops_convtransposed2d,
    'Linear': get_ops_linear,
    'ReLU': return_zero,
    'ReLU6': return_zero,
    'BatchNorm1d': return_zero,
    'BatchNorm2d': return_zero,
    'BatchNorm3d': return_zero,
    'MaxPool2d': return_zero,
    'Dropout': return_zero,
    'Dropout2d': return_zero,
    }

