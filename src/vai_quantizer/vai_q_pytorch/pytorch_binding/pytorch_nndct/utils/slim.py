# Copyright 2022 Xilinx Inc.
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

from torch import nn

def load_state_dict(model, state_dict):
  """Update the model so that the shape of the parameters match the weights
  in state dict, and then load the state dict to the updated model.
  """

  # The device of newly created nn.Parameter will be set to the data device
  # in `state_dict`, so we get the original device and reset the model
  # to this device after loading weights.
  device = next(model.parameters()).device

  for key, module in model.named_modules():
    weight_key = key + '.weight'
    bias_key = key + '.bias'
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
      module.weight = nn.Parameter(state_dict[weight_key])
      module.bias = nn.Parameter(state_dict[bias_key])
      module.running_mean = state_dict[key + '.running_mean']
      module.running_var = state_dict[key + '.running_var']
      module.num_features = module.weight.size(0)
    elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
      assert module.groups == 1
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.out_channels = module.weight.size(0)
      module.in_channels = module.weight.size(1)
    elif isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
      assert module.groups == 1
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.in_channels = module.weight.size(0)
      module.out_channels = module.weight.size(1)
    elif isinstance(module, nn.Linear):
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.out_features = module.weight.size(0)
      module.in_features = module.weight.size(1)
    else:
      pass
  model.load_state_dict(state_dict)
  model.to(device)
  return model
