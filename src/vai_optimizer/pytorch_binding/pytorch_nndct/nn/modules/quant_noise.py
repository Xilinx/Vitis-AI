

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
import math

def eval_qnoise(output, res_f, efficency, deviation, rate, stop):
  error = torch.add(output, res_f, alpha=-1).data
  noise = error.pow(2).mean()
  if noise > 0:
    eff = 1.25 * res_f.pow(2).mean().div(noise).log10().detach().cpu().numpy()
    dev = math.fabs(eff - efficency)
    if dev > 0:
      efficency = (efficency * 4 + eff) * 0.2
      deviation = (deviation * 4 + dev) * 0.2
      #print(node.name, efficency, deviation)
      if efficency > 4.0:
        rate = rate * 0.5
      if (efficency > 4.3 or
          (deviation / efficency) < 0.05 or
          math.fabs(dev - deviation / dev) < 0.05):
        stop = True
    else:
      stop = True
  else:
    stop = True
  
  return error, rate, stop, efficency, deviation

