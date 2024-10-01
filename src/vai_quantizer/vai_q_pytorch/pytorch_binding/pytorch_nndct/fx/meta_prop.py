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

import itertools
import torch
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import ShapeProp


class ValueMetaProp(FakeTensorProp):
  def run_node(self, n):
    result = super().run_node(n)
    found_tensor = False
    def extract_tensor_meta(obj):
      if isinstance(obj, FakeTensor):
          nonlocal found_tensor
          found_tensor = True
          return _extract_tensor_metadata(obj)
      else:
          return obj

    meta = map_aggregate(result, extract_tensor_meta)
    if found_tensor:
        n.meta['tensor_meta'] = meta

    n.meta['type'] = type(result)
    return result

  def propagate(self, *args):
    with self._mode:
      fake_args = [self._mode.from_tensor(a) for a in args]
      return super().run(*fake_args)


def collect_value_meta(gm, example_inputs, mode=None):
  if mode is not None:
    with mode as fake_tensor_mode:

      def to_fake_tensor(x):
          if isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor):
              return fake_tensor_mode.from_tensor(x)
          return x

      fake_parameters_and_buffers = {
          k: to_fake_tensor(v)
          for k, v in itertools.chain(
              gm.named_parameters(), gm.named_buffers()
          )
      }
      with torch.nn.utils.stateless._reparametrize_module(
          gm, fake_parameters_and_buffers
      ):
    
        ValueMetaProp(gm, fake_tensor_mode).propagate(*example_inputs)
  else:
    ShapeProp(gm).propagate(*example_inputs)
 