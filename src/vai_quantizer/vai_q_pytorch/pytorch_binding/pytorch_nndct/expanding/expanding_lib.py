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

from torch import nn
from typing import Any, List, Mapping
import os
import torch
import torch.onnx

from nndct_shared.expanding.spec import ExpandingSpec
from nndct_shared.pruning import logging

from pytorch_nndct.expanding.structured import ExpandingRunner
from pytorch_nndct.utils import profiler

def expand_and_export(model_name: str,
                      model: nn.Module,
                      input_signature: torch.Tensor,
                      channel_divisibles: List[int],
                      output_dir: str,
                      onnx_export_kwargs: Mapping[str, Any] = {},
                      export_fp16_model=True,
                      exclude_nodes: List[str] = []) -> None:

  expanding_runner = ExpandingRunner(model, input_signature)
  for channel_divisible in channel_divisibles:
    dir_path = os.path.join(output_dir,
                            model_name + "_padded_{}".format(channel_divisible))
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    expanded_model, expanding_spec = expanding_runner.expand(
        channel_divisible, exclude_nodes)
    expanded_model.eval()
    torch.save(expanded_model.state_dict(),
               os.path.join(dir_path, model_name + ".pth"))
    with open(os.path.join(dir_path, "expanding_spec"), 'w') as f:
      f.write(expanding_spec.serialize())

    torch.onnx.export(
        expanded_model,
        input_signature,
        os.path.join(dir_path, model_name + "_fp32.onnx"),
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        **onnx_export_kwargs)
    if export_fp16_model:
      expanded_model = expanded_model.cuda().half().eval()
      torch.onnx.export(
          expanded_model,
          input_signature.cuda().half(),
          os.path.join(dir_path, model_name + "_fp16.onnx"),
          export_params=True,
          opset_version=10,
          do_constant_folding=True,
          **onnx_export_kwargs)

def load_expanded_model(expanding_spec_path: str, model: nn.Module,
                        input_signature: torch.Tensor,
                        model_path: str) -> nn.Module:
  with open(expanding_spec_path) as f:
    expanding_spec = ExpandingSpec.from_string(f.read())
  expanding_runner = ExpandingRunner(model, input_signature)
  expanded_model = expanding_runner.expand_from_spec(expanding_spec)
  expanded_model.load_state_dict(torch.load(model_path))
  return expanded_model

def summary_before_and_after_padding(model: nn.Module,
                                     input_signature: torch.Tensor,
                                     channel_divisible: int) -> None:
  logging.info("before padding:")
  profiler.model_complexity(
      model, input_signature, readable=True, print_model_analysis=True)
  expanding_runner = ExpandingRunner(model.cpu(), input_signature)
  expanded_model, _ = expanding_runner.expand(channel_divisible)
  logging.info("after padding:")
  profiler.model_complexity(
      expanded_model, input_signature, readable=True, print_model_analysis=True)
