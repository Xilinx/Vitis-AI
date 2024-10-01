
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

import copy
import os
import torch
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS, NNDCT_OP
from nndct_shared.utils import (AddXopError, NndctOption, NndctScreenLogger,
                                option_util, QError, QWarning, set_option_value)
from pytorch_nndct.utils.torch_utils import CmpFlag, compare_torch_version
from pytorch_nndct.utils.onnx_utils import get_opset_version
from pytorch_nndct.utils.module_util import visualize_tensors, to_device, get_module_name
from .utils import update_nndct_blob_data
import numpy as np


# VAI QuantizeLinear
class QuantizeLinear(torch.autograd.Function):
  @staticmethod
  def symbolic(g, input, valmin, valmax, scale, zero_point, method):
    return g.op("vai::QuantizeLinear", input, valmin, valmax, scale, zero_point, method).setType(input.type())

  @staticmethod
  def forward(ctx, x, valmin, valmax, scale, zero_point, method):
    input = torch.from_numpy(np.clip((x/scale + zero_point), valmin, valmax))
    output = torch.empty_like(input)
    mth = torch.from_numpy(method)
    NndctRound(input, output, mth)
    return output.numpy() 

# VAI DequantizeLinear
class DequantizeLinear(torch.autograd.Function):
  @staticmethod
  def symbolic(g, input, scale, zero_point):
    return g.op("vai::DequantizeLinear", input, scale, zero_point).setType(input.type())

  @staticmethod
  def forward(ctx, x, scale, zero_point):
    return scale*(x - zero_point)


def export_onnx_model_for_lstm(quantizer, example_inputs, input_tensors_name, return_tensors_name, convert_script_to_qscript, output_dir, 
        verbose=False, dynamic_batch=False, opset_version=None, native_onnx=True, dump_layers=False, check_model=False, opt_graph=False):
  from torch.onnx import register_custom_op_symbolic
  from torch.onnx.symbolic_helper import parse_args

  @parse_args("v", "v", "v", "v", "v", "v", "v", "v")
  def symbolic_fix_neuron(g, input, valmin, valmax, valamp, zero_point, method, device_id, inplace):
    return g.op("vai::fix_neuron", input, valmax, valamp, method, device_id, inplace).setType(input.type())
  
  register_custom_op_symbolic("vai::fix_neuron", symbolic_fix_neuron, 9)

  opset_version = NndctOption.nndct_onnx_opset_version.value if opset_version is None else opset_version
  opset_version = get_opset_version() if opset_version==-1 else opset_version
  device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)

  script_models = quantizer.scripts if len(quantizer.scripts) > 0 else None
  quantizer.reset_status_for_exporting()
  if script_models is None:
    output_file = os.path.join(output_dir, f"{quantizer.quant_model._get_name()}_int.onnx")
    model, input_args = to_device(quantizer.quant_model, example_inputs, device)
    try:
      torch.onnx.export(quantizer.quant_model, input_args, output_file, 
                        verbose=verbose, 
                        input_names=input_tensors_name,
                        opset_version=opset_version,
                        custom_opsets={'vai' : 2})
    except Exception as e:
      NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(quantizer.quant_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}.")
    else:
      NndctScreenLogger().info(f"{get_module_name(quantizer.quant_model)}_int.onnx is generated.({output_file})")
  else:
    if len(script_models) == 1:
      inputs = [example_inputs]
      inputs_name = [input_tensors_name]
      outputs_name = [return_tensors_name]
    else:
      inputs = example_inputs
      inputs_name = input_tensors_name
      outputs_name = return_tensors_name
    

    for script, inputs, input_name, output_name in zip(script_models, inputs, inputs_name, outputs_name):
      q_model = convert_script_to_qscript(script, verbose=verbose)
      _, inputs = to_device(None, inputs, device)
      output_file =  os.path.join(output_dir, f"{get_module_name(q_model)}_int.onnx")
      try:
        torch.onnx.export(q_model, inputs, output_file, 
                          verbose=verbose, 
                          input_names=input_name,
                          opset_version=opset_version, 
                          custom_opsets={'vai' : 2})
      except Exception as e:
        NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(q_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}.")
      NndctScreenLogger().info(f"{get_module_name(q_model)}_int.onnx is generated.({output_file})")

def export_onnx_runable_model(quantizer, example_inputs, input_tensors_name, return_tensors_name, convert_script_to_qscript, output_dir, 
        verbose=False, dynamic_batch=False, opset_version=None, native_onnx=True, dump_layers=False, check_model=False, opt_graph=False):
  from torch.onnx import register_custom_op_symbolic
  from torch.onnx.symbolic_helper import parse_args
  import sys
  import torch._C._onnx as _C_onnx

  if compare_torch_version(CmpFlag.LESS, "1.7.0"):
    NndctScreenLogger().error2user(QError.TORCH_VERSION, f'Only supprt exporting onnx model with pytorch 1.7 and later version.')
    return
  
  if quantizer.contain_channel_quantize():
    if compare_torch_version(CmpFlag.LESS, "1.10.0"):
      NndctScreenLogger().error2user(QError.TORCH_VERSION, f'Only supprt exporting per_channel quantization onnx model with pytorch 1.10 and later version.')
      return
  
  # per-tensor quantization
  @parse_args("v", "i", "i", "f", "i", "i", "i", "i")
  def symbolic_fix_neuron(g, input, valmin, valmax, valamp, zero_point, method, device_id, inplace):
    if valamp < sys.float_info.min:
      scale = torch.tensor(sys.float_info.max).float()  # Avoid exportor generating double type
    else:
      scale = torch.tensor(1.0 / valamp).float()  # Avoid exportor generating double type
    zero_point = torch.tensor(0, dtype=torch.int8)  # ONNX requires zero_point to be tensor
    if not isinstance(input, torch._C.Value) or not isinstance(valmin, int) or not isinstance(valmax, int) \
            or not isinstance(valamp, float) or zero_point.dtype != torch.int8 or not isinstance(method, int) \
            or not isinstance(device_id, int) or not isinstance(inplace, int) or valamp <= 0.:
      NndctScreenLogger().error2user(QError.FIX_INPUT_TYPE, f'Data type or value illegal fix neuron in when exporting onnx model.')
    
    if compare_torch_version(CmpFlag.GREATER_EQUAL, "2.0.0"):
      NndctOption.nndct_native_onnx.value = True

    if NndctOption.nndct_native_onnx.value:
      return g.op("DequantizeLinear", g.op("QuantizeLinear", input, scale, zero_point), scale, zero_point)
    else:
      return g.op("vai::DequantizeLinear", g.op("vai::QuantizeLinear", input, torch.tensor(valmin, dtype=torch.int32), torch.tensor(valmax, dtype=torch.int32), scale, zero_point, torch.tensor(method, dtype=torch.int8), torch.tensor(1, dtype=torch.int8)), scale, zero_point, torch.tensor(1, dtype=torch.int8))
  
  # per-channel quantization
  @parse_args("v", "i", "i", "v", "v", "i", "i", "i", "i")
  def symbolic_fix_neuron_per_channel(g, input, valmin, valmax, scale, zero_point, axis, method, device_id, inplace):
    if not isinstance(input, torch._C.Value) or not isinstance(valmin, int) or not isinstance(valmax, int) \
       or not isinstance(scale, torch._C.Value) or not isinstance(zero_point, torch._C.Value) or not isinstance(method, int) \
       or not isinstance(device_id, int) or not isinstance(inplace, int):
      NndctScreenLogger().error2user(QError.FIX_INPUT_TYPE, f'Data type or value illegal fix neuron in when exporting onnx model.')
    
    if compare_torch_version(CmpFlag.GREATER_EQUAL, "2.0.0"):
      NndctOption.nndct_native_onnx.value = True

    if NndctOption.nndct_native_onnx.value:
      return g.op("DequantizeLinear", g.op("QuantizeLinear", input, scale, zero_point, axis_i=axis), scale, zero_point, axis_i=axis)
    else:
      return g.op("vai::DequantizeLinear", g.op("vai::QuantizeLinear", input, torch.tensor(valmin, dtype=torch.int32), torch.tensor(valmax, dtype=torch.int32), scale, zero_point, torch.tensor(method, dtype=torch.int8), torch.tensor(axis, dtype=torch.int8)), scale, zero_point, torch.tensor(axis, dtype=torch.int8))

  NndctOption.nndct_native_onnx.value = native_onnx # use global value in the following
  register_custom_op_symbolic("vai::fix_neuron", symbolic_fix_neuron, 9)
  register_custom_op_symbolic("vai::fix_neuron_per_channel", symbolic_fix_neuron_per_channel, 10)
  opset_version = NndctOption.nndct_onnx_opset_version.value if opset_version is None else opset_version
  opset_version = get_opset_version() if opset_version==-1 else opset_version
  device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
  script_models = quantizer.scripts if len(quantizer.scripts) > 0 else None
  quantizer.reset_status_for_exporting()
  if dynamic_batch: # dynamic batch N in [N, C, L, H, W]
    dynamic_axes = {}
    for i in range(len(input_tensors_name)):
      dynamic_axes[input_tensors_name[i]] = [0]
  else:
    dynamic_axes = None

  # check onnx: backup batchnorm eps
  batchnorm_eps_bak = {}
  if check_model:
    for module in quantizer.quant_model.modules():
      if hasattr(module, "node") and module.node.op.type == NNDCT_OP.BATCH_NORM:
        batchnorm_eps_bak[module.name] = module.eps
        module.eps = 1.0e-5

  if script_models is None:
    
    output_file = os.path.join(output_dir, f"{quantizer.quant_model._get_name()}_int.onnx")
    model, input_args = to_device(quantizer.quant_model, example_inputs, device)
    try:
      torch.onnx.export(quantizer.quant_model.eval(), input_args, output_file, 
                        input_names=input_tensors_name,
                        verbose=verbose, 
                        opset_version=opset_version,
                        dynamic_axes=dynamic_axes,
                        do_constant_folding=False)
    except Exception as e:
       NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(quantizer.quant_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}")
    else:
      NndctScreenLogger().info(f"{get_module_name(quantizer.quant_model)}_int.onnx is generated.({output_file})")

    # optimize onnx graph
    if opt_graph:
      optimize_onnx_graph(output_file)

    # dump layers
    if dump_layers:
      dump_onnx_layers(quantizer, output_dir, output_file, native_onnx, check_model) # custom ops

  else:
    if len(script_models) == 1:
      inputs = [example_inputs]
      inputs_name = [input_tensors_name]
      outputs_name = [return_tensors_name]
    else:
      inputs = example_inputs
      inputs_name = input_tensors_name
      outputs_name = return_tensors_name
    
    for script, inputs, input_name, output_name in zip(script_models, inputs, inputs_name, outputs_name):
      q_model = convert_script_to_qscript(script, verbose=verbose)
      _, inputs = to_device(None, inputs, device)
      output_file =  os.path.join(output_dir, f"{get_module_name(q_model)}_int.onnx")
      try:
        torch.onnx.export(q_model, inputs, output_file, 
                          input_names=input_name,
                          verbose=verbose, 
                          opset_version=opset_version,
                          dynamic_axes=dynamic_axes)
      except Exception as e:
        NndctScreenLogger().error2user(QError.EXPORT_ONNX, f"The {get_module_name(q_model)} can not be exported for onnx. The PyTorch internal failed reason is:\n{str(e)}.")
      else:
        NndctScreenLogger().info(f"{get_module_name(q_model)}_int.onnx is generated.({output_file}).")

      # optimize onnx graph
      if opt_graph:
        optimize_onnx_graph(output_file)

      # check model or dump layers
      if NndctOption.nndct_native_onnx.value: # native quant-dequant mode
        return
      elif dump_layers: # dump onnx layers
        dump_onnx_layers(quantizer, output_dir, output_file, native_onnx, check_model)

  # restore the value of batchnorm eps for quantizer.quant_model
  if len(batchnorm_eps_bak) > 0:
    for module in quantizer.quant_model.modules():
      if module.name in batchnorm_eps_bak.keys():
         module.eps = batchnorm_eps_bak[module.name]

def optimize_onnx_graph(onnx_path):
  import onnx
  from onnxsim import simplify
  model_org = onnx.load(onnx_path)

  # simplify onnx graph
  model_simp, check = simplify(model_org)
  if check:
    NndctScreenLogger().info(f"Optimize onnx graph successfully.")
    onnx.save(model_simp, onnx_path)
  else:
    NndctScreenLogger().info(f"Optimize onnx graph failed.")

# output intermediate layers in onnx model
def dump_onnx_layers(quantizer, output_dir, onnx_model, native_onnx, check_model):
  import onnx
  import os
  
  # Check that the model is well formed
  onnx.checker.check_model(onnx_model)

  # input data
  inputs = get_blob_input(quantizer) # numpy.ndarray
  if inputs is None:
    NndctScreenLogger().warning(f"ONNX model layers are not dumped since input data is None!")
    return

  # get model inputs and ouput names of each layer
  ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
  ort_inputs = {}
  if native_onnx:
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_model, providers=ep_list)
    org_outputs = [x for x in ort_session.get_outputs()]
    
    for i in range(len(inputs)):
      ort_inputs[ort_session.get_inputs()[i].name] = inputs[i]
  else:
    from onnxruntime_extensions import PyOrtFunction
    from pytorch_nndct.apis import load_vai_ops
    ort_session = PyOrtFunction.from_model(onnx_model)
    org_outputs = [x for x in ort_session.output_names]
    for i in range(len(inputs)):
      ort_inputs[ort_session.input_names[i]] = inputs[i]

  model = onnx.load(onnx_model)
  for node in model.graph.node:
    for output in node.output:
      if output not in org_outputs:
        model.graph.output.extend([onnx.ValueInfoProto(name=output)])
  output_names = [x.name for x in model.graph.output] # all layer names

  # save tmp model for dumping layers
  onnx_model_tmp = output_dir + f"{quantizer.quant_model._get_name()}_int_tmp.onnx" 
  onnx.save(model, onnx_model_tmp)

  # get output of each layer
  if native_onnx:
    run_ort = ort.InferenceSession(onnx_model_tmp, providers=ep_list)
    ort_outs = run_ort.run(output_names, ort_inputs)
  else: # vai QuantizeLinear/DequantizeLinear
    # setup vai::QuantizeLinear and vai::DequantizeLinear
    load_vai_ops()

    # load tmp model: onnxruntime_extensions PyOrtFunction
    run_ort = PyOrtFunction.from_model(onnx_model_tmp)

    # onnxruntime_extensions outputs
    run_ort._ensure_ort_session()
    ort_outs = run_ort.ort_session.run(output_names, ort_inputs)

  # dump layers
  layer_dir = output_dir + "/onnx_layer_output/"
  if not os.path.exists(layer_dir):
    os.mkdir(layer_dir)
  debug = False
  for i in range(len(output_names)):
    layer_name = output_names[i]
    layer_data = ort_outs[i].astype("float32") # numpy
    if debug: # permute, format txt
      layer_data = permute(layer_data)
      dump_path = layer_dir + layer_name + '.txt'
      if not os.path.exists(os.path.dirname(dump_path)):
          os.makedirs(os.path.dirname(dump_path))
      layer_data.flatten().tofile(dump_path, sep='\n', format="%.6f")
    else:
      dump_path = layer_dir + layer_name + '.bin'
      if not os.path.exists(os.path.dirname(dump_path)):
          os.makedirs(os.path.dirname(dump_path))
      layer_data.flatten().tofile(dump_path)
  os.remove(onnx_model_tmp) # remove tmp model
  NndctScreenLogger().info(f"ONNX model layers are dumped successfully.({layer_dir})")

  # check ort accuracy
  if not native_onnx and check_model:
    check_dump_onnx(quantizer, ort_outs, inputs)

def get_blob_input(quantizer):
  # update blob data
  update_nndct_blob_data(quantizer.quant_model, quantizer.graph, only_update_shape=False)
  # get input data
  input_dic = {}
  input_names = []
  for node in quantizer.graph.all_nodes():
    if node.op.type == NNDCT_OP.INPUT:
      input_idx = int(node.name.split('_')[-1]) # input_0, input_1
      input_data = np.expand_dims(node.out_tensors[0].data[0], axis=0) # numpy
      input_dic[input_idx] = input_data
      input_names.append(input_idx)
  input_names.sort() # sort: input data must be in sequence of input_0, input_1, input_2, ...
  # get input data
  inputs = [input_dic[idx] for idx in input_names]
  inputs = tuple(inputs)
  return inputs

def check_dump_onnx(quantizer, ort_outs, input): # input numpy
  rtol = 1e-4
  atol = 1e-5
  # nndct result
  device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
  input_torch = []
  for i in range(len(input)):
    input_torch.append(torch.from_numpy(input[i]).to(device))
  quant_out = quantizer.quant_model(input_torch)

  # compare ort_outs with nndct result
  if isinstance(quant_out, (tuple, list)):
    for i in range(len(quant_out)):
      np.testing.assert_allclose(quant_out[i].detach().cpu().numpy(), ort_outs[i], rtol=rtol, atol=atol)
    NndctScreenLogger().info(f"ONNX model is checked valid.")
  elif isinstance(quant_out, torch.Tensor):
    np.testing.assert_allclose(quant_out.detach().cpu().numpy(), ort_outs[0], rtol=rtol, atol=atol)
    NndctScreenLogger().info(f"ONNX model is checked valid.")
  else:
    NndctScreenLogger().warning(f"ONNX model is checked failed: quant output is not tensor, tuple or list!")

# permute data, eg. (N,C,H,W) -> (N,H,W,C)
def permute(input):
  dim = input.ndim
  if dim == 3:
    output = input.transpose(0, 2, 1)
  elif dim == 4:
    output = input.transpose(0, 2, 3, 1)
  elif dim == 5:
    output = input.transpose(0, 2, 3, 4, 1)
  else:
    output = input
  return output

def round(x, method):
  if method == 2: # half_up
    y = copy.deepcopy(x)
    y = np.where(y - np.floor(y) == 0.5, np.ceil(y), np.round(y))
  elif method == 3: # c++ std::round: negative half_down, positive half_up 
    y = copy.deepcopy(x)
    y = np.where(y < 0, np.ceil(y - 0.5), np.floor(y + 0.5))
  elif method == 4: # floor
    y = np.floor(x)
  elif method == 5: # negative half_up, positive half_even
    y = copy.deepcopy(x)
    y = np.where((y < 0) & (y - np.floor(y) == 0.5), np.ceil(y), np.round(y))
  elif method == 6: # negative half_up, positive half_down (vs method 3)
    y = copy.deepcopy(x)
    y = np.where((y < 0) & (y - np.floor(y) == 0.5), np.ceil(y), y)
    y = np.where((y > 0) & (y - np.floor(y) == 0.5), np.floor(y), np.round(y))
  elif method == 7: # up 
    y = np.ceil(x)
  else: # half_even
    y = np.round(x)
  return y

