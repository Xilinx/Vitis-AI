
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


from nndct_shared.utils import set_option_value, NndctOption
from pytorch_nndct.qproc.utils import (get_deploy_graph_list,
                                       prepare_quantizable_module,
                                       register_output_hook,
                                       set_outputs_recorder_status, to_device)


def prepare_deployable_graph(module, input_args, device, output_dir):
    module, input_args = to_device(module, input_args, device)
    quant_module, graph = prepare_quantizable_module(
        module=module,
        input_args=input_args,
        export_folder=output_dir,
        device=device)
    
    if len(graph.all_blocks()) > 1:
      quant_module.from_script(True)
    else:
      quant_module.from_script(False)

    quant_off_stat = NndctOption.nndct_quant_off.value
    param_corr_stat = NndctOption.nndct_param_corr.value
    set_option_value("nndct_quant_off", True)
    set_option_value("nndct_param_corr", False)
    register_output_hook(quant_module, record_once=True)
    set_outputs_recorder_status(quant_module, True)
    quant_module.eval()
    if isinstance(input_args, tuple):
      _ = quant_module.to(device)(*input_args)
    else:
      _ = quant_module.to(device)(input_args)
    deploy_graphs, dev_graph = get_deploy_graph_list(quant_module, graph, need_partition=False)
    set_option_value("nndct_quant_off", quant_off_stat)
    set_option_value("nndct_param_corr", param_corr_stat)
    return dev_graph, deploy_graphs


def assign_device_to_node(node_device_map):
  for node, device in node_device_map.items():
    node.target_device = device
