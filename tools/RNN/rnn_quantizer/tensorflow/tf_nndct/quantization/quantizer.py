

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
import numpy as np
import os
import tensorflow as tf

from scipy import stats

from nndct_shared import quantization as quant_util
from nndct_shared import utils as nndct_utils
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.compile import CompilerFactory
from nndct_shared.compile import DeployChecker
from nndct_shared.quantization import BaseQuantizer
from nndct_shared.utils import NndctOption, NndctScreenLogger

from tf_nndct.graph import OpTypes
from tf_nndct.quantization.ops import diffs_fix_pos
from tf_nndct.quantization.ops import fix_neuron
from tf_nndct.utils import generic_utils
from tf_nndct.utils import tensor_utils

class TFQuantizer(BaseQuantizer):

  def __init__(self, quant_mode: int, output_dir: str, bitwidth_w: int,
               bitwidth_a: int):
    super().__init__(quant_mode, output_dir, bitwidth_w, bitwidth_a)
    self._quant_model = None
    self._output_dir = output_dir
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)

    self._layer_nodes = None
    self._node_to_layer = {}
    self._cell_graphs = []

    self._dump_input = False
    self._quantized_input = {}

  def get_model_type(self):
    return FrameworkType.TENSORFLOW

  def create_fp_tensor(self, name, fp_name, tensor_type='output'):
    init_val = 0.
    if self.quant_mode == 2:
      fp = self.get_bnfp(fp_name, False, tensor_type)
      init_val = fp[1]
      #print('---- Get fix pos of {} = {}'.format(fp_name, init_val), flush=True)
    return tf.Variable(
        init_val, name=name, dtype=tf.float32, shape=None, trainable=False)

  def get_fp_and_quantize(
      self,
      input_tensor,
      fp_name,
      fp_tensor,
      fp_stat_tensor=None,
      node=None,
      tensor_type='output'):  #'input'|'output'|'param'
    # Forward the graph but not quantize parameter and activation
    if (self.quant_mode < 1 or NndctOption.nndct_quant_off.value):
      return input_tensor
    
    if input_tensor.dtype != tf.float32 and input_tensor.dtype != tf.float64:
      NndctScreenLogger().warning_once(f'The tensor type of  {fp_name} is {str(input_tensor.dtype)}. Only support float32/double quantization.')
      return input_tensor

    # get fixed position
    mth = 3
    if tensor_type != 'param':
      mth = 4
    bnfp = self.get_bnfp(fp_name, False, tensor_type)
    bw = bnfp[0]
    if self.quant_mode == 1:
      # must be in eager mode
      #print('---- Calculating fix pos of {}'.format(fp_name), flush=True)
      fp_tensor.assign(
          diffs_fix_pos(input=input_tensor, bit_width=bw, range=5, method=mth))
      bnfp[1] = (int)(fp_tensor.numpy())
      # limit max fix pos to 12
      bnfp[1] = min(12, bnfp[1])
      # record fix pos of input/output by fp_stat_tensor
      if tensor_type != 'param':
        #fp_tensor.assign(stat_act_pos(fp_tensor,
        #                              fp_stat_tensor))
        self.fp_history[tensor_type][fp_name].append(bnfp[1])
        data = np.array(self.fp_history[tensor_type][fp_name])
        bnfp[1] = stats.mode(data)[0][0]
        bnfp[1] = bnfp[1].astype(np.int32).tolist()
        fp_tensor.assign(bnfp[1])
      bnfp = self.set_bnfp(fp_name, bnfp, tensor_type)

    if self.quant_mode > 0:
      # do quantization for parameter or activation
      tensor = fix_neuron(input_tensor, fp_tensor, bw, method=mth)
      if tensor_type == 'param':
        self.update_param_to_quantized(node, fp_name, tensor.numpy())

      # XXX: Temporary.
      if self._dump_input and tensor_type == 'output' and 'input' in fp_name:
        if fp_name not in self._quantized_input:
          self._quantized_input[fp_name] = []
        self._quantized_input[fp_name].append([tensor.numpy()])

      return tensor
    else:
      return input_tensor

  def load_node_to_layer(self, layer_nodes, quant_model):
    if self.quant_mode < 1:
      return

    self._quant_model = quant_model
    self._layer_nodes = layer_nodes
    for layer, node in layer_nodes:
      self._node_to_layer[node.name] = layer
      if not self.configer.is_node_quantizable(node, lstm=True):
        continue

      params = self.configer.quant_node_params(node)
      #print('---- set quant node {} {} {}'.format(node.op.type, node.name, node.in_nodes))
      #for p in params:
      #print('---- params: {}'.format(p.name))
      layer.node = node
      if node.op.type in [OpTypes.CONV2D, OpTypes.DENSE]:
        _, layer.valid_inputs, layer.valid_output = (
            quant_util.get_flows_and_info(
                self.quant_mode,
                self,
                node_name=node.name,
                params=params,
                inputs=node.in_nodes))
        if node.in_nodes[0] == 'lstm':
          layer.valid_inputs = None
      else:
        layer.quant_info, layer.valid_inputs, layer.valid_output = (
            quant_util.get_flows_and_info(
                self.quant_mode,
                self,
                node_name=node.name,
                params=params,
                inputs=node.in_nodes))

      # layer.params_name = node.op.params
      layer.params_name = [v.name for v in node.op.params.values()]
      layer.quantizer = self

  def export_quant_config(self, export_file=None):
    file_name = export_file or self.export_file
    if isinstance(file_name, str):
      if self.quant_mode in [1, 3]:
        self.organize_quant_pos()
        with open(file_name, 'w') as f:
          f.write(nndct_utils.to_jsonstr(self.quant_config))

  def update_param_to_quantized(self, node, name, value):
    for param_name, tensor in node.op.params.items():
      if tensor.name == name:
        #if node.op.type == NNDCT_OP.CONVTRANSPOSE2D:
        #  if param_name == node.op.ParamName.WEIGHTS:
        #    value = np.copy(value).transpose(1, 0, 2, 3)
        tensor.from_ndarray(value)
        tensor_utils.tf_param_to_nndct(tensor)

  def organize_quant_pos(self):
    # Transfer inplace operation fragpos forward
    # to replace configerComannder in future
    if NndctOption.nndct_quant_off.value:
      return

    # align lstm fix pos with cell output
    if self.lstm:
      output_pos = None
      for node in self.Nndctgraph.nodes:
        if (node.name.split('/')[-1] == 'mul_2'):
          output_pos = self.get_bnfp(node.name, False)
          #print('---- Need align nodes output pos to {}'.format(output_pos[1]))

      for node in self.Nndctgraph.nodes:
        # find the last node of every cell
        if (node.name.split('/')[0] == 'rnn_cell_0' and
            node.name.split('/')[1] == 'lstm_cell_1' and
            node.op.type not in [OpTypes.SIGMOID, OpTypes.TANH]):
          self.set_bnfp(node.name, output_pos)
        elif node.op.type in [OpTypes.SIGMOID, OpTypes.TANH]:
          bnfp = self.get_bnfp(node.name, False)
          bnfp[1] = 15
          self.set_bnfp(node.name, bnfp)

    for node in self.Nndctgraph.nodes:
      # align linear OP bias fix pos with output for lstm
      if self.lstm:
        if node.op.type == OpTypes.DENSE:
          if len(node.op.params.values()) > 1:
            params = [v.name for v in node.op.params.values()]
            bnfp = self.get_bnfp(node.name, False)
            self.set_bnfp(params[1], bnfp, 'param')

  def add_rnn_cell_graph(self, direction, graph):
    self._cell_graphs.append((direction, graph))

  def dump_xmodel(self):
    if self.quant_mode < 2:
      return

    compiler = CompilerFactory.get_compiler("xmodel")
    xmodel_dir = os.path.join(self._output_dir, "xmodel")
    generic_utils.mkdir_if_not_exist(xmodel_dir)

    compile_args = []
    if self.lstm:
      for direction, graph in self._cell_graphs:
        compile_args.append((graph, {'direction': direction}))
    else:
      compile_args.append((graph, {}))

    for graph, attr_kwargs in compile_args:
      for node in graph.nodes:
        # TODO(yuwang): Set out tensor shape in parser.
        # Maybe specify shape for all tensors?
        # Input shape must be specified for xgraph shape inference.
        if node.op.type == OpTypes.INPUT:
          node.out_tensors[0].shape = node.op.attr['shape']
      try:
        compiler.do_compile(
            graph,
            os.path.join(xmodel_dir, graph.name),
            quant_config_info=self.quant_config,
            graph_attr_kwargs=attr_kwargs)
      except Exception as e:
        print('[ERROR] Failed to dump xmodel: {}'.format(e))
        return

    print('[INFO] Successfully convert nndct graph to xmodel!')

  def dump_rnn_outputs_by_timestep(self, inputs):
    self._dump_input = True

    for _, graph in self._cell_graphs:
      for node in graph.nodes:
        self._node_to_layer[node.name].enable_saving_outputs()

    self._quant_model(inputs)

    # [step0, step1, ...]
    outputs_at_all_steps = []
    for _, graph in self._cell_graphs:
      for node in graph.nodes:
        outputs = self._node_to_layer[node.name].saved_outputs()
        print('[INFO] saved outputs of {}: steps={}, shape={}'.format(
            node.name, len(outputs), [output.shape for output in outputs[0]]))

        # Ignore step 0
        for step, output in enumerate(outputs[1:]):
          if step >= len(outputs_at_all_steps):
            outputs_at_all_steps.append({})
          outputs_at_all_steps[step][node.name] = output

    def dump_node_outputs(graph, node_outputs, sub_dir):
      checker = DeployChecker(
          output_dir_name=self._output_dir, data_format="txt")
      checker.update_dump_folder(sub_dir)

      for node in graph.nodes:
        outputs = node_outputs[node.name]
        assert len(node.out_tensors) == len(outputs)
        for index, data in enumerate(outputs):
          node.out_tensors[index].from_ndarray(data)
          tensor_utils.tf_blob_to_nndct(node.out_tensors[index])

      enable_dump_weight = True
      checker.dump_nodes_output(
          graph,
          self.quant_config,
          round_method=self.quant_opt['round_method'],
          enable_dump_weight=enable_dump_weight)

    for _, graph in self._cell_graphs:
      print('[INFO] Dumping %s...' % graph.name)
      for step, node_outputs in enumerate(outputs_at_all_steps):
        sub_dir = '{}/frame_{}'.format(graph.name, step)
        dump_node_outputs(graph, node_outputs, sub_dir)
    print('[INFO] Dumping finished.')

  @property
  def quant_model(self):
    return self._quant_model
