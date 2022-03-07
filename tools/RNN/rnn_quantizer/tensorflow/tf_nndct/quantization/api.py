

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

import os
import tensorflow as tf

from nndct_shared.base import NNDCT_KEYS, NNDCT_OP, GLOBAL_MAP
from nndct_shared.utils import option_util, NndctOption, NndctScreenLogger

from tf_nndct.graph import OpTypes
from tf_nndct.graph import builder
from tf_nndct.graph import ops
from tf_nndct.graph import parser
from tf_nndct.graph import utils as graph_utils
from tf_nndct.layers import recurrent
from tf_nndct.quantization import TFQuantizer
from tf_nndct.utils import keras_utils
from tf_nndct.utils import tf_utils

from tensorflow.keras import activations

def _init_quant_mode(quant_mode):
  if isinstance(quant_mode, int):
    NndctScreenLogger().warning(f"quant_mode will not support integer value in future version. It supports string values 'calib' and 'test'.")
    qmode = quant_mode
  elif isinstance(quant_mode, str):
    if quant_mode == 'calib':
      qmode = 1
    elif quant_mode == 'test':
      qmode = 2
    else:
      NndctScreenLogger().error(f"quant_mode supported values are 'calib' and 'test'. Change it to 'calib' as calibration mode")
      qmode = 1
  else:
    NndctScreenLogger().error(f"quant_mode supported values are string 'calib' and 'test'. Change it to 'calib' as calibration mode")
    qmode = 1

  if NndctOption.nndct_quant_mode.value > 0:
    qmode = NndctOption.nndct_quant_mode.value

  if qmode == 1:
    NndctScreenLogger().info(f"Quantization calibration process start up...")
  elif qmode == 2:
    NndctScreenLogger().info(f"Quantization test process start up...")

  return qmode

def tf_quantizer(model,
                 input_signature,
                 quant_mode: str = "calib",
                 output_dir: str = "quantize_result",
                 bitwidth: int = 8):
  #initialize quant mode
  qmode = _init_quant_mode(quant_mode)

  # turn off weights equalization and bias correction
  option_util.set_option_value("nndct_param_corr", False)
  option_util.set_option_value("nndct_equalization", False)

  # lstm IP only support 16 bit activation
  quantizer = TFQuantizer(qmode, output_dir, bitwidth, 16)
  GLOBAL_MAP.set_map(NNDCT_KEYS.QUANTIZER, quantizer)
  GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_MODE, qmode)

  graph = parser.from_keras_model(model, input_signature)
  quant_model, layer_nodes = builder.KerasBuilder(graph).build(
      os.path.join(output_dir, model.name + '_quant.py'), quantized=True)

  rebuilding_results = _maybe_rebuild_rnn(quant_model)
  if rebuilding_results:
    cell_graphs = []
    cell_layer_nodes = []
    for graph, layer_nodes in rebuilding_results:
      cell_graphs.append(graph)
      cell_layer_nodes.extend(layer_nodes)
      quantizer.add_rnn_cell_graph('forward', graph)

    graph = _merge_cell_graphs(cell_graphs)
    layer_nodes = cell_layer_nodes
    # TODO(yuwang): Support backward direction.

  export_file = os.path.join(output_dir, 'merged_graph.pb')
  graph_utils.maybe_export_graph(export_file, graph)

  lstm = True if len(rebuilding_results) > 0 else False
  quantizer.setup(graph, lstm=lstm)
  quantizer.load_node_to_layer(layer_nodes, quant_model)

  return quantizer

def export_quant_config():
  quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
  if quantizer and quantizer.quant_mode == 1:
    quantizer.export_quant_config()

def _merge_cell_graphs(cell_graphs):
  def prepend_scope(obj, scope):
    obj.name = '{}/{}'.format(scope, obj.name)

  def rename_input(node):
    # XXX(yuwang): Modify input nodes names to input_0, input_1... to make it
    # easier for the compiler to inditify the inputs. We need to design
    # this more rationally in the future.
    # The node name follow the correspondence rules:
    # input_0 -> inputs
    # input_1 -> H(t-1)
    # input_2 -> C(t-1)

    if node.op.type != OpTypes.INPUT:
      return

    arg_to_input = {
        'args_0': 'input_0',
        'args_1': 'input_1',
        'args_1_1': 'input_2'
    }

    name_parts = node.name.split('/')
    name_parts[-1] = arg_to_input[name_parts[-1]]
    node.name = '/'.join(name_parts)

  graph = ops.Graph()
  for cell_graph in cell_graphs:
    scope = cell_graph.name
    for node in cell_graph.nodes:
      rename_input(node)
      prepend_scope(node, scope)
      for tensor in node.out_tensors:
        prepend_scope(tensor, scope)

      graph.add_node(node)
  return graph

def _maybe_rebuild_rnn(model):
  rebuilding_results = []
  layers = keras_utils.gather_layers(model)
  for layer in layers:
    # TODO(yuwang): Support StackedRNNCells, RNN
    if not isinstance(layer, recurrent.LSTM):
      continue

    cell = layer.cell
    assert cell.recurrent_activation == activations.get('sigmoid'), 'recurrent_activation must be "sigmoid"'
    graph_name = 'rnn_cell_%d' % len(rebuilding_results)
    cell_graph = _parse_rnn_cell(cell)
    cell_graph.name = graph_name
    rebuilt_cell, layer_nodes = builder.KerasBuilder(cell_graph).build(
        os.path.join('quantize_result', graph_name + '.py'), quantized=True)
    rebuilding_results.append((cell_graph, layer_nodes))

    _copy_attr('units', cell, rebuilt_cell)
    _copy_attr('state_size', cell, rebuilt_cell)
    _copy_attr('output_size', cell, rebuilt_cell)
    layer.cell = rebuilt_cell
  return rebuilding_results

def _copy_attr(attr, src, dst):
  """Copy attr from src to dst."""
  setattr(dst, attr, getattr(src, attr))

def _parse_rnn_cell(cell):
  input_dim = cell.kernel_i.shape[0]
  dtype = cell.kernel_i.dtype
  #print("\nconverted_code:\n", tf.autograph.to_code(cell.call))
  #inputs = array_ops.ones(shape=(1, input_dim), dtype=dtype)
  #states = cell.get_initial_state(inputs)
  #tf_graph = parser.get_func_graph(cell, None, inputs, states)

  input_spec = tf.TensorSpec(shape=(1, input_dim), dtype=dtype)
  state_spec = tf.TensorSpec(shape=(1, cell.units), dtype=dtype)
  input_signature = [input_spec, [state_spec] * 2]

  func_graph = parser.get_func_graph(cell, input_signature)
  return parser.func_graph_to_nndct(func_graph)
