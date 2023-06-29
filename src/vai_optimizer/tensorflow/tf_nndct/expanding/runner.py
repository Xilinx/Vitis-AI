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

import copy
import os
import shutil
import tensorflow as tf

from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from typing import List, Tuple

from nndct_shared.expanding.expander import ChannelExpander
from nndct_shared.expanding.spec import ExpandableGroup, ExpandingSpec
from nndct_shared.expanding.spec import ExpandingSpec
from nndct_shared.pruning import pruning_lib
from nndct_shared.pruning.pruning_lib import group_nodes
from nndct_shared.pruning import errors
from tf_nndct.graph import parser
from tf_nndct.utils import keras_utils as ku
from tf_nndct.utils import tensor_utils

keras = tf.keras

class ExpandingRunner(object):

  def __init__(self, model: keras.Model,
               input_signature: tf.TensorSpec) -> None:
    if not isinstance(model, keras.Model):
      raise ValueError('"model" must be an instance of keras.Model')

    # Check whether the model is a subclass model.
    if (not model._is_graph_network and
        not isinstance(model, keras.models.Sequential)):
      raise ValueError('Subclassed models are not supported currently.')

    self._model = model
    self._input_signature = input_signature

    self._graph = parser.from_keras_model(model, input_signature)

  def expand_from_spec(self, expanding_spec: ExpandingSpec) -> keras.Model:
    expander = ChannelExpander(self._graph)
    expanded_graph, node_expand_desc = expander.expand(expanding_spec)

    def create_from_config(layer):
      if isinstance(layer, keras.Model):
        return keras.models.clone_model(
            layer, input_tensors=None, clone_function=create_from_config)

      if layer.name in layer_config:
        config = layer_config[layer.name]
      else:
        config = layer.get_config()
      return layer.__class__.from_config(config)

    layer_config, layer_to_node = {}, {}
    for node in expanded_graph.nodes:
      node_expanding = node_expand_desc[node.name]

      if node.op.type not in pruning_lib.OPS_WITH_PARAMETERS:
        if len(list(node.op.params)) != 0:
          raise errors.OptimizerUnSupportedOpError(
              'Unsupported op with parameters: {}({})'.format(
                  node.name, node.op.type))
        continue

      if not node.layer_name:
        raise errors.OptimizerNodeError(
            'Can not get the layer corresponding to the node "{}"'.format(
                node.name))
      layer_to_node[node.layer_name] = node

      if node_expanding.added_in_channel == 0 and node_expanding.added_out_channel == 0:
        continue

      config = {}
      for name in node.op.configs:
        config[name] = node.op.get_config(name)
      # Parser reset activation to None in op's config and saved the original
      # value to op's attribute. Here we use this attribute value to build
      # the config for recreating model.
      if node.op.has_attr('activation'):
        config['activation'] = node.op.attr['activation']
      layer_config[node.layer_name] = config

    strategy = self._model._distribution_strategy or ds_context.get_strategy()
    with strategy.scope():
      model = create_from_config(self._model)
      for layer in ku.gather_layers(model):
        if not layer.weights:
          continue
        weights = tensor_utils.layer_weights_from_node(
            layer_to_node[layer.name])
        layer.set_weights(weights)
    return model

  def expand(
      self,
      channel_divisible: int = 2,
      nodes_to_exclude: List[str] = []) -> Tuple[keras.Model, ExpandingSpec]:
    groups: List[List[str]] = [
        g.nodes for g in group_nodes(
            self._graph, nodes_to_exclude, with_group_conv=False)
    ]
    expanding_spec = ExpandingSpec()
    for group in groups:
      expanding_spec.add_group(ExpandableGroup(group, channel_divisible))
    return self.expand_from_spec(expanding_spec), expanding_spec

def export_model_as_pb(model: keras.Model, output_dir: str, output_name: str):
  full_model = tf.function(lambda Input: model(Input))
  full_model = full_model.get_concrete_function(
      tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
  frozen_func = convert_variables_to_constants_v2(full_model)
  frozen_func.graph.as_graph_def()
  tf.io.write_graph(
      graph_or_graph_def=frozen_func.graph,
      logdir=output_dir,
      name=output_name,
      as_text=False)

def load_graph_def(graph_def_path):
  with tf.compat.v1.gfile.FastGFile(graph_def_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def load_and_extract_subgraph(org_model_path):
  graph_def = load_graph_def(org_model_path)
  id_node = graph_def.node[-1]
  try:
    id_node.input.remove("^NoOp")
  except:
    pass
  softmax_node = id_node.input[0]
  # import pdb; pdb.set_trace()
  graph_def = tf.compat.v1.graph_util.extract_sub_graph(graph_def,
                                                        [softmax_node])
  graph_def.node.extend([id_node])
  return graph_def

def convert_pb_to_fp16_pb(model_path: str,
                          save_path: str,
                          input_nodes: List[str],
                          output_nodes: List[str],
                          as_text=False):
  import vai_q_tensorflow as decent_q
  tmp_dir = "/tmp/tf_nndct/padding/"
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  source_graph_def = load_and_extract_subgraph(model_path)
  config = decent_q.QuantizeConfig(
      input_nodes=input_nodes,
      output_nodes=output_nodes,
      weight_bit=8,
      activation_bit=8,
      method=1,
      simulate_dpu=0,
      replace_relu6=0,
      output_dir=tmp_dir)

  decent_q.convert_datatype(source_graph_def, config, 1)

  model = load_graph_def(os.path.join(tmp_dir, "converted_model_fp16.pb"))
  for node in model.node:  # replace Add to BiasAdd to adapt for AMD MiGraphX
    if node.op == 'Add':  #and not "conv_dw" in node.name:
      node.op = "BiasAdd"
  tf.io.write_graph(
      graph_or_graph_def=model,
      logdir=os.path.dirname(save_path),
      name=os.path.basename(save_path),
      as_text=False)
  os.remove(os.path.join(tmp_dir, "converted_model_fp16.pb"))

def expand_and_export(model_name: str,
                      model: keras.Model,
                      input_signature: tf.TensorSpec,
                      output_dir: str,
                      channel_divisibles: List[int],
                      input_nodes: List[str],
                      output_nodes: List[str],
                      export_fp16_model=True) -> None:

  expanding_runner = ExpandingRunner(model, input_signature)
  for channel_divisible in channel_divisibles:
    dir_path = os.path.join(output_dir,
                            model_name + "_padded_{}".format(channel_divisible))
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    expanded_model, expanding_spec = expanding_runner.expand(channel_divisible)
    with open(os.path.join(dir_path, "expanding_spec"), 'w') as f:
      f.write(expanding_spec.serialize())
    export_model_as_pb(expanded_model, dir_path, model_name + "_fp32.pb")
    if export_fp16_model:
      convert_pb_to_fp16_pb(
          os.path.join(dir_path, model_name + "_fp32.pb"),
          os.path.join(dir_path, model_name + "_fp16.pb"), input_nodes,
          output_nodes)
