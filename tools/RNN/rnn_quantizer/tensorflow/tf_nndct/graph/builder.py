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

import imp

from tensorflow.python.ops import array_ops
from tensorflow.python import keras
from tensorflow.python.util import nest

from tf_nndct.graph import OpTypes
from tf_nndct.graph import parser
from tf_nndct.graph import utils
from tf_nndct.graph import writer as writer_lib
from tf_nndct.layers import base_layer
from tf_nndct.utils import keras_utils
from tf_nndct.utils import logging
from tf_nndct.utils import tensor_utils

class ClassSpec(object):
  def __init__(self, name, base=keras.Model, call_fn_name='call',
        quantized=False):
    self.name = name
    self.base = base
    self.call_fn_name = call_fn_name
    self.quantized = quantized

class KerasBuilder(object):

  def __init__(self, graph):
    self._graph = graph

  def build(self, filepath, quantized=False, as_layer=False):
    class_name = self._graph.name
    base_class = keras.Model
    call_fn_name = 'call'
    if as_layer:
      if quantized:
        base_class = base_layer.Layer
        call_fn_name = '_internal_call'
      else:
        base_class = keras.layers.Layer
    class_spec = ClassSpec(class_name, base_class, call_fn_name, quantized)
    writer = writer_lib.GraphCodeGenerator(self._graph, class_spec)
    layer_to_node = writer.write(filepath)

    # TODO(yuwang): Use code below.
    #py_module_name = "_".join(["nndct", module_name])
    #spec = importlib.util.spec_from_file_location(py_module_name, filepath)
    #py_module = importlib.util.module_from_spec(spec)
    #sys.modules[py_module_name] = py_module
    #spec.loader.exec_module(py_module)
    #rebuilt_module = py_module.__dict__[module_name]()
    loaded_module = imp.load_source('nndct_rebuilt_model', filepath)
    rebuilt_model = getattr(loaded_module, class_name)()

    dummy_inputs = []
    for spec in nest.flatten(self._graph.input_signature):
      logging.vlog(1, spec)
      dummy_inputs.append(array_ops.ones(spec.shape, dtype=spec.dtype))
    dummy_inputs = nest.pack_sequence_as(self._graph.input_signature,
                                         dummy_inputs)

    #input_data = dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0]
    rebuilt_model(*dummy_inputs)

    layer_nodes = []
    # Reload weights
    for layer_name, node in layer_to_node.items():
      layer = getattr(rebuilt_model, layer_name)
      # If there is a ParamName definition, then map ParamName's member to
      # keras layer's param; If there is no ParamName,
      # then export params in the order they are saved in the op.
      weights = []
      if hasattr(node.op, 'ParamName'):
        params = keras_utils.keras_layer_params(layer)
        for name in params.keys():
          param = utils.op_param_by_name(node.op, name)
          if not param:
            continue
          ndarray = tensor_utils.to_tf_numpy(node.op.get_param(param))
          weights.append(ndarray)
          logging.vlog(2, 'Reload weights of {}: name={}, shape={}'.format(
              layer.name, name, ndarray.shape))
      else:
        for name, tensor in node.op.params.items():
          ndarray = tensor_utils.to_tf_numpy(tensor)
          weights.append(ndarray)
          logging.vlog(2, "Reload weights of {}: name={}, shape={}".format(
              layer.name, name, ndarray.shape))
      layer_nodes.append((layer, node))
      if weights:
        layer.set_weights(weights)
    return rebuilt_model, layer_nodes

def rebuild_model(model, input_signature):
  graph = parser.from_keras_model(model, input_signature)
  builder = KerasBuilder(graph)
  rebuilt_model, layer_names = builder.build('%s_rebuilt.py' % model.name)
  return rebuilt_model
