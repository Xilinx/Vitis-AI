"""Tests the quantize_graph.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import gfile
from tensorflow.contrib.decent_q.python import utils as q_utils
from tensorflow.contrib.decent_q.python.ops import fix_neuron_ops


class QuantizeGraphTest(test_util.TensorFlowTestCase):
  def _build_graph(self, is_freezed=True, with_bias=True, with_fix_op=False):
    images = array_ops.placeholder(dtypes.float32, shape=[None, 4, 4, 3],
            name="input")
    if with_fix_op:
      images = fix_neuron_ops.fix_neuron(images, 1, 1, name="input/aquant")
    if is_freezed:
      w = constant_op.constant([[[[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]]],
                                dtype=dtypes.float32,
                                shape=[1, 3, 3, 3],
                                name="w")
    else:
      w = variables.VariableV1([[[[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]]],
                                dtype=dtypes.float32,
                                shape=[1, 3, 3, 3],
                                name="w")
    feature = nn.conv2d(images, filter=w, name="conv2d",
            strides=[1, 1, 1, 1], padding="SAME")
    if with_bias:
      if is_freezed:
        b = constant_op.constant([0.1, 0.2, 0.3], dtype=dtypes.float32, shape=[3], name="b")
      else:
        b = variables.VariableV1([0.1, 0.2, 0.3], dtype=dtypes.float32, shape=[3], name="b")
      feature = nn.bias_add(feature, b, name="bias")
    feature = nn.relu(feature, name="relu")
    return

  def testCheckNodeNames(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      graph_def = ops.get_default_graph().as_graph_def()
      q_utils.check_node_names(graph_def, ["conv2d", "bias", "relu"])

  def testGetNodeDtypes(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      graph_def = ops.get_default_graph().as_graph_def()
      conv2d_type = q_utils.get_node_dtypes(graph_def, ["conv2d"])[0]
      self.assertEqual(conv2d_type, "float32")

  def testGetNodeShape(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      graph_def = ops.get_default_graph().as_graph_def()
      target_shapes = q_utils.get_node_shapes(graph_def, ["input"])[0]
      self.assertEqual(target_shapes, [None, 4, 4, 3])

  def testGetQuantizedNodes(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True, with_fix_op=True)
      graph_def = ops.get_default_graph().as_graph_def()
      target_node = q_utils.get_quantized_nodes(graph_def, ["input"])[0]
      self.assertEqual(target_node, "input/aquant")
      target_node = q_utils.get_quantized_nodes(graph_def, ["conv2d"])[0]
      self.assertEqual(target_node, "conv2d")

  def testGetEdgeTensors(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      graph = ops.get_default_graph()
      input_tensors, output_tensors = q_utils.get_edge_tensors(
              graph, ["input"], ["relu"])
      self.assertEqual(type(input_tensors[0]).__name__, "Tensor")
      self.assertEqual(input_tensors[0].name, "input:0")
      self.assertEqual(type(output_tensors[0]).__name__, "Tensor")
      self.assertEqual(output_tensors[0].name, "relu:0")


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  test.main()
