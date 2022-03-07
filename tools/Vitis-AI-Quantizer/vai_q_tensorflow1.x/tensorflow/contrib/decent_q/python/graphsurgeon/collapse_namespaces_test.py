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

"""Tests the collapse_namespaces.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
from tensorflow.contrib.decent_q.python.graphsurgeon import *


class GraphSurgeonTest(test_util.TensorFlowTestCase):
  def _build_graph(self, is_freezed=True):
    images = array_ops.placeholder(dtypes.float32, shape=[None, 4, 4, 3],
            name="input")
    w = constant_op.constant([[[[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                              [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                              [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]]],
                              dtype=dtypes.float32,
                              shape=[1, 3, 3, 3],
                              name="Conv/w")
    feature = nn.conv2d(images, filter=w, name="Conv/conv2d_1",
            strides=[1, 1, 1, 1], padding="SAME")
    b = constant_op.constant([0.1, 0.2, 0.3], dtype=dtypes.float32,
            shape=[3], name="Conv/b_1")
    feature = nn.bias_add(feature, b, name="Conv/bias_1")
    feature = nn.relu(feature, name="relu")
    return


  def testCollapseNamespace(self):
    def get_namespaces_map():
      # collapsed_op = gs.create_node(name="test", op="TestOp")
      collapsed_op = create_node(name="test", op="TestOp")
      nm = {"Conv": collapsed_op}
      return nm

    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      input_graph_def = ops.get_default_graph().as_graph_def()
    nm = get_namespaces_map()

    output_graph_def = collapse_namespaces(input_graph_def, nm)
    node_names = ["input", "test", "relu"]
    for node in output_graph_def.node:
      # print(node.name)
      self.assertIn(node.name, node_names)
      node_names.remove(node.name)



if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  test.main()
