# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for transforms.py API code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

LayerNode = transforms.LayerNode


class LayerNodeTest(tf.test.TestCase):

  def testEqualityLayerNode(self):
    conv_layer = {
        'name': 'conv2d',
        'class_name': 'Conv2D',
        'config': {
            'name': 'conv2d',
        }
    }
    dense_layer = {
        'name': 'dense',
        'class_name': 'Dense',
        'config': {
            'name': 'dense',
        }
    }

    self.assertNotEqual(LayerNode(conv_layer), LayerNode(dense_layer))

    self.assertEqual(LayerNode(conv_layer), LayerNode(conv_layer))
    self.assertEqual(
        LayerNode(conv_layer), LayerNode(copy.deepcopy(conv_layer)))

    self.assertNotEqual(
        LayerNode(conv_layer,
                  input_layers=[LayerNode(conv_layer), LayerNode(dense_layer)]),
        LayerNode(conv_layer,
                  input_layers=[LayerNode(conv_layer), LayerNode(conv_layer)]))

    self.assertEqual(
        LayerNode(conv_layer,
                  input_layers=[LayerNode(conv_layer), LayerNode(dense_layer)]),
        LayerNode(conv_layer,
                  input_layers=[LayerNode(conv_layer), LayerNode(dense_layer)]))


if __name__ == '__main__':
  tf.test.main()
