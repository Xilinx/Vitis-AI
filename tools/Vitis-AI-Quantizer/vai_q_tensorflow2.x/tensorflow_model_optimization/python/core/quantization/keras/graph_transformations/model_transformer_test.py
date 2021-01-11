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
"""Tests for Model Transformation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

ModelTransformer = model_transformer.ModelTransformer
Transform = transforms.Transform
LayerPattern = transforms.LayerPattern
LayerNode = transforms.LayerNode

keras = tf.keras


class ModelTransformerTest(tf.test.TestCase, parameterized.TestCase):

  @staticmethod
  def _batch(dims, batch_size):
    """Adds provided batch_size to existing dims.

    If dims is (None, 5, 2), returns (batch_size, 5, 2)

    Args:
      dims: Dimensions
      batch_size: batch_size

    Returns:
      dims with batch_size added as first parameter of list.
    """
    if dims[0] is None:
      dims[0] = batch_size
    return dims

  @staticmethod
  def _get_layer(model, n_excluding_input, model_type):
    # Argument n excludes the input layer.
    if model_type == 'functional':
      return model.layers[n_excluding_input + 1]
    else:
      return model.layers[n_excluding_input]

  def _create_model_inputs(self, model):
    return np.random.randn(*self._batch(model.input.get_shape().as_list(), 1))

  def _simple_dense_model(self, model_type='functional'):
    if model_type == 'functional':
      inp = keras.layers.Input((3,))
      x = keras.layers.Dense(2)(inp)
      out = keras.layers.ReLU(6.0)(x)
      return keras.Model(inp, out)
    elif model_type == 'sequential':
      return keras.Sequential(
          [keras.layers.Dense(2, input_shape=(3,)),
           keras.layers.ReLU(6.0)])

  def _assert_config(self, expected_config, actual_config, exclude_keys=None):
    """Asserts that the two config dictionaries are equal.

    This method is used to compare keras Model and Layer configs. It provides
    the ability to exclude the keys we don't want compared.

    Args:
      expected_config: Config which we expect.
      actual_config: Actual received config.
      exclude_keys: List of keys to not check against.
    """
    expected_config = expected_config.copy()
    actual_config = actual_config.copy()

    def _remove_keys(config):
      """Removes all exclude_keys (including nested) from the dict."""
      for key in exclude_keys:
        if key in config:
          del config[key]

      for _, v in config.items():
        if isinstance(v, dict):
          _remove_keys(v)

        if isinstance(v, list):
          for item in v:
            if isinstance(item, dict):
              _remove_keys(item)

    if exclude_keys:
      _remove_keys(expected_config)
      _remove_keys(actual_config)

    self.assertDictEqual(expected_config, actual_config)

  def _assert_model_results_equal(self, model, transformed_model):
    inputs = self._create_model_inputs(model)
    self.assertAllClose(
        model.predict(inputs), transformed_model.predict(inputs))

  # Transform classes for testing.

  class ReplaceDenseLayer(transforms.Transform):
    """Replaces `Dense` layers with `MyDense`, a simple inherited layer.

    This `Transform` class replaces `Dense` layers with a class `MyDense`
    which is simply an empty inheritance of `Dense`. This makes it easy to test
    the transformation code.
    """

    class MyDense(keras.layers.Dense):
      pass

    def pattern(self):
      return LayerPattern('Dense')

    def replacement(self, match_layer):
      match_layer_config = match_layer.layer['config']
      my_dense_layer = self.MyDense(**match_layer_config)

      replace_layer = keras.layers.serialize(my_dense_layer)
      replace_layer['name'] = replace_layer['config']['name']

      return LayerNode(replace_layer, match_layer.weights, [])

    def custom_objects(self):
      return {'MyDense': self.MyDense}

  @parameterized.parameters(['sequential', 'functional'])
  def testReplaceSingleLayerWithSingleLayer_OneOccurrence(self, model_type):
    model = self._simple_dense_model(model_type)

    transformed_model, _ = ModelTransformer(
        model, [self.ReplaceDenseLayer()]).transform()

    # build_input_shape is a TensorShape object and the two objects are not
    # considered the same even though the shapes are the same.
    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['class_name', 'build_input_shape'])

    self.assertEqual(
        'MyDense',
        self._get_layer(transformed_model, 0, model_type).__class__.__name__)

    self._assert_model_results_equal(model, transformed_model)

  @parameterized.parameters(['sequential', 'functional'])
  def testReplaceSingleLayerWithSingleLayer_MultipleOccurrences(
      self, model_type):
    # Note that the functional and sequential model architectures
    # are different. The first is a tree and the second is a chain.
    if model_type == 'functional':
      inp = keras.layers.Input((3,))
      x1 = keras.layers.Dense(2)(inp)
      x2 = keras.layers.Dense(2)(inp)
      out1 = keras.layers.ReLU(6.0)(x1)
      out2 = keras.layers.ReLU(6.0)(x2)
      model = keras.Model(inp, [out1, out2])
    else:
      model = keras.Sequential([
          keras.layers.Dense(2, input_shape=(3,)),
          keras.layers.Dense(2),
          keras.layers.ReLU(6.0)
      ])

    transformed_model, _ = ModelTransformer(
        model, [self.ReplaceDenseLayer()]).transform()

    # build_input_shape is a TensorShape object and the two objects are not
    # considered the same even though the shapes are the same.
    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['class_name', 'build_input_shape'])

    self.assertEqual(
        'MyDense',
        self._get_layer(transformed_model, 0, model_type).__class__.__name__)
    self.assertEqual(
        'MyDense',
        self._get_layer(transformed_model, 1, model_type).__class__.__name__)

    self._assert_model_results_equal(model, transformed_model)

  @parameterized.parameters(['sequential', 'functional'])
  def testReplaceSingleLayerWithSingleLayer_MatchParameters(self, model_type):

    class RemoveBiasInDense(transforms.Transform):
      """Replaces Dense layers with matching layers with `use_bias=False`."""

      def pattern(self):
        return LayerPattern('Dense', {'use_bias': True})

      def replacement(self, match_layer):
        match_layer_config = match_layer.layer['config']
        # Remove bias
        match_layer_weights = match_layer.weights
        match_layer_weights.popitem()

        match_layer_config['use_bias'] = False
        new_dense_layer = keras.layers.Dense(**match_layer_config)

        replace_layer = keras.layers.serialize(new_dense_layer)
        replace_layer['name'] = replace_layer['config']['name']

        return LayerNode(replace_layer, match_layer_weights, [])

    model = self._simple_dense_model(model_type)

    transformed_model, _ = ModelTransformer(
        model, [RemoveBiasInDense()]).transform()

    # build_input_shape is a TensorShape object and the two objects are not
    # considered the same even though the shapes are the same.
    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['use_bias', 'build_input_shape'])

    first_noninput_layer = self._get_layer(transformed_model, 0, model_type)
    self.assertFalse(first_noninput_layer.use_bias)

    # Should match since bias is initialized with zeros.
    self._assert_model_results_equal(model, transformed_model)

  @parameterized.parameters(['sequential', 'functional'])
  def testReplaceSingleLayer_WithMultipleLayers(self, model_type):

    class ReplaceDenseWithDenseAndActivation(transforms.Transform):
      """Dense => (Dense -> ReLU)."""

      def pattern(self):
        return LayerPattern('Dense')

      def replacement(self, match_layer):
        activation_layer = keras.layers.Activation('linear')
        layer_config = keras.layers.serialize(activation_layer)
        layer_config['name'] = activation_layer.name

        activation_layer_node = LayerNode(
            layer_config, input_layers=[match_layer])

        return activation_layer_node

    if model_type == 'functional':
      inp = keras.layers.Input((3,))
      out = keras.layers.Dense(2)(inp)
      model = keras.Model(inp, out)
    elif model_type == 'sequential':
      model = keras.Sequential([keras.layers.Dense(2, input_shape=(3,))])

    transformed_model, _ = ModelTransformer(
        model, [ReplaceDenseWithDenseAndActivation()]).transform()

    if model_type == 'functional':
      # Extra Input layer over Sequential.
      num_expected_layers = 3
    elif model_type == 'sequential':
      num_expected_layers = 2

    self.assertLen(transformed_model.layers, num_expected_layers)

    self.assertIsInstance(
        self._get_layer(transformed_model, 0, model_type), keras.layers.Dense)
    self.assertIsInstance(
        self._get_layer(transformed_model, 1, model_type),
        keras.layers.Activation)

  def testReplaceSingleLayer_WithMultipleLayers_InputLayer(self):

    class ReplaceInputWithInputAndActivation(transforms.Transform):
      """InputLayer => (InputLayer -> Activation)."""

      def pattern(self):
        return LayerPattern('InputLayer')

      def replacement(self, match_layer):
        activation_layer = keras.layers.Activation('linear')
        layer_config = keras.layers.serialize(activation_layer)
        layer_config['name'] = activation_layer.name

        activation_layer_node = LayerNode(
            layer_config,
            input_layers=[match_layer])

        return activation_layer_node

    inp1 = keras.layers.Input((3,))
    inp2 = keras.layers.Input((3,))
    out = keras.layers.Concatenate()([inp1, inp2])
    model = keras.Model([inp1, inp2], out)

    transformed_model, _ = ModelTransformer(
        model, [ReplaceInputWithInputAndActivation()]).transform()

    self.assertLen(transformed_model.layers, 5)
    self.assertIsInstance(transformed_model.layers[0], keras.layers.InputLayer)
    self.assertIsInstance(transformed_model.layers[1], keras.layers.InputLayer)
    self.assertIsInstance(transformed_model.layers[2], keras.layers.Activation)
    self.assertIsInstance(transformed_model.layers[3], keras.layers.Activation)

  @parameterized.parameters(['sequential', 'functional'])
  def testReplaceChainOfLayers_WithSingleLayer(self, model_type):

    class FuseReLUIntoDense(transforms.Transform):
      """Fuse ReLU into Dense layers."""

      def pattern(self):
        return LayerPattern('ReLU', inputs=[LayerPattern('Dense')])

      def replacement(self, match_layer):
        dense_layer_config = match_layer.input_layers[0].layer['config']
        dense_layer_weights = match_layer.input_layers[0].weights
        dense_layer_config['activation'] = 'relu'

        new_dense_layer = keras.layers.Dense(**dense_layer_config)

        replace_layer = keras.layers.serialize(new_dense_layer)
        replace_layer['name'] = replace_layer['config']['name']

        return LayerNode(replace_layer, dense_layer_weights, [])

    if model_type == 'functional':
      inp = keras.layers.Input((3,))
      out = keras.layers.Dense(2, activation='relu')(inp)
      model_fused = keras.Model(inp, out)
    else:
      model_fused = keras.Sequential(
          [keras.layers.Dense(2, activation='relu', input_shape=(3,))])

    if model_type == 'functional':
      inp = keras.layers.Input((3,))
      x = keras.layers.Dense(2)(inp)
      out = keras.layers.ReLU()(x)
      model = keras.Model(inp, out)
    else:
      model = keras.Sequential(
          [keras.layers.Dense(2, input_shape=(3,)),
           keras.layers.ReLU()])
    model.set_weights(model_fused.get_weights())

    transformed_model, _ = ModelTransformer(
        model, [FuseReLUIntoDense()]).transform()

    self._assert_config(
        model_fused.get_config(),
        transformed_model.get_config(),
        # Layers have different names in the models, but same config.
        # Consider verifying the names loosely.
        #
        # build_input_shape is a TensorShape object and the two objects are not
        # considered the same even though the shapes are the same.
        [
            'input_layers', 'output_layers', 'name', 'inbound_nodes',
            'build_input_shape'
        ])

    self._assert_model_results_equal(model, transformed_model)
    self._assert_model_results_equal(model_fused, transformed_model)

  @parameterized.parameters(['sequential', 'functional'])
  def testReplaceChainOfLayers_WithChainOfLayers(self, model_type):

    class Replace2DenseLayers(transforms.Transform):
      """Replaces 2 Dense layers with the same dense layers.

      Doesn't make any meaningful change to the layer. Just verifies that
      replacing multiple layers works as expected.
      """

      def pattern(self):
        return LayerPattern('Dense', inputs=[LayerPattern('Dense')])

      def replacement(self, match_layer):
        # Adds a modification so the transform happens. If the layers are
        # exactly the same, they get ignored by transformer.
        match_layer.metadata['key'] = 'value'
        return match_layer

    if model_type == 'functional':
      inp = keras.layers.Input((3,))
      x = keras.layers.Dense(3)(inp)
      x = keras.layers.Dense(2)(x)
      model = keras.Model(inp, x)
    else:
      model = keras.Sequential(
          [keras.layers.Dense(3, input_shape=(3,)),
           keras.layers.Dense(2)])

    transformed_model, _ = ModelTransformer(
        model, [Replace2DenseLayers()]).transform()

    self._assert_model_results_equal(model, transformed_model)

    # build_input_shape is a TensorShape object and the two objects are not
    # considered the same even though the shapes are the same.
    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['build_input_shape'])

  def testReplaceListOfLayers_Sequential(self):
    class ReplaceConvBatchNorm(transforms.Transform):
      """Replaces a ConvBatchNorm pattern with the same set of layers.

      Doesn't make any meaningful change to the layer. Just verifies that
      replacing multiple layers works as expected.
      """

      def pattern(self):
        return LayerPattern('BatchNormalization',
                            inputs=[LayerPattern('Conv2D')])

      def replacement(self, match_layer):
        # Adds a modification so the transform happens. If the layers are
        # exactly the same, they get ignored by transformer.
        match_layer.metadata['key'] = 'value'
        return match_layer

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 5, input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])
    model_layer_names = [layer.name for layer in model.layers]

    transformed_model, _ = ModelTransformer(
        model, [ReplaceConvBatchNorm()]).transform()
    transformed_model_layer_names = [
        layer.name for layer in transformed_model.layers]

    self.assertEqual(model_layer_names, transformed_model_layer_names)

  def testReplaceTreeOfLayers_WithSingleLayer(self):
    # TODO(pulkitb): Implement
    pass

  def testReplaceTreeOfLayers_WithTreeOfLayers(self):
    # TODO(pulkitb): Implement
    pass

  @parameterized.parameters(['sequential', 'functional'])
  def testReplace_AlreadyReplacedLayer_WithAnotherMatch(self, model_type):
    """Verifies a layer replaced using a transform can be replaced again."""

    class ReplaceReLUWithSoftmax(Transform):

      def pattern(self):
        return LayerPattern('ReLU')

      def replacement(self, match_layer):
        replace_layer = keras.layers.serialize(keras.layers.Softmax())
        replace_layer['name'] = replace_layer['config']['name']
        return LayerNode(replace_layer)

    class ReplaceSoftmaxWithELU(Transform):

      def pattern(self):
        return LayerPattern('Softmax')

      def replacement(self, match_layer):
        replace_layer = keras.layers.serialize(keras.layers.ELU())
        replace_layer['name'] = replace_layer['config']['name']
        return LayerNode(replace_layer)

    model = self._simple_dense_model(model_type)
    transformed_model, _ = ModelTransformer(
        model,
        [ReplaceReLUWithSoftmax(), ReplaceSoftmaxWithELU()],
        candidate_layers=set([layer.name for layer in model.layers])
    ).transform()

    self.assertEqual(transformed_model.layers[-1].__class__.__name__, 'ELU')

  @parameterized.parameters(['sequential', 'functional'])
  def testDoesNotMatchForever_IfReplacementEqualsMatch(self, model_type):

    class ReplaceWithSelf(Transform):

      def pattern(self):
        return LayerPattern('ReLU', inputs=[LayerPattern('Dense')])

      def replacement(self, match_layer):
        return match_layer

    model = self._simple_dense_model(model_type)

    transformed_model, _ = ModelTransformer(
        model, [ReplaceWithSelf()]).transform()

    # build_input_shape is a TensorShape object and the two objects are not
    # considered the same even though the shapes are the same.
    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['build_input_shape'])

  # Negative Tests
  # TODO(pulkitb): Add negative tests
  # 1. Handles layer being part of multiple models.

  class VerifyMatch(Transform):

    def __init__(self, pattern):
      self._pattern = pattern
      self._matched = False

    def pattern(self):
      return self._pattern

    def replacement(self, match_layer):
      self._matched = True
      return match_layer

    def matched(self):
      return self._matched

    def reset(self):
      self._matched = False

  @parameterized.parameters(['sequential', 'functional'])
  def testPatternShouldOnlyMatch_CandidateLayers(self, model_type):
    pattern = LayerPattern('ReLU', inputs=[LayerPattern('Dense')])
    transform = self.VerifyMatch(pattern)

    model = self._simple_dense_model(model_type)
    layer_names = [layer.name for layer in model.layers]

    # By default matches everything.
    ModelTransformer(model, [transform]).transform()
    self.assertTrue(transform.matched())

    # Matches when all layers passed in.
    transform.reset()
    ModelTransformer(model, [transform], layer_names).transform()
    self.assertTrue(transform.matched())

    # Fails. Dense missing.
    transform.reset()
    ModelTransformer(model, [transform], [model.layers[-2].name]).transform()
    self.assertFalse(transform.matched())

    # Fails. ReLU missing.
    transform.reset()
    ModelTransformer(model, [transform], [model.layers[-1].name]).transform()
    self.assertFalse(transform.matched())

  @parameterized.parameters(['sequential', 'functional'])
  def testPatternCanMatch_MultipleLayers(self, model_type):
    pattern = LayerPattern('Conv2D|DepthwiseConv2D')
    transform = self.VerifyMatch(pattern)

    if model_type == 'functional':
      inp = keras.layers.Input((3, 3, 3))
      x = keras.layers.Conv2D(3, (2, 2))(inp)
      conv_model = keras.Model(inp, x)
    else:
      conv_model = keras.Sequential(
          [keras.layers.Conv2D(3, (2, 2), input_shape=(3, 3, 3))])

    ModelTransformer(conv_model, [transform]).transform()
    self.assertTrue(transform.matched())

    if model_type == 'functional':
      inp = keras.layers.Input((3, 3, 3))
      x = keras.layers.DepthwiseConv2D((2, 2))(inp)
      depth_conv_model = keras.Model(inp, x)
    else:
      depth_conv_model = keras.Sequential(
          [keras.layers.DepthwiseConv2D((2, 2), input_shape=(3, 3, 3))])

    transform.reset()
    ModelTransformer(depth_conv_model, [transform]).transform()
    self.assertTrue(transform.matched())

  def testPatternCanMatch_HeadNodeWithMultipleConsumers(self):
    # Dense -> Dense2  -> ReLU
    #                  -> ReLU2
    #
    # where Dense2, the head node in the pattern, has multiple consumers
    # (ReLU and ReLU2).
    pattern = LayerPattern('Dense', inputs=[LayerPattern('Dense')])
    transform = self.VerifyMatch(pattern)

    inp = keras.layers.Input(3)
    x = keras.layers.Dense(2)(inp)
    y = keras.layers.Dense(2)(x)
    out1 = keras.layers.ReLU(6.0)(y)
    out2 = keras.layers.ReLU(6.0)(y)

    model = keras.Model(inp, [out1, out2])

    ModelTransformer(model, [transform]).transform()
    self.assertTrue(transform.matched())

  def testPatternDoesNotSupportMatch_IntermediateNodeWithMultipleConsumers(
      self):
    # Dense -> Dense2  -> ReLU
    #                  -> ReLU2
    #
    # where Dense2, an intermediate node in the pattern, has multiple consumers
    # (ReLU and ReLU2).
    pattern = LayerPattern(
        'ReLU', inputs=[LayerPattern('Dense', inputs=[LayerPattern('Dense')])])
    transform = self.VerifyMatch(pattern)

    inp = keras.layers.Input(3)
    x = keras.layers.Dense(2)(inp)
    y = keras.layers.Dense(2)(x)
    out1 = keras.layers.ReLU(6.0)(y)
    out2 = keras.layers.ReLU(6.0)(y)

    model = keras.Model(inp, [out1, out2])

    ModelTransformer(model, [transform]).transform()
    self.assertFalse(transform.matched())

  @parameterized.parameters(['sequential', 'functional'])
  def testLayerMetadataPassedAndReplacedInTransforms(self, model_type):

    class ReplaceLayerMetadata(Transform):

      def pattern(self):
        return LayerPattern('Dense')

      def replacement(self, match_layer):
        if match_layer.metadata['key'] == 'Hello':
          match_layer.metadata['key'] = 'World'
        return match_layer

    model = self._simple_dense_model(model_type)

    dense_layer = self._get_layer(model, 0, model_type)
    relu_layer = self._get_layer(model, 1, model_type)

    layer_metadata = {
        dense_layer.name: {
            'key': 'Hello'
        },
        relu_layer.name: {
            'key': 'Hello'
        },
    }

    expected_metadata = {
        dense_layer.name: {
            'key': 'World'
        },
        relu_layer.name: {
            'key': 'Hello'
        }
    }

    transformer = ModelTransformer(
        model, [ReplaceLayerMetadata()], None, layer_metadata)
    transformed_model, updated_metadata = transformer.transform()

    self.assertEqual(expected_metadata, updated_metadata)

    # build_input_shape is a TensorShape object and the two objects are not
    # considered the same even though the shapes are the same.
    self._assert_config(model.get_config(), transformed_model.get_config(),
                        ['build_input_shape'])

  # Validation Tests

  def testRaisesErrorForSubclassModels(self):
    class MyModel(keras.Model):
      pass

    with self.assertRaises(ValueError):
      ModelTransformer(MyModel(), [self.ReplaceDenseLayer()]).transform()

if __name__ == '__main__':
  tf.test.main()
