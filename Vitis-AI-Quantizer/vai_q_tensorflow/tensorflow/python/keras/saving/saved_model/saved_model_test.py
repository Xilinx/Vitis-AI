# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Tests for saving/loading function for keras Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column.dense_features import DenseFeatures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import save as keras_save
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import save as tf_save
from tensorflow.python.util import tf_inspect


class LayerWithLearningPhase(keras.engine.base_layer.Layer):

  def build(self, input_shape):
    self.input_spec = keras.layers.InputSpec(shape=[None] * len(input_shape))
    self.built = True

  def call(self, x, training=None):
    if training is None:
      training = keras.backend.learning_phase()
    output = tf_utils.smart_cond(
        training, lambda: x * 0, lambda: array_ops.identity(x))
    if not context.executing_eagerly():
      output._uses_learning_phase = True  # pylint: disable=protected-access
    return output

  def compute_output_shape(self, input_shape):
    return input_shape


class LayerWithLoss(keras.layers.Layer):

  def call(self, inputs):
    self.add_loss(math_ops.reduce_sum(inputs), inputs)
    return inputs


@test_util.run_all_in_graph_and_eager_modes
class TestModelSavingAndLoadingV2(keras_parameterized.TestCase):

  def _save_model_dir(self, dirname='saved_model'):
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    return os.path.join(temp_dir, dirname)

  @keras_parameterized.run_with_all_model_types
  def test_model_save_and_load(self):
    input_arr = np.random.random((1, 3)).astype(np.float32)
    target_arr = np.random.random((1, 4)).astype(np.float32)

    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.layers[-1].activity_regularizer = regularizers.get('l2')
    model.activity_regularizer = regularizers.get('l2')
    model.compile(
        loss='mse',
        optimizer='rmsprop')
    model.train_on_batch(input_arr, target_arr)

    def callable_loss():
      return math_ops.reduce_sum(model.weights[0])
    model.add_loss(callable_loss)
    saved_model_dir = self._save_model_dir()
    tf_save.save(model, saved_model_dir)
    loaded = keras_load.load(saved_model_dir)
    self.evaluate(variables.variables_initializer(loaded.variables))
    self.assertAllClose(self.evaluate(model.weights),
                        self.evaluate(loaded.weights))

    input_arr = constant_op.constant(
        np.random.random((1, 3)).astype(np.float32))
    self.assertAllClose(self.evaluate(model(input_arr)),
                        self.evaluate(loaded(input_arr)))
    # Validate losses. The order of conditional losses may change between the
    # model and loaded model, so sort the losses first.
    if context.executing_eagerly():
      self.assertAllClose(sorted(self.evaluate(model.losses)),
                          sorted(self.evaluate(loaded.losses)))
    else:
      self.assertAllClose(self.evaluate(model.get_losses_for(None)),
                          self.evaluate(loaded.get_losses_for(None)))
      self.assertAllClose(
          sorted(self.evaluate(model.get_losses_for(input_arr))),
          sorted(self.evaluate(loaded.get_losses_for(input_arr))))

  def test_trainable_weights(self):
    layer = keras.layers.Dense(4, name='custom_layer')
    layer.build([3,])
    layer.add_weight(
        'extra_weight', shape=[],
        initializer=init_ops.constant_initializer(11),
        trainable=True)
    layer.add_weight(
        'extra_weight_2', shape=[],
        initializer=init_ops.constant_initializer(12),
        trainable=False)

    saved_model_dir = self._save_model_dir()
    self.evaluate(variables.variables_initializer(layer.variables))
    tf_save.save(layer, saved_model_dir)
    loaded = keras_load.load(saved_model_dir)
    self.evaluate(variables.variables_initializer(loaded.variables))

    equal_attrs = ['name', '_expects_training_arg', 'trainable']
    for attr in equal_attrs:
      self.assertEqual(getattr(layer, attr), getattr(loaded, attr))

    all_close = ['weights', 'trainable_weights', 'non_trainable_weights']
    for attr in all_close:
      self.assertAllClose(self.evaluate(getattr(layer, attr)),
                          self.evaluate(getattr(loaded, attr)))

  def test_maintains_losses(self):
    """Tests that the layer losses do not change before and after export."""
    model = keras.models.Sequential([LayerWithLoss()])
    model.compile(
        loss='mse',
        optimizer='rmsprop')
    input_arr = np.random.random((1, 3)).astype(np.float32)
    target_arr = np.random.random((1, 3)).astype(np.float32)

    # Test that symbolic losses are maintained (train_on_batch saves symbolic
    # losses.)
    model.train_on_batch(input_arr, target_arr)
    previous_losses = model.losses[:]

    saved_model_dir = self._save_model_dir()
    tf_save.save(model, saved_model_dir)
    self.assertAllEqual(previous_losses, model.losses)

    if context.executing_eagerly():
      # Test that eager losses are maintained.
      model(input_arr)  # Calls model eagerly, creating eager losses.
      previous_losses = model.losses[:]
      tf_save.save(model, saved_model_dir)
      self.assertAllEqual(previous_losses, model.losses)

  def test_layer_with_learning_phase(self):
    layer = LayerWithLearningPhase()
    layer.build([None, None])
    saved_model_dir = self._save_model_dir()
    tf_save.save(layer, saved_model_dir)
    loaded = keras_load.load(saved_model_dir)
    input_arr = array_ops.ones((4, 3))

    # Run the layer, and use the keras backend learing phase
    keras.backend.set_learning_phase(0)
    self.assertAllEqual(input_arr, loaded(input_arr))
    keras.backend.set_learning_phase(1)
    self.assertAllEqual(array_ops.zeros((4, 3)), loaded(input_arr))

    # Run the layer while explicitly setting the training argument
    self.assertAllEqual(
        input_arr, loaded(input_arr, training=constant_op.constant(False)))
    self.assertAllEqual(
        array_ops.zeros((4, 3)),
        loaded(input_arr, training=constant_op.constant(True)))

  @keras_parameterized.run_with_all_model_types
  def test_standard_loader(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.activity_regularizer = regularizers.get('l2')
    def eager_loss():
      return math_ops.reduce_sum(model.weights[0])
    model.add_loss(eager_loss)

    # Call predict to ensure that all layers are built and inputs are set.
    model.predict(np.random.random((1, 3)))
    saved_model_dir = self._save_model_dir()

    tf_save.save(model, saved_model_dir)

    loaded = tf_load.load(saved_model_dir)
    self.evaluate(variables.variables_initializer(loaded.variables))
    all_close = ['variables', 'trainable_variables',
                 'non_trainable_variables']
    for attr in all_close:
      self.assertAllClose(self.evaluate(getattr(model, attr)),
                          self.evaluate(getattr(loaded.keras_api, attr)))
    self.assertLen(loaded.regularization_losses, 1)
    expected_layers = len(model.layers)
    self.assertEqual(expected_layers, len(loaded.keras_api.layers))
    input_arr = array_ops.ones((4, 3))
    self.assertAllClose(self.evaluate(model(input_arr)),
                        self.evaluate(loaded(input_arr, training=False)))

  @keras_parameterized.run_with_all_model_types
  def test_compiled_model(self):
    input_arr = np.random.random((1, 3))
    target_arr = np.random.random((1, 4))

    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    expected_predict = model.predict(input_arr)

    # Compile and save model.
    model.compile('rmsprop', 'mse')
    saved_model_dir = self._save_model_dir()
    tf_save.save(model, saved_model_dir)

    # TODO(b/134519980): Issue with model.fit if the model call function uses
    # a tf.function (Graph mode only).
    with context.eager_mode():
      loaded = keras_load.load(saved_model_dir)
      actual_predict = loaded.predict(input_arr)
      self.assertAllClose(expected_predict, actual_predict)

      loss_before = loaded.evaluate(input_arr, target_arr)
      loaded.fit(input_arr, target_arr)
      loss_after = loaded.evaluate(input_arr, target_arr)
      self.assertLess(loss_after, loss_before)
      predict = loaded.predict(input_arr)

      ckpt_path = os.path.join(self.get_temp_dir(), 'weights')
      loaded.save_weights(ckpt_path)

    # Ensure that the checkpoint is compatible with the original model.
    model.load_weights(ckpt_path)
    self.assertAllClose(predict, model.predict(input_arr))

  def test_metadata_input_spec(self):
    class LayerWithNestedSpec(keras.layers.Layer):

      def __init__(self):
        super(LayerWithNestedSpec, self).__init__()
        self.input_spec = {
            'a': keras.layers.InputSpec(max_ndim=3, axes={-1: 2}),
            'b': keras.layers.InputSpec(shape=(None, 2, 3), dtype='float16')}

    layer = LayerWithNestedSpec()
    saved_model_dir = self._save_model_dir()
    tf_save.save(layer, saved_model_dir)
    loaded = keras_load.load(saved_model_dir)
    self.assertEqual(3, loaded.input_spec['a'].max_ndim)
    self.assertEqual({-1: 2}, loaded.input_spec['a'].axes)
    self.assertAllEqual([None, 2, 3], loaded.input_spec['b'].shape)
    self.assertEqual('float16', loaded.input_spec['b'].dtype)

  def test_multi_input_model(self):
    input_1 = keras.layers.Input(shape=(3,))
    input_2 = keras.layers.Input(shape=(5,))
    model = keras.Model([input_1, input_2], [input_1, input_2])
    saved_model_dir = self._save_model_dir()

    model.save(saved_model_dir, save_format='tf')
    loaded = keras_load.load(saved_model_dir)
    input_arr_1 = np.random.random((1, 3)).astype('float32')
    input_arr_2 = np.random.random((1, 5)).astype('float32')

    outputs = loaded([input_arr_1, input_arr_2])
    self.assertAllEqual(input_arr_1, outputs[0])
    self.assertAllEqual(input_arr_2, outputs[1])

  def test_revived_sequential(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(5, input_shape=(3,),
                                 kernel_regularizer=regularizers.get('l2')))
    model.add(keras.layers.Dense(2, kernel_regularizer=regularizers.get('l2')))

    self.evaluate(variables.variables_initializer(model.variables))

    saved_model_dir = self._save_model_dir()
    model.save(saved_model_dir, save_format='tf')
    loaded = keras_load.load(saved_model_dir)

    self.assertLen(loaded.layers, 2)
    self.assertLen(loaded.losses, 2)

    loaded.pop()

    self.assertLen(loaded.layers, 1)
    self.assertLen(loaded.losses, 1)

    loaded.add(keras.layers.Dense(2, kernel_regularizer=regularizers.get('l2')))

    self.assertLen(loaded.layers, 2)
    self.assertLen(loaded.losses, 2)

  def testBatchNormUpdates(self):
    model = keras.models.Sequential(
        keras.layers.BatchNormalization(input_shape=(1,)))
    self.evaluate(variables.variables_initializer(model.variables))
    saved_model_dir = self._save_model_dir()
    model.save(saved_model_dir, save_format='tf')
    loaded = keras_load.load(saved_model_dir)
    self.evaluate(variables.variables_initializer(loaded.variables))
    input_arr_1 = np.array([[11], [12], [13]]).astype('float32')
    self.assertAllClose(self.evaluate(loaded.layers[-1].moving_mean), [0])
    self.evaluate(loaded(input_arr_1, training=True))
    self.assertAllClose(self.evaluate(loaded.layers[-1].moving_mean), [0.12])
    self.evaluate(loaded(input_arr_1, training=False))
    self.assertAllClose(self.evaluate(loaded.layers[-1].moving_mean), [0.12])

  def testSaveWithSignatures(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(5, input_shape=(3,),
                                 kernel_regularizer=regularizers.get('l2')))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4, kernel_regularizer=regularizers.get('l2')))

    input_arr = np.random.random((2, 3)).astype(np.float32)
    target_arr = np.random.random((2, 4)).astype(np.float32)

    model.compile(
        loss='mse',
        optimizer='rmsprop')
    model.train_on_batch(input_arr, target_arr)

    @def_function.function(input_signature=[tensor_spec.TensorSpec((None, 3))])
    def predict(inputs):
      return {'predictions': model(inputs)}

    feature_configs = {
        'inputs': parsing_ops.FixedLenFeature(
            shape=[2, 3], dtype=dtypes.float32)}

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([None], dtypes.string)])
    def parse_and_predict(examples):
      features = parsing_ops.parse_single_example(examples[0], feature_configs)
      return {'predictions': model(features['inputs']),
              'layer_1_outputs': model.layers[0](features['inputs'])}

    saved_model_dir = self._save_model_dir()
    model.save(saved_model_dir, save_format='tf', signatures={
        'predict': predict,
        'parse_and_predict': parse_and_predict})
    model.save('/tmp/saved', save_format='tf', signatures={
        'predict': predict,
        'parse_and_predict': parse_and_predict})

    loaded = keras_load.load(saved_model_dir)

    self.assertAllClose(
        model.predict(input_arr),
        loaded.signatures['predict'](
            ops.convert_to_tensor(input_arr))['predictions'])

    feature = {
        'inputs': feature_pb2.Feature(
            float_list=feature_pb2.FloatList(value=input_arr.flatten()))}
    example = example_pb2.Example(
        features=feature_pb2.Features(feature=feature))
    outputs = loaded.signatures['parse_and_predict'](
        ops.convert_to_tensor([example.SerializeToString()]))
    self.assertAllClose(model.predict(input_arr), outputs['predictions'])
    self.assertAllClose(model.layers[0](input_arr), outputs['layer_1_outputs'])

  def testTrainingDefaults(self):
    def assert_training_default(fn, default_value):
      arg_spec = tf_inspect.getfullargspec(fn)
      index = len(arg_spec.args) - arg_spec.args.index('training')
      self.assertEqual(arg_spec.defaults[-index], default_value)

    class LayerWithTrainingRequiredArg(keras.engine.base_layer.Layer):

      def call(self, inputs, training):
        return tf_utils.smart_cond(
            training, lambda: inputs * 0, lambda: array_ops.identity(inputs))

    class LayerWithTrainingDefaultTrue(keras.engine.base_layer.Layer):

      def call(self, inputs, training=True):
        return tf_utils.smart_cond(
            training, lambda: inputs * 0, lambda: array_ops.identity(inputs))

    class Model(keras.models.Model):

      def __init__(self):
        super(Model, self).__init__()
        self.layer_with_training_default_none = LayerWithLearningPhase()
        self.layer_with_training_default_true = LayerWithTrainingDefaultTrue()
        self.layer_with_required_training_arg = LayerWithTrainingRequiredArg()

      def call(self, inputs):
        x = self.layer_with_training_default_none(inputs)
        x += self.layer_with_training_default_true(inputs)
        x += self.layer_with_required_training_arg(inputs, False)
        return x

    model = Model()
    # Build and set model inputs
    model.predict(np.ones([1, 3]).astype('float32'))
    saved_model_dir = self._save_model_dir()
    model.save(saved_model_dir, save_format='tf')
    load = tf_load.load(saved_model_dir)

    assert_training_default(load.__call__, False)
    assert_training_default(
        load.layer_with_training_default_none.__call__, False)
    assert_training_default(
        load.layer_with_training_default_true.__call__, True)

    # Assert that there are no defaults for layer with required training arg
    arg_spec = tf_inspect.getfullargspec(
        load.layer_with_required_training_arg.__call__)
    self.assertFalse(arg_spec.defaults)  # defaults is None or empty

  def testTraceModelWithKwarg(self):
    class Model(keras.models.Model):

      def call(self, inputs, keyword=None):
        return array_ops.identity(inputs)

    model = Model()
    prediction = model.predict(np.ones([1, 3]).astype('float32'))
    saved_model_dir = self._save_model_dir()
    model.save(saved_model_dir, save_format='tf')

    loaded = keras_load.load(saved_model_dir)
    self.assertAllClose(prediction,
                        loaded.predict(np.ones([1, 3]).astype('float32')))

  def testFeatureColumns(self):
    # TODO(b/120099662): Error with table initialization with Keras models in
    # graph mode.
    if context.executing_eagerly():
      numeric = fc.numeric_column('a')
      bucketized = fc.bucketized_column(numeric, boundaries=[5, 10, 15])
      cat_vocab = fc.categorical_column_with_vocabulary_list(
          'b', ['1', '2', '3'])
      one_hot = fc.indicator_column(cat_vocab)
      embedding = fc.embedding_column(cat_vocab, dimension=8)
      feature_layer = DenseFeatures([bucketized, one_hot, embedding])
      model = keras.models.Sequential(feature_layer)

      features = {'a': np.array([13, 15]), 'b': np.array(['1', '2'])}
      predictions = model.predict(features)

      saved_model_dir = self._save_model_dir()
      model.save(saved_model_dir, save_format='tf')
      loaded = keras_load.load(saved_model_dir)
      loaded_predictions = loaded.predict(features)
      self.assertAllClose(predictions, loaded_predictions)



class TestLayerCallTracing(test.TestCase):

  def test_functions_have_same_trace(self):

    class Layer(keras.engine.base_layer.Layer):

      def call(self, inputs):
        return inputs

      def call2(self, inputs):
        return inputs * 2

    layer = Layer()
    call_collection = keras_save.LayerCallCollection(layer)
    fn = call_collection.add_function(layer.call, 'call')
    fn2 = call_collection.add_function(layer.call2, 'call2')

    fn(np.ones((2, 3)))
    fn(np.ones((4, 5)))

    self.assertLen(fn._list_all_concrete_functions_for_serialization(), 2)
    self.assertLen(fn2._list_all_concrete_functions_for_serialization(), 2)

    # Check that the shapes are correct
    self.assertEqual(
        {(2, 3), (4, 5)},
        set(tuple(c.structured_input_signature[0][0].shape.as_list())
            for c in fn2._list_all_concrete_functions_for_serialization()))

  def test_training_arg_replacement(self):

    def assert_num_traces(layer_cls, training_keyword):
      layer = layer_cls()
      call_collection = keras_save.LayerCallCollection(layer)
      fn = call_collection.add_function(layer.call, 'call')

      fn(np.ones((2, 3)), training=True)
      self.assertLen(fn._list_all_concrete_functions_for_serialization(), 2)

      fn(np.ones((2, 4)), training=False)
      self.assertLen(fn._list_all_concrete_functions_for_serialization(), 4)

      if training_keyword:
        fn(np.ones((2, 5)), True)
        self.assertLen(fn._list_all_concrete_functions_for_serialization(), 6)
        fn(np.ones((2, 6)))
        self.assertLen(fn._list_all_concrete_functions_for_serialization(), 8)

    class LayerWithTrainingKeyword(keras.engine.base_layer.Layer):

      def call(self, inputs, training=False):
        return inputs * training

    assert_num_traces(LayerWithTrainingKeyword, training_keyword=True)

    class LayerWithKwargs(keras.engine.base_layer.Layer):

      def call(self, inputs, **kwargs):
        return inputs * kwargs['training']

    assert_num_traces(LayerWithKwargs, training_keyword=False)

    class LayerWithChildLayer(keras.engine.base_layer.Layer):

      def __init__(self):
        self.child = LayerWithKwargs()
        super(LayerWithChildLayer, self).__init__()

      def call(self, inputs):
        return self.child(inputs)

    assert_num_traces(LayerWithChildLayer, training_keyword=False)

  @test_util.run_in_graph_and_eager_modes
  def test_maintains_losses(self):
    layer = LayerWithLoss()
    layer(np.ones((2, 3)))
    previous_losses = layer.losses[:]

    call_collection = keras_save.LayerCallCollection(layer)
    fn = call_collection.add_function(layer.call, 'call')
    fn(np.ones((2, 3)))

    self.assertAllEqual(previous_losses, layer.losses)


if __name__ == '__main__':
  test.main()
