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
"""Tests for Keras' base preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

from tensorflow.python import keras

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine import base_preprocessing_layer_v1
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


# Define a test-only implementation of CombinerPreprocessingLayer to validate
# its correctness directly.
class AddingPreprocessingLayer(
    base_preprocessing_layer.CombinerPreprocessingLayer):
  _SUM_NAME = "sum"

  def __init__(self, **kwargs):
    super(AddingPreprocessingLayer, self).__init__(
        combiner=self.AddingCombiner(), **kwargs)

  def build(self, input_shape):
    super(AddingPreprocessingLayer, self).build(input_shape)
    self._sum = self._add_state_variable(
        name=self._SUM_NAME,
        shape=(1,),
        dtype=dtypes.float32,
        initializer=init_ops.zeros_initializer)

  def set_total(self, sum_value):
    """This is an example of how a subclass would implement a direct setter.

    These methods should generally just create a dict mapping the correct names
    to the relevant passed values, and call self._set_state_variables() with the
    dict of data.

    Args:
      sum_value: The total to set.
    """
    self._set_state_variables({self._SUM_NAME: [sum_value]})

  def call(self, inputs):
    return inputs + self._sum

  # Define a Combiner for this layer class.
  class AddingCombiner(base_preprocessing_layer.Combiner):

    def compute(self, batch_values, accumulator=None):
      """Compute a step in this computation, returning a new accumulator."""
      new_accumulator = 0 if batch_values is None else np.sum(batch_values)
      if accumulator is None:
        return new_accumulator
      else:
        return self.merge([accumulator, new_accumulator])

    def merge(self, accumulators):
      """Merge several accumulators to a single accumulator."""
      # Combine accumulators and return the result.
      result = accumulators[0]
      for accumulator in accumulators[1:]:
        result = np.sum([np.sum(result), np.sum(accumulator)])
      return result

    def extract(self, accumulator):
      """Convert an accumulator into a dict of output values."""
      # We have to add an additional dimension here because the weight shape
      # is (1,) not None.
      return {AddingPreprocessingLayer._SUM_NAME: [accumulator]}

    def restore(self, output):
      """Create an accumulator based on 'output'."""
      # There is no special internal state here, so we just return the relevant
      # internal value. We take the [0] value here because the weight itself
      # is of the shape (1,) and we want the scalar contained inside it.
      return output[AddingPreprocessingLayer._SUM_NAME][0]

    def serialize(self, accumulator):
      """Serialize an accumulator for a remote call."""
      return compat.as_bytes(json.dumps(accumulator))

    def deserialize(self, encoded_accumulator):
      """Deserialize an accumulator received from 'serialize()'."""
      return json.loads(compat.as_text(encoded_accumulator))


class AddingPreprocessingLayerV1(
    AddingPreprocessingLayer,
    base_preprocessing_layer_v1.CombinerPreprocessingLayer):
  pass


def get_layer():
  if context.executing_eagerly():
    return AddingPreprocessingLayer()
  else:
    return AddingPreprocessingLayerV1()


@keras_parameterized.run_all_keras_modes
class PreprocessingLayerTest(keras_parameterized.TestCase):

  def test_adapt_list_fails(self):
    """Test that non-Dataset/Numpy inputs cause a reasonable error."""
    input_dataset = [1, 2, 3, 4, 5]

    layer = get_layer()
    with self.assertRaisesRegex(ValueError, ".*a Dataset or a Numpy.*"):
      layer.adapt(input_dataset)

  def test_adapt_infinite_dataset_fails(self):
    """Test that preproc layers fail if an infinite dataset is passed."""
    input_dataset = dataset_ops.Dataset.from_tensor_slices(
        np.array([[1], [2], [3], [4], [5], [0]])).repeat()

    layer = get_layer()
    with self.assertRaisesRegex(ValueError, ".*infinite number of elements.*"):
      layer.adapt(input_dataset)

  def test_pre_build_injected_update_with_no_build_fails(self):
    """Test external update injection before build() is called fails."""
    input_dataset = np.array([1, 2, 3, 4, 5])

    layer = get_layer()
    combiner = layer._combiner
    updates = combiner.extract(combiner.compute(input_dataset))

    with self.assertRaisesRegex(RuntimeError, ".*called after build.*"):
      layer._set_state_variables(updates)

  def test_setter_update(self):
    """Test the prototyped setter method."""
    input_data = keras.Input(shape=(1,))
    layer = get_layer()
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    layer.set_total(15)

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_pre_build_adapt_update_numpy(self):
    """Test that preproc layers can adapt() before build() is called."""
    input_dataset = np.array([1, 2, 3, 4, 5])

    layer = get_layer()
    layer.adapt(input_dataset)

    input_data = keras.Input(shape=(1,))
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_post_build_adapt_update_numpy(self):
    """Test that preproc layers can adapt() after build() is called."""
    input_dataset = np.array([1, 2, 3, 4, 5])

    input_data = keras.Input(shape=(1,))
    layer = get_layer()
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    layer.adapt(input_dataset)

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_pre_build_injected_update(self):
    """Test external update injection before build() is called."""
    input_dataset = np.array([1, 2, 3, 4, 5])

    layer = get_layer()
    combiner = layer._combiner
    updates = combiner.extract(combiner.compute(input_dataset))

    layer.build((1,))
    layer._set_state_variables(updates)

    input_data = keras.Input(shape=(1,))
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_post_build_injected_update(self):
    """Test external update injection after build() is called."""
    input_dataset = np.array([1, 2, 3, 4, 5])
    input_data = keras.Input(shape=(1,))
    layer = get_layer()
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    combiner = layer._combiner
    updates = combiner.extract(combiner.compute(input_dataset))
    layer._set_state_variables(updates)

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_pre_build_adapt_update_dataset(self):
    """Test that preproc layers can adapt() before build() is called."""
    input_dataset = dataset_ops.Dataset.from_tensor_slices(
        np.array([[1], [2], [3], [4], [5], [0]]))

    layer = get_layer()
    layer.adapt(input_dataset)

    input_data = keras.Input(shape=(1,))
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_post_build_adapt_update_dataset(self):
    """Test that preproc layers can adapt() after build() is called."""
    input_dataset = dataset_ops.Dataset.from_tensor_slices(
        np.array([[1], [2], [3], [4], [5], [0]]))

    input_data = keras.Input(shape=(1,))
    layer = get_layer()
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    layer.adapt(input_dataset)

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_further_tuning(self):
    """Test that models can be tuned with multiple calls to 'adapt'."""

    input_dataset = np.array([1, 2, 3, 4, 5])

    layer = get_layer()
    layer.adapt(input_dataset)

    input_data = keras.Input(shape=(1,))
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

    layer.adapt(np.array([1, 2]), reset_state=False)
    self.assertAllEqual([[19], [20], [21]], model.predict([1., 2., 3.]))

  def test_further_tuning_post_injection(self):
    """Test that models can be tuned with multiple calls to 'adapt'."""

    input_dataset = np.array([1, 2, 3, 4, 5])

    layer = get_layer()

    input_data = keras.Input(shape=(1,))
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    model._experimental_run_tf_function = testing_utils.should_run_tf_function()

    combiner = layer._combiner
    updates = combiner.extract(combiner.compute(input_dataset))
    layer._set_state_variables(updates)
    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

    layer.adapt(np.array([1, 2]), reset_state=False)
    self.assertAllEqual([[19], [20], [21]], model.predict([1., 2., 3.]))

  def test_weight_based_state_transfer(self):
    """Test that preproc layers can transfer state via get/set weights.."""

    def get_model():
      input_data = keras.Input(shape=(1,))
      layer = get_layer()
      output = layer(input_data)
      model = keras.Model(input_data, output)
      model._run_eagerly = testing_utils.should_run_eagerly()
      model._experimental_run_tf_function = (
          testing_utils.should_run_tf_function())
      return (model, layer)

    input_dataset = np.array([1, 2, 3, 4, 5])
    model, layer = get_model()
    layer.adapt(input_dataset)
    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

    # Create a new model and verify it has no state carryover.
    weights = model.get_weights()
    model_2, _ = get_model()
    self.assertAllEqual([[1], [2], [3]], model_2.predict([1., 2., 3.]))

    # Transfer state from model to model_2 via get/set weights.
    model_2.set_weights(weights)
    self.assertAllEqual([[16], [17], [18]], model_2.predict([1., 2., 3.]))

  def test_weight_based_state_transfer_with_further_tuning(self):
    """Test that transferred state can be used to further tune a model.."""

    def get_model():
      input_data = keras.Input(shape=(1,))
      layer = get_layer()
      output = layer(input_data)
      model = keras.Model(input_data, output)
      model._run_eagerly = testing_utils.should_run_eagerly()
      model._experimental_run_tf_function = (
          testing_utils.should_run_tf_function())
      return (model, layer)

    input_dataset = np.array([1, 2, 3, 4, 5])
    model, layer = get_model()
    layer.adapt(input_dataset)
    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

    # Transfer state from model to model_2 via get/set weights.
    weights = model.get_weights()
    model_2, layer_2 = get_model()
    model_2.set_weights(weights)

    # Further adapt this layer based on the transferred weights.
    layer_2.adapt(np.array([1, 2]), reset_state=False)
    self.assertAllEqual([[19], [20], [21]], model_2.predict([1., 2., 3.]))


if __name__ == "__main__":
  test.main()
