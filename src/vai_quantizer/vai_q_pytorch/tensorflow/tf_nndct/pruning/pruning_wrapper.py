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

import inspect
import numpy as np

from tensorflow import keras
from nndct_shared.pruning import errors
from tf_nndct.pruning import pruning_impl

K = keras.backend

class PruneMaskedWeight(keras.layers.Wrapper):
  """This wrapper augments a keras layer so the weight tensor may be pruned.

  This wrapper implements magnitude-based pruning of the weight tensors.
  Magnitude-based pruning achieves a target sparsity (s% of zeros) for a given
  weight tensor by monitoring the distribution of the absolute values of the
  weight tensor and determining the weight value (referred to as threshold)
  below which s% of elements lie. For every weight tensor being pruned, the
  wrapper maintains an identically shaped tensor (referred to as mask) which
  stores 0 if the weight value lies below the threshold.
  The mask and thresholds are computed during the training based on the
  evolution of the weight values.

  Block sparse patterns:
  For certain SIMD hardware architectures, it may be beneficial to induce
  spatially correlated sparsity. To train models in which the weight tensors
  have block sparse structure, the pruning wrapper can be configured with
  the block_height and block_width configuration parameters set to the desired
  block configuration (2x2, 4x4, 4x1, 1x8, etc). This is applicable to
  rank-2 weight tensor only and the tensor partitioned into non-overlapping
  blocks of size [block_height, block_dim]. Either the average or max absolute
  value in this block is taken as a proxy for the entire block
  (set by block_pooling_function configuration parameter)
  while computing the distribution of the weight values and
  the threshold for pruning.

  Custom keras layers:
  The pruning wrapper can also be applied to a user-defined keras layer.
  Such a layer may contain one or more weight tensors that may be pruned.
  To apply pruning wrapper to such layers, the layer should be a `PrunableLayer`
  instance or, more directly, user should define a `get_prunable_weights` method
  for the layer (Check the pruning_wrapper_test.CustomLayerPrunable for more
  details about how to define a user-defined prunable layer).

  Sparsity function:
  The target sparsity for the weight tensors are set through the
  pruning_schedule parameter of the pruning wrapper. The user must create a
  python callable that returns a scalar tensorflow tensor and pass this
  callable to the sparsity_function parameter. This scalar tensor contains the
  target sparsity value for the weight tensors in the layer.
  The wrapper provides the following pre-built sparsity functions:

  """

  def __init__(self, layer, **kwargs):
    """Create a pruning wrapper for a keras layer.

    Args:
      layer: The keras layer to be pruned.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """

    # An instance of the Pruning class. This class contains the logic to prune
    # the weights of this layer.
    self.pruning_obj = None

    # A list of all (weight,mask) tuples for this layer
    self.pruning_vars = []

    if not isinstance(layer, tf.keras.layers.Layer):
      raise errors.OptimizerKerasLayerError(
          'Please initialize `Prune` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))

    # TODO(pulkitb): This should be pushed up to the wrappers.py
    # Name the layer using the wrapper and underlying layer name.
    # Prune(Dense) becomes prune_dense_1
    kwargs.update({'name': 'pruned_{}'.format(layer.name)})
    super(PruneMaskedWeight, self).__init__(layer, **kwargs)

    self._track_trackable(layer, name='layer')

    # TODO(yunluli): Work-around to handle the first layer of Sequential model
    # properly. Can remove this when it is implemented in the Wrapper base
    # class.
    #
    # Enables end-user to prune the first layer in Sequential models, while
    # passing the input shape to the original layer.
    #
    # tf.keras.Sequential(
    #   prune_masked_weight(tf.keras.layers.Dense(2, input_shape=(3,)))
    # )
    #
    # as opposed to
    #
    # tf.keras.Sequential(
    #   prune_masked_weight(tf.keras.layers.Dense(2), input_shape=(3,))
    # )
    #
    # Without this code, the pruning wrapper doesn't have an input
    # shape and being the first layer, this causes the model to not be
    # built. Being not built is confusing since the end-user has passed an
    # input shape.
    if not hasattr(self, '_batch_input_shape') and hasattr(
        layer, '_batch_input_shape'):
      self._batch_input_shape = self.layer._batch_input_shape
    #metrics.MonitorBoolGauge('prune_masked_weight_wrapper_usage').set(
    #    layer.__class__.__name__)

  def build(self, input_shape):
    super(PruneMaskedWeight, self).build(input_shape)

    weight_vars, mask_vars, = [], []

    # For each of the weights, add mask variables.
    for weight in self.layer.weights:
      # res2a_branch2a/kernel:0 -> kernel
      weight_name = weight.name.split('/')[-1].split(':')[0]
      mask = self.add_weight(
          weight_name + '_mask',
          shape=weight.shape,
          initializer=tf.keras.initializers.get('ones'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)

      weight_vars.append(weight)
      mask_vars.append(mask)

    pruning_vars = list(zip(weight_vars, mask_vars))
    # Create a pruning object
    self.pruning_obj = pruning_impl.Pruning(pruning_vars)

  def call(self, inputs, training=None, **kwargs):
    if training is None:
      training = K.learning_phase()

    # Always execute the op that performs weights = weights * mask
    # Relies on UpdatePruningStep callback to ensure the weights
    # are sparse after the final backpropagation.
    #
    # self.add_update does nothing during eager execution.
    self.add_update(self.pruning_obj.weight_mask_op())
    # TODO(evcu) remove this check after dropping py2 support. In py3 getargspec
    # is deprecated.
    if hasattr(inspect, 'getfullargspec'):
      args = inspect.getfullargspec(self.layer.call).args
    else:
      args = inspect.getargspec(self.layer.call).args
    # Propagate the training bool to the underlying layer if it accepts
    # training as an arg.
    if 'training' in args:
      return self.layer.call(inputs, training=training, **kwargs)

    return self.layer.call(inputs, **kwargs)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  def get_config(self):
    return super(PruneMaskedWeight, self).get_config()

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    deserialize_keras_object = keras.utils.deserialize_keras_object  # pylint: disable=g-import-not-at-top
    layer = keras.layers.deserialize(config.pop('layer'))
    config['layer'] = layer

    return cls(**config)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  #@property
  #def updates(self):
  #  return self.layer.updates + self._updates

  #@property
  #def losses(self):
  #  return self.layer.losses + self._losses

  #def get_weights(self):
  #  return self.layer.get_weights()

  #def set_weights(self, weights):
  #  self.layer.set_weights(weights)

def collect_prunable_layers(model):
  """Recursively collect the prunable layers in the model."""
  return [
      layer for layer in model.submodules
      if isinstance(layer, PruneMaskedWeight)
  ]
