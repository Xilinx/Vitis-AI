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
"""Keras ClusterWeights wrapper API."""

import tensorflow as tf

from tensorflow.keras import initializers

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

keras = tf.keras
k = keras.backend
Layer = keras.layers.Layer
Wrapper = keras.layers.Wrapper


class ClusterWeights(Wrapper):
  """This wrapper augments a keras layer so that the weight tensor(s) can be
  clustered.

  This wrapper implements nearest neighbor clustering algorithm. This algorithm
  ensures that only a specified number of unique values are used in a weight
  tensor. This allows for certain types of hardware to benefit from advanced
  weight compression techniques and the associated reduction in model memory
  footprint and bandwidth.

  From practical standpoint this is implemented using a lookup table to hold the
  cluster centroid values during model training. The weight array is populated
  with 'gather' operation so that during back propagation the gradients can be
  calculated in a normal way. The lookup table is then adjusted using the
  cumulative gradient values for the weights that correspond to the same
  centroid.

  The number of unique values required as well as the way cluster centroids
  are initialized are passed in the wrapper's constructor.

  The initial values of cluster centroids are fine-tuned during the training.
  """

  def __init__(self,
               layer,
               number_of_clusters,
               cluster_centroids_init,
               **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `Cluster` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(layer)

    if isinstance(layer, clusterable_layer.ClusterableLayer):
      # A user-defined custom layer
      super(ClusterWeights, self).__init__(layer, **kwargs)
    elif clustering_registry.ClusteringRegistry.supports(layer):
      super(ClusterWeights, self).__init__(
          clustering_registry.ClusteringRegistry.make_clusterable(layer),
          **kwargs
      )
    else:
      raise ValueError(
          'Please initialize `Cluster` with a supported layer. Layers should '
          'either be a `ClusterableLayer` instance, or should be supported by '
          'the ClusteringRegistry. You passed: {input}'.format(
              input=layer.__class__
          )
      )

    if not isinstance(number_of_clusters, int):
      raise ValueError(
          "number_of_clusters must be an integer. Given: {}".format(
              number_of_clusters.__class__
          )
      )

    if number_of_clusters <= 1:
      raise ValueError(
          "number_of_clusters must be greater than 1. Given: {}".format(
              number_of_clusters
          )
      )

    self._track_trackable(layer, name='layer')

    # The way how cluster centroids will be initialized
    self.cluster_centroids_init = cluster_centroids_init

    # The number of cluster centroids
    self.number_of_clusters = number_of_clusters

    # Stores the pairs of weight names and references to their tensors
    self.clustered_vars = []

    # Stores references to class instances that implement different clustering
    # behaviour for different shapes of objects
    self.clustering_impl = {}

    # A dictionary that stores pairs of weight names and their respective
    # indices lookup tables
    self.pulling_indices_tf = {}

    # A dictionary that stores pairs of weight names and their respective
    # cluster centroids lookup tables
    self.cluster_centroids_tf = {}

    # A list for restoring the original order of weights later on, see the
    # comments in the code for usage explanations
    self.restore = []

    # setattr will remove the original weights from layer.weights array. We need
    # to memorise the original state of the array since saving the model relies
    # on the variables order in layer.weights rather than on values stored in
    # e.g. kernel/bias attributes of the layer object.
    self.gone_variables = []

    # If the input shape was specified, then we need to preserve this
    # information in the layer. If this info is not preserved, then the `built`
    # state will not be preserved between serializations.
    if not hasattr(self, '_batch_input_shape')\
        and hasattr(layer, '_batch_input_shape'):
      self._batch_input_shape = self.layer._batch_input_shape

  @staticmethod
  def _make_layer_name(layer):
    return '{}_{}'.format('cluster', layer.name)

  @staticmethod
  def _weight_name(name):
    """Extracts the weight name from the full TensorFlow variable name.

    For example, returns 'kernel' for 'dense_2/kernel:0'.

    Args:
      name: TensorFlow variable name.

    Returns:
      Extracted weight name.
    """
    return name.split(':')[0].split('/')[-1]

  def build(self, input_shape):
    super(ClusterWeights, self).build(input_shape)

    clusterable_weights = self.layer.get_clusterable_weights()

    # Map automatically assigned TF variable name (e.g. 'dense/kernel:0') to
    # provided human readable name (e.g. as in Dense(10).kernel)
    clusterable_weights_to_variables = {}

    for weight_name, weight in clusterable_weights:
      # If a variable appears in this loop, then it is going to be removed from
      # self._trainable_weights. We need to memorise what variables are going
      # away so that later we are able to restore them. We have to do this to
      # maintain the original order of the weights in the underlying layer.
      # Incorrect order results in the incorrect OPs weights configurations.

      # We can be sure that weight will be found in this array since the
      # variable is either in the self._trainable_weights or in
      # self._non_trainable_weights and self.weights is the result of
      # concatenation of those arrays
      original_index = self.layer.weights.index(weight)
      self.gone_variables.append(original_index)

      # Again, not sure if this is needed. Leaving for now.
      clusterable_weights_to_variables[self._weight_name(weight.name)] =\
          weight_name

      # Build initial cluster centroids for a given tensor. Factory returns a
      # class and we init an object immediately
      centroid_initializer = clustering_centroids.CentroidsInitializerFactory.\
          get_centroid_initializer(
              self.cluster_centroids_init
          )(weight, self.number_of_clusters)

      cluster_centroids = centroid_initializer.get_cluster_centroids()

      # Use k.batch_get_value since we need to initialize the variables with an
      # initial value taken from a Tensor object. For each weight there is a
      # different set of cluster centroids
      self.cluster_centroids_tf[weight_name] = self.add_weight(
          'cluster_centroids_tf',
          shape=(self.number_of_clusters,),
          dtype=weight.dtype,
          trainable=True,
          initializer=initializers.Constant(
              value=k.batch_get_value([cluster_centroids])[0]
          )
      )

      # There are vectorised implementations of look-ups, we use a new one for
      # different number of dimensions.
      clustering_impl_cls = clustering_registry.ClusteringLookupRegistry().\
          get_clustering_impl(self.layer, weight_name)
      self.clustering_impl[weight_name] = clustering_impl_cls(
          self.cluster_centroids_tf[weight_name]
      )

      # We find the nearest cluster centroids and store them so that ops can
      # build their weights upon it. These indices are calculated once and
      # stored forever. We use to make look-ups from self.cluster_centroids_tf
      pulling_indices = self.clustering_impl[weight_name].\
          get_pulling_indices(weight)
      self.pulling_indices_tf[weight_name] = self.add_weight(
          'pulling_indices_tf',
          shape=pulling_indices.shape,
          dtype=tf.int32,
          trainable=False,
          initializer=initializers.Constant(
              value=k.batch_get_value([pulling_indices])[0]
          )
      )

      # We store these pairs to easily update this variables later on
      self.clustered_vars.append((weight_name, weight))

    # We use currying here to get an updater which can be triggered at any time
    # in future and it would return the latest version of clustered weights
    def get_updater(for_weight_name):
      def fn():
        return self.clustering_impl[for_weight_name].get_clustered_weight(
            self.pulling_indices_tf[for_weight_name]
        )

      return fn

    # This will allow us to restore the order of weights later
    # This loop stores pairs of weight names and how to restore them

    for ct, weight in enumerate(self.layer.weights):
      name = self._weight_name(weight.name)
      if ct in self.gone_variables:
        # Again, not sure if this is needed
        weight_name = clusterable_weights_to_variables[name]
        self.restore.append((name, get_updater(weight_name)))
      else:
        self.restore.append((name, weight))

  def call(self, inputs):
    # Go through all tensors and replace them with their clustered copies.
    for weight_name, _ in self.clustered_vars:
      setattr(
          self.layer, weight_name,
          self.clustering_impl[weight_name].get_clustered_weight(
              self.pulling_indices_tf[weight_name]
          )
      )

    return self.layer.call(inputs)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  def get_config(self):
    base_config = super(ClusterWeights, self).get_config()
    config = {
        'number_of_clusters': self.number_of_clusters,
        'cluster_centroids_init': self.cluster_centroids_init,
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()

    number_of_clusters = config.pop('number_of_clusters')
    cluster_centroids_init = config.pop('cluster_centroids_init')
    config['number_of_clusters'] = number_of_clusters
    config['cluster_centroids_init'] = cluster_config.CentroidInitialization(
        cluster_centroids_init)

    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    layer = deserialize_layer(config.pop('layer'),
                              custom_objects=custom_objects)
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
    return self.layer.trainable_weights + self._trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses

  def get_weights(self):
    return self.layer.get_weights()

  def set_weights(self, weights):
    self.layer.set_weights(weights)
