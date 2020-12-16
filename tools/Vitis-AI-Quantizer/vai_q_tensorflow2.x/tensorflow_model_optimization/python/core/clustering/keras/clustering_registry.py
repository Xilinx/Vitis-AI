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
"""Registry responsible for built-in keras classes."""

import abc
import six
import tensorflow as tf

from tensorflow.keras import layers

from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer


@six.add_metaclass(abc.ABCMeta)
class AbstractClusteringAlgorithm(object):
  """
  The reason to have an abstract class here is to be able to implement highly
  efficient vectorised look-ups.

  We do not utilise looping for that purpose, instead we `smartly` reshape and
  tile arrays. The trade-off is that we are potentially using way more memory
  than we would have if looping is used.

  Each class that inherits from this class is supposed to implement a particular
  lookup function for a certain shape.

  For example, look-ups for 2D table will be different in the case of 3D.
  """

  def __init__(self, clusters_centroids):
    """
    For generating clustered tensors we will need two things: cluster centroids
    and the final shape tensor must have.
    :param clusters_centroids: An array of shape (N,) that contains initial
      values of clusters centroids.
    """
    self.cluster_centroids = clusters_centroids

  @abc.abstractmethod
  def get_pulling_indices(self, weight):
    """
    Takes a weight(can be 1D, 2D or ND) and creates tf.int32 array of the same
    shape that will hold indices of cluster centroids clustered arrays elements
    will be pulled from.

    In the current setup pulling indices are meant to be created once and used
    everywhere
    :param weight: ND array of weights. For each weight in this array the
      closest cluster centroids is found.
    :return: ND array of the same shape as `weight` parameter of the type
      tf.int32. The returned array contain weight lookup indices
    """
    pass

  def get_clustered_weight(self, pulling_indices):
    """
    Takes an array with integer number that represent lookup indices and forms a
    new array according to the given indices.
    :param pulling_indices: an array of indices used for lookup.
    :return: array with the same shape as `pulling_indices`. Each array element
      is a member of self.cluster_centroids
    """
    return tf.reshape(
        tf.gather(self.cluster_centroids,
                  tf.reshape(pulling_indices, shape=(-1,))),
        pulling_indices.shape
    )


class ConvolutionalWeightsCA(AbstractClusteringAlgorithm):
  """
  Look-ups for convolutional kernels, e.g. tensors with shape [B,W,H,C]
  """

  def get_pulling_indices(self, weight):
    clst_num = self.cluster_centroids.shape[0]
    tiled_weights = tf.tile(tf.expand_dims(weight, 4), [1, 1, 1, 1, clst_num])

    # Do the ugly reshape to the clustering points
    tiled_cluster_centroids = tf.stack(
        [tf.tile(tf.stack(
            [tf.reshape(self.cluster_centroids, [1, 1, clst_num])] *
            weight.shape[-2], axis=2),
                 [weight.shape[0], weight.shape[1], 1, 1])] * weight.shape[-1],
        axis=3)

    # We find the nearest cluster centroids and store them so that ops can build
    # their kernels upon it
    pulling_indices = tf.argmin(
        tf.abs(tiled_weights - tiled_cluster_centroids), axis=4
    )

    return pulling_indices


class DenseWeightsCA(AbstractClusteringAlgorithm):
  """
  Dense layers store their weights in 2D tables, i.e. tensor of the shape [U, D]
  """

  def get_pulling_indices(self, weight):
    clst_num = self.cluster_centroids.shape[0]
    tiled_weights = tf.tile(tf.expand_dims(weight, axis=2), [1, 1, clst_num])
    tiled_cluster_centroids = tf.tile(
        tf.reshape(self.cluster_centroids, [1, 1, clst_num]),
        [weight.shape[0], weight.shape[1], 1])

    # We find the nearest cluster centroids and store them so that ops can build
    # their kernels upon it
    pulling_indices = tf.argmin(tf.abs(tiled_weights - tiled_cluster_centroids),
                                axis=2)

    return pulling_indices


class BiasWeightsCA(AbstractClusteringAlgorithm):
  """
  Biases are stored as tensors of rank 0
  """

  def get_pulling_indices(self, weight):
    clst_num = self.cluster_centroids.shape[0]
    tiled_weights = tf.tile(tf.expand_dims(weight, axis=1), [1, clst_num])
    tiled_cluster_centroids = tf.tile(
        tf.reshape(self.cluster_centroids, [1, clst_num]), [weight.shape[0], 1])

    pulling_indices = tf.argmin(tf.abs(tiled_weights - tiled_cluster_centroids),
                                axis=1)

    return pulling_indices


class ClusteringLookupRegistry(object):
  """
  The keys represent built-in keras layers and the values represent the
  strategy accoding to which clustering will be done.
  If the key is not present in the map, that means that there is nothing to
  work on, or the strategy is not currently supported
  """
  _LAYERS_RESHAPE_MAP = {
      layers.Conv1D: {'kernel': ConvolutionalWeightsCA},
      layers.Conv2D: {'kernel': ConvolutionalWeightsCA},
      layers.Conv2DTranspose: {'kernel': ConvolutionalWeightsCA},
      layers.Conv3D: {'kernel': ConvolutionalWeightsCA},
      layers.Conv3DTranspose: {'kernel': ConvolutionalWeightsCA},
      layers.SeparableConv1D: {'pointwise_kernel': ConvolutionalWeightsCA},
      layers.SeparableConv2D: {'pointwise_kernel': ConvolutionalWeightsCA},
      layers.Dense: {'kernel': DenseWeightsCA},
      layers.Embedding: {'embeddings': DenseWeightsCA},
      layers.LocallyConnected1D: {'kernel': ConvolutionalWeightsCA},
      layers.LocallyConnected2D: {'kernel': ConvolutionalWeightsCA},
  }

  @classmethod
  def register_new_implementation(cls, new_impl):
    """
    For custom user-defined objects define the way how clusterable weights
    are going to be formed. If weights are any of these, 1D,2D or 4D, please
    consider using existing implementations: BiasWeightsCA,
    ConvolutionalWeightsCA and DenseWeightsCA.

    :param new_impl: dictionary. Keys are classes and values are dictionaries.
      The latter have strings as keys and values are classes inherited from
      AbstractClusteringAlgorithm. Normally, the set keys of the latter
      dictionaries should match the set of clusterable weights names for the
      layer.
    :return: None
    """
    if not isinstance(new_impl, dict):
      raise TypeError("new_impl must be a dictionary")
    for k, v in new_impl.items():
      if not isinstance(v, dict):
        raise TypeError(
            "Every value of new_impl must be a dictionary. Item for key {key} "
            "has class {vclass}".format(
                key=k,
                vclass=v
            )
        )

    cls._LAYERS_RESHAPE_MAP.update(new_impl)

  @classmethod
  def get_clustering_impl(cls, layer, weight_name):
    """
    Returns a certain reshape/lookup implementation for a given array
    :param layer: A layer that is being clustered
    :param weight_name: concrete weight name to be clustered.
    :return: a concrete implementation of a lookup algorithm
    """
    if not layer.__class__ in cls._LAYERS_RESHAPE_MAP:
      raise ValueError(
          "Class {given_class} has not been registerd in the"
          "ClusteringLookupRegistry. Use ClusteringLookupRegistry."
          "register_new_implemenetation to fix this.".format(
              given_class=layer.__class__
          )
      )
    if weight_name not in cls._LAYERS_RESHAPE_MAP[layer.__class__]:
      raise ValueError(
          "Weight with the name '{given_weight_name}' for class {given_class} "
          "has not been registerd in the ClusteringLookupRegistry. Use "
          "ClusteringLookupRegistry.register_new_implemenetation "
          "to fix this.".format(
              given_class=layer.__class__,
              given_weight_name=weight_name
          )
      )
    # Different weights will have different shapes hence there is double hash
    # map lookup.
    return cls._LAYERS_RESHAPE_MAP[layer.__class__][weight_name]


class ClusteringRegistry(object):
  """Registry responsible for built-in keras layers."""

  # The keys represent built-in keras layers and the values represent the
  # the variables within the layers which hold the kernel weights. This
  # allows the wrapper to access and modify the weights.
  _LAYERS_WEIGHTS_MAP = {
      layers.ELU: [],
      layers.LeakyReLU: [],
      layers.ReLU: [],
      layers.Softmax: [],
      layers.ThresholdedReLU: [],
      layers.Conv1D: ['kernel'],
      layers.Conv2D: ['kernel'],
      layers.Conv2DTranspose: ['kernel'],
      layers.Conv3D: ['kernel'],
      layers.Conv3DTranspose: ['kernel'],
      layers.Cropping1D: [],
      layers.Cropping2D: [],
      layers.Cropping3D: [],
      layers.DepthwiseConv2D: [],
      layers.SeparableConv1D: ['pointwise_kernel'],
      layers.SeparableConv2D: ['pointwise_kernel'],
      layers.UpSampling1D: [],
      layers.UpSampling2D: [],
      layers.UpSampling3D: [],
      layers.ZeroPadding1D: [],
      layers.ZeroPadding2D: [],
      layers.ZeroPadding3D: [],
      layers.Activation: [],
      layers.ActivityRegularization: [],
      layers.Dense: ['kernel'],
      layers.Dropout: [],
      layers.Flatten: [],
      layers.Lambda: [],
      layers.Masking: [],
      layers.Permute: [],
      layers.RepeatVector: [],
      layers.Reshape: [],
      layers.SpatialDropout1D: [],
      layers.SpatialDropout2D: [],
      layers.SpatialDropout3D: [],
      layers.Embedding: ['embeddings'],
      layers.LocallyConnected1D: ['kernel'],
      layers.LocallyConnected2D: ['kernel'],
      layers.Add: [],
      layers.Average: [],
      layers.Concatenate: [],
      layers.Dot: [],
      layers.Maximum: [],
      layers.Minimum: [],
      layers.Multiply: [],
      layers.Subtract: [],
      layers.AlphaDropout: [],
      layers.GaussianDropout: [],
      layers.GaussianNoise: [],
      layers.BatchNormalization: [],
      layers.LayerNormalization: [],
      layers.AveragePooling1D: [],
      layers.AveragePooling2D: [],
      layers.AveragePooling3D: [],
      layers.GlobalAveragePooling1D: [],
      layers.GlobalAveragePooling2D: [],
      layers.GlobalAveragePooling3D: [],
      layers.GlobalMaxPooling1D: [],
      layers.GlobalMaxPooling2D: [],
      layers.GlobalMaxPooling3D: [],
      layers.MaxPooling1D: [],
      layers.MaxPooling2D: [],
      layers.MaxPooling3D: [],
  }

  _RNN_CELLS_WEIGHTS_MAP = {
      # NOTE: RNN cells are added via compat.v1 and compat.v2 to support legacy
      # TensorFlow 2.X behavior where the v2 RNN uses the v1 RNNCell instead of
      # the v2 RNNCell.
      tf.compat.v1.keras.layers.GRUCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v2.keras.layers.GRUCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v1.keras.layers.LSTMCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v2.keras.layers.LSTMCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v1.keras.experimental.PeepholeLSTMCell: [
          'kernel', 'recurrent_kernel'
      ],
      tf.compat.v2.keras.experimental.PeepholeLSTMCell: [
          'kernel', 'recurrent_kernel'
      ],
      tf.compat.v1.keras.layers.SimpleRNNCell: ['kernel', 'recurrent_kernel'],
      tf.compat.v2.keras.layers.SimpleRNNCell: ['kernel', 'recurrent_kernel'],
  }

  _RNN_LAYERS = {
      layers.GRU,
      layers.LSTM,
      layers.RNN,
      layers.SimpleRNN,
  }

  _RNN_CELLS_STR = ', '.join(str(_RNN_CELLS_WEIGHTS_MAP.keys()))

  _RNN_CELL_ERROR_MSG = (
      'RNN Layer {} contains cell type {} which is either not supported or does'
      'not inherit ClusterableLayer. The cell must be one of {}, or implement '
      'ClusterableLayer.'
  )

  @classmethod
  def supports(cls, layer):
    """Returns whether the registry supports this layer type.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if layer.__class__ in cls._LAYERS_WEIGHTS_MAP:
      return True

    if layer.__class__ in cls._RNN_LAYERS:
      for cell in cls._get_rnn_cells(layer):
        if cell.__class__ not in cls._RNN_CELLS_WEIGHTS_MAP \
                and not isinstance(cell, clusterable_layer.ClusterableLayer):
          return False
      return True

    return False

  @staticmethod
  def _get_rnn_cells(rnn_layer):
    if isinstance(rnn_layer.cell, layers.StackedRNNCells):
      return rnn_layer.cell.cells
    return [rnn_layer.cell]

  @classmethod
  def _is_rnn_layer(cls, layer):
    return layer.__class__ in cls._RNN_LAYERS

  @classmethod
  def _weight_names(cls, layer):
    return cls._LAYERS_WEIGHTS_MAP[layer.__class__]

  @classmethod
  def make_clusterable(cls, layer):
    """Modifies a built-in layer object to support clustering.

    Args:
      layer: layer to modify for support.

    Returns:
      The modified layer object.

    """

    if not cls.supports(layer):
      raise ValueError('Layer ' + str(layer.__class__) + ' is not supported.')

    def get_clusterable_weights():
      return [(weight, getattr(layer, weight)) for weight in
              cls._weight_names(layer)]

    def get_clusterable_weights_rnn():  # pylint: disable=missing-docstring
      def get_clusterable_weights_rnn_cell(cell):
        if cell.__class__ in cls._RNN_CELLS_WEIGHTS_MAP:
          return [(weight, getattr(cell, weight))
                  for weight in cls._RNN_CELLS_WEIGHTS_MAP[cell.__class__]]

        if isinstance(cell, clusterable_layer.ClusterableLayer):
          return cell.get_clusterable_weights()

        raise ValueError(cls._RNN_CELL_ERROR_MSG.format(
            layer.__class__, cell.__class__, cls._RNN_CELLS_WEIGHTS_MAP.keys()))

      clusterable_weights = []
      for rnn_cell in cls._get_rnn_cells(layer):
        clusterable_weights.extend(get_clusterable_weights_rnn_cell(rnn_cell))
      return clusterable_weights

    if cls._is_rnn_layer(layer):
      layer.get_clusterable_weights = get_clusterable_weights_rnn
    else:
      layer.get_clusterable_weights = get_clusterable_weights

    return layer
