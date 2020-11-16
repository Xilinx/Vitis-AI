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
"""Registry responsible for built-in keras classes."""

import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer

layers = tf.keras.layers


class PruneRegistry(object):
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
      TensorFlowOpLayer: [],
  }

  _RNN_CELLS_WEIGHTS_MAP = {
      # Allowlist via compat.v1 and compat.v2 to support legacy TensorFlow 2.X
      # behavior where the v2 RNN uses the v1 RNNCell instead of the v2 RNNCell.
      # See b/145939875 for details.
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

  _RNN_LAYERS = frozenset({
      layers.GRU,
      layers.LSTM,
      layers.RNN,
      layers.SimpleRNN,
  })

  _RNN_CELLS_STR = ', '.join(str(_RNN_CELLS_WEIGHTS_MAP.keys()))

  _RNN_CELL_ERROR_MSG = (
      'RNN Layer {} contains cell type {} which is either not supported or does'
      'not inherit PrunableLayer. The cell must be one of {}, or implement '
      'PrunableLayer.')

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
            and not isinstance(cell, prunable_layer.PrunableLayer):
          return False
      return True

    return False

  @staticmethod
  def _get_rnn_cells(rnn_layer):
    if isinstance(rnn_layer.cell, layers.StackedRNNCells):
      return rnn_layer.cell.cells
    else:
      return [rnn_layer.cell]

  @classmethod
  def _is_rnn_layer(cls, layer):
    return layer.__class__ in cls._RNN_LAYERS

  @classmethod
  def _weight_names(cls, layer):
    return cls._LAYERS_WEIGHTS_MAP[layer.__class__]

  @classmethod
  def make_prunable(cls, layer):
    """Modifies a built-in layer object to support pruning.

    Args:
      layer: layer to modify for support.

    Returns:
      The modified layer object.

    """

    if not cls.supports(layer):
      raise ValueError('Layer ' + str(layer.__class__) + ' is not supported.')

    def get_prunable_weights():
      return [getattr(layer, weight) for weight in cls._weight_names(layer)]

    def get_prunable_weights_rnn():  # pylint: disable=missing-docstring
      def get_prunable_weights_rnn_cell(cell):
        if cell.__class__ in cls._RNN_CELLS_WEIGHTS_MAP:
          return [getattr(cell, weight)
                  for weight in cls._RNN_CELLS_WEIGHTS_MAP[cell.__class__]]

        if isinstance(cell, prunable_layer.PrunableLayer):
          return cell.get_prunable_weights()

        raise ValueError(cls._RNN_CELL_ERROR_MSG.format(
            layer.__class__, cell.__class__, cls._RNN_CELLS_WEIGHTS_MAP.keys()))

      prunable_weights = []
      for rnn_cell in cls._get_rnn_cells(layer):
        prunable_weights.extend(get_prunable_weights_rnn_cell(rnn_cell))
      return prunable_weights

    if cls._is_rnn_layer(layer):
      layer.get_prunable_weights = get_prunable_weights_rnn
    else:
      layer.get_prunable_weights = get_prunable_weights

    return layer
