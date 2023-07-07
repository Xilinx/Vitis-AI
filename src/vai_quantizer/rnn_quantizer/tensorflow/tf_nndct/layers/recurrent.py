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
#

"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion

from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.training.tracking import data_structures

from tf_nndct.utils import tf_utils as _tf_utils

if _tf_utils.tf_version() >= LooseVersion('2.6'):
  from keras.utils import tf_utils
  from keras.layers import recurrent
else:
  from tensorflow.python.keras.utils import tf_utils
  from tensorflow.python.keras.layers import recurrent

class LSTMCell(recurrent.DropoutRNNCellMixin, keras_layers.Layer):
  """Cell class for the LSTM layer.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.

  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(LSTMCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    # if self.recurrent_dropout != 0 and implementation != 1:
    #   logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
    #   self.implementation = 1
    # else:
    #   self.implementation = implementation

    # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
    # and fixed after 2.7.16. Converting the state_size to wrapper around
    # NoDependency(), so that the base_layer.__setattr__ will not convert it to
    # ListWrapper. Down the stream, self.states will be a list since it is
    # generated from nest.map_structure with list, and tuple(list) will work
    # properly.
    self.state_size = data_structures.NoDependency([self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def _build_v200(self, input_shape):
    input_dim = input_shape[-1]
    kernel_args = {
        'shape': (input_dim, self.units),
        'initializer': self.kernel_initializer,
        'regularizer': self.kernel_regularizer,
        'constraint': self.kernel_constraint,
    }
    self.kernel_i = self.add_weight(name='kernel_i', **kernel_args)
    self.kernel_f = self.add_weight(name='kernel_f', **kernel_args)
    self.kernel_c = self.add_weight(name='kernel_c', **kernel_args)
    self.kernel_o = self.add_weight(name='kernel_o', **kernel_args)

    recurrent_args = {
        'shape': (self.units, self.units),
        'initializer': self.recurrent_initializer,
        'regularizer': self.recurrent_regularizer,
        'constraint': self.recurrent_constraint,
    }
    self.recurrent_kernel_i = self.add_weight(
        name='recurrent_kernel_i', **recurrent_args)
    self.recurrent_kernel_f = self.add_weight(
        name='recurrent_kernel_f', **recurrent_args)
    self.recurrent_kernel_c = self.add_weight(
        name='recurrent_kernel_c', **recurrent_args)
    self.recurrent_kernel_o = self.add_weight(
        name='recurrent_kernel_o', **recurrent_args)

    if self.use_bias:
      bias_initializer = self.bias_initializer
      if self.unit_forget_bias:
        forget_bias_initializer = initializers.Ones
      else:
        forget_bias_initializer = bias_initializer

      bias_args = {
          'shape': (self.units,),
          'regularizer': self.bias_regularizer,
          'constraint': self.bias_constraint,
      }
      self.bias_i = self.add_weight(
          name='bias_i', initializer=bias_initializer, **bias_args)
      self.bias_f = self.add_weight(
          name='bias_f', initializer=forget_bias_initializer, **bias_args)
      self.bias_c = self.add_weight(
          name='bias_c', initializer=bias_initializer, **bias_args)
      self.bias_o = self.add_weight(
          name='bias_o', initializer=bias_initializer, **bias_args)
    else:
      self.bias_i = None
      self.bias_f = None
      self.bias_c = None
      self.bias_o = None
    self.built = True

  @tf_utils.shape_type_conversion
  def _build_v210(self, input_shape):
    input_dim = input_shape[-1]
    args = {
        'shape': (input_dim, self.units),
        'initializer': self.kernel_initializer,
        'regularizer': self.kernel_regularizer,
        'constraint': self.kernel_constraint,
        'caching_device': default_caching_device
    }

    self.kernel_i = self.add_weight(name='kernel_i', **args)
    self.kernel_f = self.add_weight(name='kernel_f', **args)
    self.kernel_c = self.add_weight(name='kernel_c', **args)
    self.kernel_o = self.add_weight(name='kernel_o', **args)

    self.recurrent_kernel_i = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel_i',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel_f = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel_f',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel_c = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel_c',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel_o = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel_o',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)
    if self.use_bias:
      if self.unit_forget_bias:

        # TODO(yuwang): Split to 4 initers.
        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias_i = self.add_weight(
          shape=(self.units,),
          name='bias_i',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
      self.bias_f = self.add_weight(
          shape=(self.units,),
          name='bias_f',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
      self.bias_c = self.add_weight(
          shape=(self.units,),
          name='bias_c',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
      self.bias_o = self.add_weight(
          shape=(self.units,),
          name='bias_o',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias_i = None
      self.bias_f = None
      self.bias_c = None
      self.bias_o = None
    self.built = True

  def build(self, input_shape):
    input_dim = input_shape[-1]
    tf_version = _tf_utils.tf_version()

    if tf_version >= '2.1.0':
      default_caching_device = _caching_device(self)

    if tf_version < '2.1.0':
      kernel_args = {
          'shape': (input_dim, self.units),
          'initializer': self.kernel_initializer,
          'regularizer': self.kernel_regularizer,
          'constraint': self.kernel_constraint,
      }
      recurrent_args = {
          'shape': (self.units, self.units),
          'initializer': self.recurrent_initializer,
          'regularizer': self.recurrent_regularizer,
          'constraint': self.recurrent_constraint,
      }
    else:
      # There is an addtional 'default_caching_device' argument after tf 2.1
      kernel_args = {
          'shape': (input_dim, self.units),
          'initializer': self.kernel_initializer,
          'regularizer': self.kernel_regularizer,
          'constraint': self.kernel_constraint,
          'caching_device': default_caching_device
      }
      recurrent_args = {
          'shape': (self.units, self.units),
          'initializer': self.recurrent_initializer,
          'regularizer': self.recurrent_regularizer,
          'constraint': self.recurrent_constraint,
          'caching_device': default_caching_device
      }

    # Split kernel/recurrent_kernel/bias to 4 parts as RNN compiler
    # requires this.
    self.kernel_i = self.add_weight(name='kernel_i', **kernel_args)
    self.kernel_f = self.add_weight(name='kernel_f', **kernel_args)
    self.kernel_c = self.add_weight(name='kernel_c', **kernel_args)
    self.kernel_o = self.add_weight(name='kernel_o', **kernel_args)

    self.recurrent_kernel_i = self.add_weight(
        name='recurrent_kernel_i', **recurrent_args)
    self.recurrent_kernel_f = self.add_weight(
        name='recurrent_kernel_f', **recurrent_args)
    self.recurrent_kernel_c = self.add_weight(
        name='recurrent_kernel_c', **recurrent_args)
    self.recurrent_kernel_o = self.add_weight(
        name='recurrent_kernel_o', **recurrent_args)

    if self.use_bias:
      bias_initializer = self.bias_initializer
      if self.unit_forget_bias:
        forget_bias_initializer = initializers.get('ones')
      else:
        forget_bias_initializer = bias_initializer

      if tf_version < '2.1.0':
        bias_args = {
            'shape': (self.units,),
            'regularizer': self.bias_regularizer,
            'constraint': self.bias_constraint,
        }
      else:
        bias_args = {
            'shape': (self.units,),
            'regularizer': self.bias_regularizer,
            'constraint': self.bias_constraint,
            'caching_device': default_caching_device
        }
      self.bias_i = self.add_weight(
          name='bias_i', initializer=bias_initializer, **bias_args)
      self.bias_f = self.add_weight(
          name='bias_f', initializer=forget_bias_initializer, **bias_args)
      self.bias_c = self.add_weight(
          name='bias_c', initializer=bias_initializer, **bias_args)
      self.bias_o = self.add_weight(
          name='bias_o', initializer=bias_initializer, **bias_args)
    else:
      self.bias_i = None
      self.bias_f = None
      self.bias_c = None
      self.bias_o = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(
        x_f +
        K.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    return c, o

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if 0 < self.dropout < 1.:
      inputs_i = inputs * dp_mask[0]
      inputs_f = inputs * dp_mask[1]
      inputs_c = inputs * dp_mask[2]
      inputs_o = inputs * dp_mask[3]
    else:
      inputs_i = inputs
      inputs_f = inputs
      inputs_c = inputs
      inputs_o = inputs
    # k_i, k_f, k_c, k_o = array_ops.split(
    #     self.kernel, num_or_size_splits=4, axis=1)
    x_i = K.dot(inputs_i, self.kernel_i)
    x_f = K.dot(inputs_f, self.kernel_f)
    x_c = K.dot(inputs_c, self.kernel_c)
    x_o = K.dot(inputs_o, self.kernel_o)
    if self.use_bias:
      # b_i, b_f, b_c, b_o = array_ops.split(
      #     self.bias, num_or_size_splits=4, axis=0)
      x_i = K.bias_add(x_i, self.bias_i)
      x_f = K.bias_add(x_f, self.bias_f)
      x_c = K.bias_add(x_c, self.bias_c)
      x_o = K.bias_add(x_o, self.bias_o)

    if 0 < self.recurrent_dropout < 1.:
      h_tm1_i = h_tm1 * rec_dp_mask[0]
      h_tm1_f = h_tm1 * rec_dp_mask[1]
      h_tm1_c = h_tm1 * rec_dp_mask[2]
      h_tm1_o = h_tm1 * rec_dp_mask[3]
    else:
      h_tm1_i = h_tm1
      h_tm1_f = h_tm1
      h_tm1_c = h_tm1
      h_tm1_o = h_tm1
    # x = (x_i, x_f, x_c, x_o)
    # h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
    # c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)

    i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))
    f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.recurrent_kernel_f))
    c = f * c_tm1 + i * self.activation(x_c +
                                        K.dot(h_tm1_c, self.recurrent_kernel_c))
    o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o))

    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
    }
    base_config = super(LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(
        _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))

class LSTM(keras_layers.RNN):
  """Long Short-Term Memory layer - Hochreiter 1997.

   Note that this cell is not optimized for performance on GPU. Please use
  `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: sigmoid (`sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs..
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications.
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):

    cell = LSTMCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True))
    keras_layers.RNN.__init__(
        self,
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [keras_layers.InputSpec(ndim=3)]

_generate_zero_filled_state_for_cell = recurrent._generate_zero_filled_state_for_cell

if _tf_utils.tf_version() < LooseVersion('2.1.0'):
  _caching_device = None
else:
  _caching_device = recurrent._caching_device
