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
# ==============================================================================
"""Fast finetune for quantized tf.keras models."""

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.engine import data_adapter
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils

keras = tf.keras
logger = common_utils.VAILogger


class SubModel(keras.Model):
  """A custom keras.Model class to support multi-optimizer."""

  def __init__(self, layer, act):
    super(SubModel, self).__init__()
    new_input = keras.Input(layer.get_input_shape_at(0)[1:], name='new_input')
    new_output = layer(new_input)
    if act.name != layer.name:
      new_output = act(new_output)
    self.sub_model = keras.Model(
        inputs=new_input, outputs=new_output, name='sub_model')
    self.w = self.sub_model.trainable_weights[0]
    if hasattr(layer.layer, 'use_bias') and layer.layer.use_bias:
      self.use_bias = True
      self.b = self.sub_model.trainable_weights[1]
    else:
      self.use_bias = False
      self.b = None
    self.loss_tracker = keras.metrics.Mean(name="loss")

  def call(self, x, training):
    return self.sub_model(x, training)

  def compile(self, loss, w_opt, b_opt=None):
    super(SubModel, self).compile()
    self.loss = loss
    self.w_opt = w_opt
    if self.use_bias:
      self.b_opt = b_opt

  @property
  def metrics(self):
    return [self.loss_tracker]

  def test_step(self, data):
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
    y_pred = self.call(x, training=False)
    loss_value = self.loss(y, y_pred, sample_weight)
    self.loss_tracker.update_state(loss_value)
    return {'loss': self.loss_tracker.result()}

  def train_step(self, data):
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
    with tf.GradientTape(persistent=True) as tape:
      y_pred = self.call(x, training=True)
      loss_value = self.loss(y, y_pred, sample_weight)

    # Get gradients of loss wrt the weights.
    w_gradients = tape.gradient(loss_value, self.w)
    self.w_opt.apply_gradients(zip([w_gradients], [self.w]))

    if self.use_bias:
      b_gradients = tape.gradient(loss_value, self.b)
      self.b_opt.apply_gradients(zip([b_gradients], [self.b]))

    del tape
    self.loss_tracker.update_state(loss_value)
    return {'loss': self.loss_tracker.result()}


def _get_layer_input(model, layer_name, input_data, batch_size, steps):
  """Get the predict result of layer's input."""
  target_layer = model.get_layer(layer_name)
  layer_model = tf.keras.Model(inputs=model.input, outputs=target_layer.input)
  return layer_model.predict(
      input_data,
      batch_size=batch_size,
      steps=steps,
      verbose=logger.debug_enabled())


def _get_layer_output(model, layer_name, input_data, batch_size, steps):
  """Get the predict result of layer's ouput."""
  target_layer = model.get_layer(layer_name)
  layer_model = tf.keras.Model(inputs=model.input, outputs=target_layer.output)
  return layer_model.predict(
      input_data,
      batch_size=batch_size,
      steps=steps,
      verbose=logger.debug_enabled())


def _get_module_io(model, input_name, output_name, input_data, batch_size,
                   steps):
  """Get the predict result of module's input and ouput."""
  input_layer = model.get_layer(input_name)
  output_layer = model.get_layer(output_name)
  layer_model = tf.keras.Model(
      inputs=model.input, outputs=[input_layer.input, output_layer.output])
  inputs, outputs = layer_model.predict(
      input_data,
      batch_size=batch_size,
      steps=steps,
      verbose=logger.debug_enabled())
  return inputs, outputs


def _eval_model_loss(quant_model, float_outputs, layer, dataset, batch_size,
                     steps):
  """Evaluate the mse loss of the float model and quant model for given layer."""
  quant_outputs = _get_layer_output(quant_model, layer.name, dataset,
                                    batch_size, steps)
  return tf.keras.losses.mean_squared_error(float_outputs,
                                            quant_outputs).numpy().mean()


def fast_finetune(quant_model, float_model, calib_dataset, calib_batch_size,
                  calib_steps, ft_epochs):
  """Do Ada-Quant Fast Finetuning."""

  target_modules = []
  for layer in quant_model.layers:
    if isinstance(
        layer,
        vitis_quantize_wrapper.QuantizeWrapper) and layer.trainable_weights:

      activation = layer.layer.activation.activation
      if isinstance(activation,
                    vitis_quantize_aware_activation.NoQuantizeActivation):
        act = layer.outbound_nodes[0].outbound_layer
      else:
        act = layer
      target_modules.append({'layer': layer, 'act': act})

  # Main loop
  for i, module in enumerate(target_modules):
    layer = module['layer']
    act = module['act']
    logger.info("Fast Finetuning({}/{}): {} -> {}".format(
        i + 1, len(target_modules), layer.layer.name, act.layer.name))

    logger.debug("Cache float inputs and outputs of layer: {} -> {}".format(
        layer.layer.name, act.layer.name))
    float_inputs, float_outputs = _get_module_io(
        model=float_model,
        input_name=layer.layer.name,
        output_name=act.layer.name,
        input_data=calib_dataset,
        batch_size=calib_batch_size,
        steps=calib_steps)

    sub_model = SubModel(layer, act)
    sub_model.compile(
        loss=tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE),
        w_opt=keras.optimizers.Adam(learning_rate=1e-5),
        b_opt=keras.optimizers.Adam(learning_rate=1e-3))

    logger.debug("Get initial loss...")
    best_loss = 0
    loss_type = 'layer'
    if loss_type == 'layer':
      best_loss = sub_model.evaluate(
          float_inputs,
          float_outputs,
          batch_size=calib_batch_size,
          steps=calib_steps,
          verbose=logger.debug_enabled())
    else:
      best_loss = _eval_model_loss(quant_model, float_outputs, layer,
                                   calib_dataset, calib_batch_size, calib_steps)
    best_params = sub_model.get_weights()

    for e in range(ft_epochs):
      logger.debug("Epoch {}/{}".format(e + 1, ft_epochs))
      sub_model.fit(
          float_inputs,
          float_outputs,
          batch_size=calib_batch_size,
          steps_per_epoch=calib_steps,
          epochs=1,
          verbose=logger.debug_enabled())

      logger.debug("Get new loss...")
      new_loss = 0
      if loss_type == 'layer':
        new_loss = sub_model.evaluate(
            float_inputs,
            float_outputs,
            batch_size=calib_batch_size,
            steps=calib_steps,
            verbose=logger.debug_enabled())
      else:
        new_loss = _eval_model_loss(quant_model, float_outputs, layer,
                                    calib_dataset, calib_batch_size,
                                    calib_steps)

      logger.debug("Best Loss: {:.2e}, new Loss: {:.2e}".format(
          best_loss, new_loss))
      if new_loss < best_loss:
        logger.debug("Update best loss: {:.2e} -> {:.2e}".format(
            best_loss, new_loss))
        best_loss = new_loss
        best_params = sub_model.get_weights()
      else:
        logger.debug("Revert best loss: {:.2e} -> {:.2e}".format(
            new_loss, best_loss))
        sub_model.set_weights(best_params)
        break

    if logger.debug_enabled():
      model_utils.save_model(quant_model, 'fast_ft_{}.h5'.format(i + 1),
                             './debug/')

  return
