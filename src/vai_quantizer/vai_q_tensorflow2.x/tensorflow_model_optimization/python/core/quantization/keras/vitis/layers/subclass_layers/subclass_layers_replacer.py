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
"""Replacer for HF transformers subclass layer."""

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger


def ExtractInputShapeFromInputData(inputs):
  """Extract input shape from input data."""
  if inputs is None:
    return None

  def _extract_shape(data):
    if isinstance(data, np.ndarray):
      return data.shape  # single tuple
    elif isinstance(data, tf.Tensor):
      return data.shape  # tf.TensorShape instance
    else:
      return None

  input_shape = _extract_shape(inputs)

  if input_shape is None:
    if isinstance(inputs, (list, tuple)):
      input_shape = []  # list
      for data in inputs:
        shape = _extract_shape(data)
        if shape is None:
          logger.error("Unsupported data format")
        else:
          input_shape.append(shape)
    elif isinstance(inputs, dict):
      input_shape = {}  # dict
      for name, data in inputs.items():
        shape = _extract_shape(data)
        if shape is None:
          logger.error("Unsupported data format")
        else:
          input_shape[name] = shape
    else:
      logger.error("Unsupported input format")

  return input_shape


class SubclassLayersReplacer(object):
  """Replacer for HF transformers subclass layer."""

  def __init__(self, config=None):
    self.config = config

    self.model = None

  def get_model_config(self, model):
    """get config from the model."""

    if not isinstance(model, tf.keras.Model):
      return None

    # From model
    if hasattr(model, "config"):
      self.config = model.config
      return model.config

    # From layer
    for layer in model.layers:
      if hasattr(layer, "config"):
        self.config = layer.config
        return layer.config

    return self.config

  def build_model(self, model, inputs):
    """build the model for initialize"""

    if not isinstance(model, tf.keras.Model):
      logger.error("Feed a keras model instance")
    elif inputs is None:
      logger.error("Feed a input data for model")

    #TODO: Those should be configurable in the future

    from .hf_transformers import hf_transformers_bert
    if (model.__class__.__name__ == "TFBertForQuestionAnswering"):
      self.model = hf_transformers_bert.TFBertForQuestionAnsweringNew(
          self.config)
    else:
      return None

    # You cannot build your model by calling `build`
    # if your layers do not support float type inputs.
    #self.model.build(ExtractInputShapeFromInputData(inputs))

    # Instead, in order to instantiate and build your
    # model, call your model on real tensor data.
    self.model.predict(inputs, batch_size=1, steps=1)
    # To avoid errors in next prediction after replacement,
    # we compile the model to clear model.predict_function cache.
    optimizer = 'adam' if (model.optimizer is None) else model.optimizer
    self.model.compile(optimizer=optimizer, loss=None)

    return self.model

  def apply(self, subclass):
    """Replace subclass with a quantizable one."""

    #TODO: Those should be configurable in the future

    if isinstance(subclass, keras.Model):
      # Try to replace the entire subclassed model with self.model
      if subclass.__class__.__name__ == "TFBertForQuestionAnswering":
        return self.model

    elif isinstance(subclass, keras.layers.Layer):
      # Try to replace subclassing layer with corresponding layer from self.model
      if subclass.__class__.__name__ != "TFBertMainLayer":
        return None

      for layer in self.model.layers:
        if layer.__class__.__name_ == "Quant" + subclass.__class__.__name__:
          return layer

    return None
