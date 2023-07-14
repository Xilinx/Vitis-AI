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

import os
import collections
import copy
import datetime
import random
import string

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils.model_utils import (
    is_quantize_layer, is_layer_wrapper, is_subclass_layer, is_subclass_model,
    get_sub_layers_dict, set_sub_layer_weights, show_sub_layers_tree)

from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.subclass_layers import keras_layers_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers.subclass_layers import subclass_layers_replacer

from .replacements import Replacement


class SubclassReplacer(Replacement):
  """Replace the subclass with a quantizable one."""

  def __init__(self):
    """These parameters are used for replacer."""
    self.config = None

    self.replacer = subclass_layers_replacer.SubclassLayersReplacer()

  def _replace_subclass_layer(self, layer, parent):
    """Replace subclass layer with a quantizable one."""
    if parent is not None:
      pass

    if not is_subclass_layer(layer):
      return None

    qlayer = self.replacer.apply(layer)
    if not qlayer is None:
      set_sub_layer_weights(layer, qlayer)

    return qlayer

  def _recreate_subclass_model(self, model):
    """Recreate a new quantizable model."""
    if not is_subclass_model(model):
      return None

    qmodel = self.replacer.apply(model)
    if not qmodel is None:
      set_sub_layer_weights(model, qmodel)

    return qmodel

  def preprocess(self, model, inputs=None):
    """Do some preprocess."""

    # Get config from the model
    self.replacer.get_model_config(model)

    # Create quantizable model and build
    self.replacer.build_model(model, inputs)

    # Try to recreate a new quantizable model
    return self._recreate_subclass_model(model)

  def work(self, model, inputs):
    """Subclass replacer gets to work."""

    show_sub_layers_tree(model, caption_str='original subclasses')
    qmodel = self.preprocess(model, inputs)

    if not qmodel is None:
      show_sub_layers_tree(qmodel, caption_str='recreate subclasses')
      return qmodel  # recreate a new model, return directly

    self.worker = self._replace_subclass_layer
    self._traverse_sub_layers(model)
    show_sub_layers_tree(model, caption_str='replaced subclasses')

    return model


class SublayerWrapper(Replacement):
  """Wrap sublayers within a subclass layer for quantization."""

  def __init__(self, quantize_registry=None, mode="QCB"):
    """These parameters are needed by Vitis Wrapper."""
    self.quantize_registry = quantize_registry
    self.mode = mode

    self.wrapper = keras_layers_wrapper.KerasLayersWrapper(
        self.quantize_registry, self.mode)

  def _wrap_keras_layer(self, layer, parent=None):
    """Wrap keras layer for quantization."""
    if parent is not None and not parent.__class__.__name__.startswith("Quant"):
      return None

    if is_subclass_layer(layer) or is_quantize_layer(layer):
      return None

    qlayer = self.wrapper.apply(layer)
    if not qlayer is None:
      pass

    return qlayer

  def _remove_keras_layer(self, layer, parent=None):
    """Remove keras layer by replacing it with identity layer."""
    if parent is not None and not parent.__class__.__name__.startswith("Quant"):
      return None

    if is_subclass_layer(layer) or is_quantize_layer(layer):
      return None

    qlayer = self.wrapper.remove(layer)
    if not qlayer is None:
      pass

    return qlayer

  def _rename_sub_layers(self, subclass, hierarchy=''):
    """Rename leaf sublayers to avoid having duplicated names."""
    def _rename_sublayer(layer, hierarchy):
      if is_layer_wrapper(layer):
        layer.layer._name = hierarchy + layer.layer.name
      else:
        layer._name = hierarchy + layer.name

    sub_layers_dict = get_sub_layers_dict(subclass)

    for value in sub_layers_dict.values():
      if isinstance(value, list):
        for layer in value:
          if is_subclass_layer(layer):
            self._rename_sub_layers(layer, hierarchy + layer.name + '/')
          elif is_quantize_layer(layer):
            _rename_sublayer(layer, hierarchy)
      else:
        layer = value
        if is_subclass_layer(layer):
          self._rename_sub_layers(layer, hierarchy + layer.name + '/')
        elif is_quantize_layer(layer):
          _rename_sublayer(layer, hierarchy)

  def postprocess(self, model, inputs=None, **kwargs):
    """Do some postprocess."""

    # Remove the keras dropout sublayers
    if 'remove_dropout' in kwargs and kwargs['remove_dropout'] == True:
      self.worker = self._remove_keras_layer
      self._traverse_sub_layers(model)

    # Execute a prediction to initialize quantize layers variables, otherwise
    # we cannot copy weights from a quantized model to another, for example,
    # coping weights from a 'QCB' model to a 'QAT' model for initialization.
    if inputs is not None:
      model.predict(inputs, batch_size=1, steps=1)
      # Compile the model to clear model.predict_function cache.
      optimizer = 'adam' if (model.optimizer is None) else model.optimizer
      model.compile(optimizer=optimizer, loss=None)

    # Rename the quantized sublayers to discriminate them
    if 'rename_sublayers' in kwargs and kwargs['rename_sublayers'] == True:
      self._rename_sub_layers(model, model.name + '/')

    return model

  def work(self, model, inputs, **kwargs):
    """Sublayer wrapper gets to work."""
    self.worker = self._wrap_keras_layer
    self._traverse_sub_layers(model)
    show_sub_layers_tree(model, caption_str='wrapped sublayers')

    if 'remove_dropout' not in kwargs or kwargs['remove_dropout'] == True:
      self.worker = self._remove_keras_layer
      self._traverse_sub_layers(model)
      show_sub_layers_tree(model, caption_str='removed sublayers')

    self.postprocess(model, inputs=inputs, rename_sublayers=True)
    show_sub_layers_tree(model, caption_str='renamed sublayers', show_leaf=True)

    return model
