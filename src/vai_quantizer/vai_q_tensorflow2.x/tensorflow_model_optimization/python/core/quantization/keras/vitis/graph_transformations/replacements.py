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
"""Quantize (subclassed) model by replacing layers with a quantizable one."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils.model_utils import (
    is_layer_wrapper, is_subclass_layer, is_subclass_model, get_sub_layers_dict,
    set_sub_layer_weights, show_sub_layers_tree)


class Replacement(object):
  """Quantize (subclassed) model by replacing sublayer with a quantizable one."""

  def __init__(self):
    """This worker get a target layer and return a quantizable one."""
    self.worker = None
    self.params = None

  def _complete_replace_attr(self, subclass, attributes_cache):
    """Replace layers by resetting all of the attributes."""
    if len(attributes_cache) == 0:
      return

    def _reset_build_compile_trackers(subclass):
      """Reset state trackers for model to allow building and compiling."""
      # Reset build state
      subclass.built = False
      subclass.inputs = None
      subclass.outputs = None

      # Reset compile state
      subclass._is_compiled = False
      if not tf.compat.v1.executing_eagerly_outside_functions():
        subclass._v1_compile_was_called = False
      subclass.optimizer = None

    setattr_tracking = subclass._setattr_tracking
    subclass._setattr_tracking = False

    subclass._self_tracked_trackables = []
    for name, attr in attributes_cache.items():
      setattr(subclass, name, attr)
      subclass._self_tracked_trackables.append(attr)

    if isinstance(subclass, tf.keras.Model):
      _reset_build_compile_trackers(subclass)
    subclass._setattr_tracking = setattr_tracking

  def _simple_replace_attr(self, subclass, attributes_cache):
    """Replace layers by set attributes."""
    for name, attr in attributes_cache.items():
      setattr(subclass, name, attr)

  def _traverse_sub_layers(self, subclass, method='simple'):
    """Look into sub layers for replacement."""
    sub_layers_dict = get_sub_layers_dict(subclass)

    attributes_cache = {}

    for name, layer in sub_layers_dict.items():
      if isinstance(layer, list):
        # Convert (ListWrapper or tuple) to list
        attr = list(getattr(subclass, name))

        for i, l in enumerate(layer):
          ql = self.worker(l, parent=subclass)

          if not ql is None:
            attr[i] = ql
          elif is_subclass_layer(l):
            self._traverse_sub_layers(l)

        # This attribute will unified to list
        attributes_cache[name] = attr
      else:
        qlayer = self.worker(layer, parent=subclass)

        if not qlayer is None:
          attributes_cache[name] = qlayer
        elif is_subclass_layer(layer):
          self._traverse_sub_layers(layer)

          if method == 'complete':
            attributes_cache[name] = layer

    if method == 'complete':
      self._complete_replace_attr(subclass, attributes_cache)
    else:
      self._simple_replace_attr(subclass, attributes_cache)
