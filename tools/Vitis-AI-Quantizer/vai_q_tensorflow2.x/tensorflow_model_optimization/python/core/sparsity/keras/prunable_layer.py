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
"""Abstract Base Class for making a custom keras layer prunable."""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class PrunableLayer(object):
  """Abstract Base Class for making your own keras layer prunable.

  Custom keras layers which want to add pruning should implement this class.

  """

  @abc.abstractmethod
  def get_prunable_weights(self):
    """Returns list of prunable weight tensors.

    All the weight tensors which the layer wants to be pruned during
    training must be returned by this method.

    Returns: List of weight tensors/kernels in the keras layer which must be
        pruned during training.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
