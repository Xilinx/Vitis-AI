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
"""Module containing sparsity code built on Keras abstractions."""
# pylint: disable=g-bad-import-order

from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_scope
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning

from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import PruningSummaries

from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PruningSchedule
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import ConstantSparsity
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay

from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer

# pylint: enable=g-bad-import-order
