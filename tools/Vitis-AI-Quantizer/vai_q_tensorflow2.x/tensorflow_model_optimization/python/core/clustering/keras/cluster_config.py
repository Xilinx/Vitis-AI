# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Configuration classes for clustering."""

import enum


class CentroidInitialization(str, enum.Enum):
  """Specifies how the cluster centroids should be initialized.

  * `LINEAR`: Cluster centroids are evenly spaced between the minimum and
      maximum values of a given weight tensor.
  * `RANDOM`: Centroids are sampled using the uniform distribution between the
      minimum and maximum weight values in a given layer.
  * `DENSITY_BASED`: Density-based sampling obtained as follows: first a
       cumulative distribution function is built for the weights, then the Y
       axis is evenly spaced into as many regions as many clusters we want to
       have. After this the corresponding X values are obtained and used to
       initialize the clusters centroids.
  """
  LINEAR = "LINEAR"
  RANDOM = "RANDOM"
  DENSITY_BASED = "DENSITY_BASED"
