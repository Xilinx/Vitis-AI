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
"""Clusters centroids initialization API for Keras clustering API."""

import abc
import six
import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config

k = tf.keras.backend
CentroidInitialization = cluster_config.CentroidInitialization

@six.add_metaclass(abc.ABCMeta)
class AbstractCentroidsInitialisation:
  """
  Abstract base class for implementing different cluster centroid
  initialisation algorithms. Must be initialised with a reference to the
  weights and implement the single method below.
  """

  def __init__(self, weights, number_of_clusters):
    self.weights = weights
    self.number_of_clusters = number_of_clusters

  @abc.abstractmethod
  def get_cluster_centroids(self):
    pass


class LinearCentroidsInitialisation(AbstractCentroidsInitialisation):
  """
  Spaces cluster centroids evenly in the interval [min(weights), max(weights)]
  """

  def get_cluster_centroids(self):
    weight_min = tf.reduce_min(self.weights)
    weight_max = tf.reduce_max(self.weights)
    cluster_centroids = tf.linspace(weight_min,
                                    weight_max,
                                    self.number_of_clusters)
    return cluster_centroids


class RandomCentroidsInitialisation(AbstractCentroidsInitialisation):
  """
  Sample centroids randomly and uniformly from the interval
  [min(weights), max(weights)]
  """

  def get_cluster_centroids(self):
    weight_min = tf.reduce_min(self.weights)
    weight_max = tf.reduce_max(self.weights)
    cluster_centroids = tf.random.uniform(shape=(self.number_of_clusters,),
                                          minval=weight_min,
                                          maxval=weight_max,
                                          dtype=self.weights.dtype)
    return cluster_centroids


class TFLinearEquationSolver:
  """
  Solves a linear equantion y=ax+b for either y or x.

  The line equation is defined with two points (x1, y1) and (x2,y2)
  """

  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2

    # Writing params for y=ax+b
    self.a = (y2 - y1) / tf.maximum(x2 - x1, 0.001)
    self.b = y1 - x1 * ((y2 - y1) / tf.maximum(x2 - x1, 0.001))

  def solve_for_x(self, y):
    """
    For a given y value, find x at which linear function takes value y
    :param y: the y value
    :return: the corresponding x value
    """
    return (y - self.b) / self.a

  def solve_for_y(self, x):
    """
    For a given x value, find y at which linear function takes value x
    :param x: the x value
    :return: the corresponding y value
    """
    return self.a * x + self.b


class TFCumulativeDistributionFunction:
  """
  Takes an array and builds cumulative distribution function(CDF)
  """

  def __init__(self, weights):
    self.weights = weights

  def get_cdf_value(self, given_weight):
    mask = tf.less_equal(self.weights, given_weight)
    less_than = tf.cast(tf.math.count_nonzero(mask), dtype=tf.float32)
    return less_than / tf.size(self.weights, out_type=tf.float32)


class DensityBasedCentroidsInitialisation(AbstractCentroidsInitialisation):
  """
  This initialisation means that we build a cumulative distribution
  function(CDF), then linearly space y-axis of this function then find the
  corresponding x-axis points. In order to simplify the implementation, here is
  a plan how it is achieved:
  1. Calculate CDF values at points spaced linearly between weight_min and
  weight_max(e.g. 20 points)
  2. Build an array of values linearly spaced between 0 and 1(probability)
  3. Go through the second array and find segment of CDF that contains this
  y-axis value, \\hat{y}
  4. interpolate linearly between those two points, get a line equation y=ax+b
  5. solve equation \\hat{y}=ax+b for x. The found x value is a new cluster
  centroid
  """

  def get_cluster_centroids(self):
    weight_min = tf.reduce_min(self.weights)
    weight_max = tf.reduce_max(self.weights)
    # Calculating interpolation nodes, +/- 0.01 is introduced to guarantee that
    # CDF will have 0 and 1 and the first and last value respectively.
    # The value 30 is a guess. We just need a sufficiently large number here
    # since we are going to interpolate values linearly anyway and the initial
    # guess will drift away. For these reasons we do not really
    # care about the granularity of the lookup.
    cdf_x_grid = tf.linspace(weight_min - 0.01, weight_max + 0.01, 30)

    f = TFCumulativeDistributionFunction(weights=self.weights)

    cdf_values = k.map_fn(f.get_cdf_value, cdf_x_grid)

    probability_space = tf.linspace(0 + 0.01, 1, self.number_of_clusters)

    # Use upper-bound algorithm to find the appropriate bounds
    matching_indices = tf.searchsorted(sorted_sequence=cdf_values,
                                       values=probability_space,
                                       side='right')

    # Interpolate linearly between every found indices I at position using I at
    # pos n-1 as a second point. The value of x is a new cluster centroid
    def get_single_centroid(i):
      i_clipped = tf.minimum(i, tf.size(cdf_values) - 1)
      i_previous = tf.maximum(0, i_clipped - 1)

      s = TFLinearEquationSolver(x1=cdf_x_grid[i_clipped],
                                 y1=cdf_values[i_clipped],
                                 x2=cdf_x_grid[i_previous],
                                 y2=cdf_values[i_previous])

      y = cdf_values[i_clipped]

      single_centroid = s.solve_for_x(y)
      return single_centroid

    centroids = k.map_fn(get_single_centroid,
                         matching_indices,
                         dtype=tf.float32)
    cluster_centroids = tf.reshape(centroids, (self.number_of_clusters,))
    return cluster_centroids


class CentroidsInitializerFactory:
  """
  Factory that creates concrete initializers for factory centroids.
  To implement a custom one, inherit from AbstractCentroidsInitialisation
  and implement all the required methods.

  After this, update CentroidsInitialiserFactory.__initialisers hashtable to
  reflect new methods available.
  """
  _initialisers = {
      CentroidInitialization.LINEAR : LinearCentroidsInitialisation,
      CentroidInitialization.RANDOM : RandomCentroidsInitialisation,
      CentroidInitialization.DENSITY_BASED :
          DensityBasedCentroidsInitialisation
  }

  @classmethod
  def init_is_supported(cls, init_method):
    return init_method in cls._initialisers

  @classmethod
  def get_centroid_initializer(cls, init_method):
    """
    :param init_method: a CentroidInitialization value representing the init
      method requested
    :return: A concrete implementation of AbstractCentroidsInitialisation
    :raises: ValueError if the requested centroid initialization method is not
      recognised
    """
    if not cls.init_is_supported(init_method):
      raise ValueError(
          "Unknown initialisation method: {init_method}. Allowed values are : "
          "{allowed}".format(
              init_method=init_method,
              allowed=','.join(cls._initialisers.keys())
          ))

    return cls._initialisers[init_method]
