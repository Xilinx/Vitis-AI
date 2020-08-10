# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras metrics functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import metrics
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class FalsePositivesTest(test.TestCase):

  def test_config(self):
    fp_obj = metrics.FalsePositives(name='my_fp', thresholds=[0.4, 0.9])
    self.assertEqual(fp_obj.name, 'my_fp')
    self.assertEqual(len(fp_obj.variables), 1)
    self.assertEqual(fp_obj.thresholds, [0.4, 0.9])

    # Check save and restore config
    fp_obj2 = metrics.FalsePositives.from_config(fp_obj.get_config())
    self.assertEqual(fp_obj2.name, 'my_fp')
    self.assertEqual(len(fp_obj2.variables), 1)
    self.assertEqual(fp_obj2.thresholds, [0.4, 0.9])

  def test_unweighted(self):
    fp_obj = metrics.FalsePositives()
    self.evaluate(variables.variables_initializer(fp_obj.variables))

    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = fp_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = fp_obj.result()
    self.assertAllClose(7., result)

  def test_weighted(self):
    fp_obj = metrics.FalsePositives()
    self.evaluate(variables.variables_initializer(fp_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(14., self.evaluate(result))

  def test_unweighted_with_thresholds(self):
    fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(fp_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))

    update_op = fp_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = fp_obj.result()
    self.assertAllClose([7., 4., 2.], result)

  def test_weighted_with_thresholds(self):
    fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(fp_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))
    sample_weight = ((1.0, 2.0, 3.0, 5.0), (7.0, 11.0, 13.0, 17.0),
                     (19.0, 23.0, 29.0, 31.0), (5.0, 15.0, 10.0, 0))

    result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose([125., 42., 12.], self.evaluate(result))

  def test_threshold_limit(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'Threshold values must be in \[0, 1\]. Invalid values: \[-1, 2\]'):
      metrics.FalsePositives(thresholds=[-1, 0.5, 2])

    with self.assertRaisesRegexp(
        ValueError,
        r'Threshold values must be in \[0, 1\]. Invalid values: \[None\]'):
      metrics.FalsePositives(thresholds=[None])


@test_util.run_all_in_graph_and_eager_modes
class FalseNegativesTest(test.TestCase):

  def test_config(self):
    fn_obj = metrics.FalseNegatives(name='my_fn', thresholds=[0.4, 0.9])
    self.assertEqual(fn_obj.name, 'my_fn')
    self.assertEqual(len(fn_obj.variables), 1)
    self.assertEqual(fn_obj.thresholds, [0.4, 0.9])

    # Check save and restore config
    fn_obj2 = metrics.FalseNegatives.from_config(fn_obj.get_config())
    self.assertEqual(fn_obj2.name, 'my_fn')
    self.assertEqual(len(fn_obj2.variables), 1)
    self.assertEqual(fn_obj2.thresholds, [0.4, 0.9])

  def test_unweighted(self):
    fn_obj = metrics.FalseNegatives()
    self.evaluate(variables.variables_initializer(fn_obj.variables))

    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = fn_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = fn_obj.result()
    self.assertAllClose(3., result)

  def test_weighted(self):
    fn_obj = metrics.FalseNegatives()
    self.evaluate(variables.variables_initializer(fn_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = fn_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(5., self.evaluate(result))

  def test_unweighted_with_thresholds(self):
    fn_obj = metrics.FalseNegatives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(fn_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))

    update_op = fn_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = fn_obj.result()
    self.assertAllClose([1., 4., 6.], result)

  def test_weighted_with_thresholds(self):
    fn_obj = metrics.FalseNegatives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(fn_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))
    sample_weight = ((3.0,), (5.0,), (7.0,), (4.0,))

    result = fn_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose([4., 16., 23.], self.evaluate(result))


@test_util.run_all_in_graph_and_eager_modes
class TrueNegativesTest(test.TestCase):

  def test_config(self):
    tn_obj = metrics.TrueNegatives(name='my_tn', thresholds=[0.4, 0.9])
    self.assertEqual(tn_obj.name, 'my_tn')
    self.assertEqual(len(tn_obj.variables), 1)
    self.assertEqual(tn_obj.thresholds, [0.4, 0.9])

    # Check save and restore config
    tn_obj2 = metrics.TrueNegatives.from_config(tn_obj.get_config())
    self.assertEqual(tn_obj2.name, 'my_tn')
    self.assertEqual(len(tn_obj2.variables), 1)
    self.assertEqual(tn_obj2.thresholds, [0.4, 0.9])

  def test_unweighted(self):
    tn_obj = metrics.TrueNegatives()
    self.evaluate(variables.variables_initializer(tn_obj.variables))

    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = tn_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = tn_obj.result()
    self.assertAllClose(3., result)

  def test_weighted(self):
    tn_obj = metrics.TrueNegatives()
    self.evaluate(variables.variables_initializer(tn_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = tn_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(4., self.evaluate(result))

  def test_unweighted_with_thresholds(self):
    tn_obj = metrics.TrueNegatives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(tn_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))

    update_op = tn_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = tn_obj.result()
    self.assertAllClose([2., 5., 7.], result)

  def test_weighted_with_thresholds(self):
    tn_obj = metrics.TrueNegatives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(tn_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))
    sample_weight = ((0.0, 2.0, 3.0, 5.0),)

    result = tn_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose([5., 15., 23.], self.evaluate(result))


@test_util.run_all_in_graph_and_eager_modes
class TruePositivesTest(test.TestCase):

  def test_config(self):
    tp_obj = metrics.TruePositives(name='my_tp', thresholds=[0.4, 0.9])
    self.assertEqual(tp_obj.name, 'my_tp')
    self.assertEqual(len(tp_obj.variables), 1)
    self.assertEqual(tp_obj.thresholds, [0.4, 0.9])

    # Check save and restore config
    tp_obj2 = metrics.TruePositives.from_config(tp_obj.get_config())
    self.assertEqual(tp_obj2.name, 'my_tp')
    self.assertEqual(len(tp_obj2.variables), 1)
    self.assertEqual(tp_obj2.thresholds, [0.4, 0.9])

  def test_unweighted(self):
    tp_obj = metrics.TruePositives()
    self.evaluate(variables.variables_initializer(tp_obj.variables))

    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = tp_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = tp_obj.result()
    self.assertAllClose(7., result)

  def test_weighted(self):
    tp_obj = metrics.TruePositives()
    self.evaluate(variables.variables_initializer(tp_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = tp_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(12., self.evaluate(result))

  def test_unweighted_with_thresholds(self):
    tp_obj = metrics.TruePositives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(tp_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))

    update_op = tp_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = tp_obj.result()
    self.assertAllClose([6., 3., 1.], result)

  def test_weighted_with_thresholds(self):
    tp_obj = metrics.TruePositives(thresholds=[0.15, 0.5, 0.85])
    self.evaluate(variables.variables_initializer(tp_obj.variables))

    y_pred = constant_op.constant(((0.9, 0.2, 0.8, 0.1), (0.2, 0.9, 0.7, 0.6),
                                   (0.1, 0.2, 0.4, 0.3), (0, 1, 0.7, 0.3)))
    y_true = constant_op.constant(((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0),
                                   (1, 1, 1, 1)))

    result = tp_obj(y_true, y_pred, sample_weight=37.)
    self.assertAllClose([222., 111., 37.], self.evaluate(result))


@test_util.run_all_in_graph_and_eager_modes
class PrecisionTest(test.TestCase):

  def test_config(self):
    p_obj = metrics.Precision(
        name='my_precision', thresholds=[0.4, 0.9], top_k=15, class_id=12)
    self.assertEqual(p_obj.name, 'my_precision')
    self.assertEqual(len(p_obj.variables), 2)
    self.assertEqual([v.name for v in p_obj.variables],
                     ['true_positives:0', 'false_positives:0'])
    self.assertEqual(p_obj.thresholds, [0.4, 0.9])
    self.assertEqual(p_obj.top_k, 15)
    self.assertEqual(p_obj.class_id, 12)

    # Check save and restore config
    p_obj2 = metrics.Precision.from_config(p_obj.get_config())
    self.assertEqual(p_obj2.name, 'my_precision')
    self.assertEqual(len(p_obj2.variables), 2)
    self.assertEqual(p_obj2.thresholds, [0.4, 0.9])
    self.assertEqual(p_obj2.top_k, 15)
    self.assertEqual(p_obj2.class_id, 12)

  def test_value_is_idempotent(self):
    p_obj = metrics.Precision(thresholds=[0.3, 0.72])
    y_pred = random_ops.random_uniform(shape=(10, 3))
    y_true = random_ops.random_uniform(shape=(10, 3))
    update_op = p_obj.update_state(y_true, y_pred)
    self.evaluate(variables.variables_initializer(p_obj.variables))

    # Run several updates.
    for _ in range(10):
      self.evaluate(update_op)

    # Then verify idempotency.
    initial_precision = self.evaluate(p_obj.result())
    for _ in range(10):
      self.assertArrayNear(initial_precision, self.evaluate(p_obj.result()),
                           1e-3)

  def test_unweighted(self):
    p_obj = metrics.Precision()
    y_pred = constant_op.constant([1, 0, 1, 0], shape=(1, 4))
    y_true = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
    self.evaluate(variables.variables_initializer(p_obj.variables))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(0.5, self.evaluate(result))

  def test_unweighted_all_incorrect(self):
    p_obj = metrics.Precision(thresholds=[0.5])
    inputs = np.random.randint(0, 2, size=(100, 1))
    y_pred = constant_op.constant(inputs)
    y_true = constant_op.constant(1 - inputs)
    self.evaluate(variables.variables_initializer(p_obj.variables))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(0, self.evaluate(result))

  def test_weighted(self):
    p_obj = metrics.Precision()
    y_pred = constant_op.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
    y_true = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    self.evaluate(variables.variables_initializer(p_obj.variables))
    result = p_obj(
        y_true,
        y_pred,
        sample_weight=constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
    weighted_tp = 3.0 + 4.0
    weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
    expected_precision = weighted_tp / weighted_positives
    self.assertAlmostEqual(expected_precision, self.evaluate(result))

  def test_div_by_zero(self):
    p_obj = metrics.Precision()
    y_pred = constant_op.constant([0, 0, 0, 0])
    y_true = constant_op.constant([0, 0, 0, 0])
    self.evaluate(variables.variables_initializer(p_obj.variables))
    result = p_obj(y_true, y_pred)
    self.assertEqual(0, self.evaluate(result))

  def test_unweighted_with_threshold(self):
    p_obj = metrics.Precision(thresholds=[0.5, 0.7])
    y_pred = constant_op.constant([1, 0, 0.6, 0], shape=(1, 4))
    y_true = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
    self.evaluate(variables.variables_initializer(p_obj.variables))
    result = p_obj(y_true, y_pred)
    self.assertArrayNear([0.5, 0.], self.evaluate(result), 0)

  def test_weighted_with_threshold(self):
    p_obj = metrics.Precision(thresholds=[0.5, 1.])
    y_true = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
    y_pred = constant_op.constant([[1, 0], [0.6, 0]],
                                  shape=(2, 2),
                                  dtype=dtypes.float32)
    weights = constant_op.constant([[4, 0], [3, 1]],
                                   shape=(2, 2),
                                   dtype=dtypes.float32)
    self.evaluate(variables.variables_initializer(p_obj.variables))
    result = p_obj(y_true, y_pred, sample_weight=weights)
    weighted_tp = 0 + 3.
    weighted_positives = (0 + 3.) + (4. + 0.)
    expected_precision = weighted_tp / weighted_positives
    self.assertArrayNear([expected_precision, 0], self.evaluate(result), 1e-3)

  def test_multiple_updates(self):
    p_obj = metrics.Precision(thresholds=[0.5, 1.])
    y_true = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
    y_pred = constant_op.constant([[1, 0], [0.6, 0]],
                                  shape=(2, 2),
                                  dtype=dtypes.float32)
    weights = constant_op.constant([[4, 0], [3, 1]],
                                   shape=(2, 2),
                                   dtype=dtypes.float32)
    self.evaluate(variables.variables_initializer(p_obj.variables))
    update_op = p_obj.update_state(y_true, y_pred, sample_weight=weights)
    for _ in range(2):
      self.evaluate(update_op)

    weighted_tp = (0 + 3.) + (0 + 3.)
    weighted_positives = ((0 + 3.) + (4. + 0.)) + ((0 + 3.) + (4. + 0.))
    expected_precision = weighted_tp / weighted_positives
    self.assertArrayNear([expected_precision, 0], self.evaluate(p_obj.result()),
                         1e-3)

  def test_unweighted_top_k(self):
    p_obj = metrics.Precision(top_k=3)
    y_pred = constant_op.constant([0.2, 0.1, 0.5, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    self.evaluate(variables.variables_initializer(p_obj.variables))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(1. / 3, self.evaluate(result))

  def test_weighted_top_k(self):
    p_obj = metrics.Precision(top_k=3)
    y_pred1 = constant_op.constant([0.2, 0.1, 0.4, 0, 0.2], shape=(1, 5))
    y_true1 = constant_op.constant([0, 1, 1, 0, 1], shape=(1, 5))
    self.evaluate(variables.variables_initializer(p_obj.variables))
    self.evaluate(
        p_obj(
            y_true1,
            y_pred1,
            sample_weight=constant_op.constant([[1, 4, 2, 3, 5]])))

    y_pred2 = constant_op.constant([0.2, 0.6, 0.4, 0.2, 0.2], shape=(1, 5))
    y_true2 = constant_op.constant([1, 0, 1, 1, 1], shape=(1, 5))
    result = p_obj(y_true2, y_pred2, sample_weight=constant_op.constant(3))

    tp = (2 + 5) + (3 + 3)
    predicted_positives = (1 + 2 + 5) + (3 + 3 + 3)
    expected_precision = tp / predicted_positives
    self.assertAlmostEqual(expected_precision, self.evaluate(result))

  def test_unweighted_class_id(self):
    p_obj = metrics.Precision(class_id=2)
    self.evaluate(variables.variables_initializer(p_obj.variables))

    y_pred = constant_op.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(p_obj.true_positives))
    self.assertAlmostEqual(0, self.evaluate(p_obj.false_positives))

    y_pred = constant_op.constant([0.2, 0.1, 0, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(p_obj.true_positives))
    self.assertAlmostEqual(0, self.evaluate(p_obj.false_positives))

    y_pred = constant_op.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 0, 0, 0], shape=(1, 5))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(0.5, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(p_obj.true_positives))
    self.assertAlmostEqual(1, self.evaluate(p_obj.false_positives))

  def test_unweighted_top_k_and_class_id(self):
    p_obj = metrics.Precision(class_id=2, top_k=2)
    self.evaluate(variables.variables_initializer(p_obj.variables))

    y_pred = constant_op.constant([0.2, 0.6, 0.3, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(p_obj.true_positives))
    self.assertAlmostEqual(0, self.evaluate(p_obj.false_positives))

    y_pred = constant_op.constant([1, 1, 0.9, 1, 1], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(p_obj.true_positives))
    self.assertAlmostEqual(0, self.evaluate(p_obj.false_positives))

  def test_unweighted_top_k_and_threshold(self):
    p_obj = metrics.Precision(thresholds=.7, top_k=2)
    self.evaluate(variables.variables_initializer(p_obj.variables))

    y_pred = constant_op.constant([0.2, 0.8, 0.6, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 1], shape=(1, 5))
    result = p_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(p_obj.true_positives))
    self.assertAlmostEqual(0, self.evaluate(p_obj.false_positives))


@test_util.run_all_in_graph_and_eager_modes
class RecallTest(test.TestCase):

  def test_config(self):
    r_obj = metrics.Recall(
        name='my_recall', thresholds=[0.4, 0.9], top_k=15, class_id=12)
    self.assertEqual(r_obj.name, 'my_recall')
    self.assertEqual(len(r_obj.variables), 2)
    self.assertEqual([v.name for v in r_obj.variables],
                     ['true_positives:0', 'false_negatives:0'])
    self.assertEqual(r_obj.thresholds, [0.4, 0.9])
    self.assertEqual(r_obj.top_k, 15)
    self.assertEqual(r_obj.class_id, 12)

    # Check save and restore config
    r_obj2 = metrics.Recall.from_config(r_obj.get_config())
    self.assertEqual(r_obj2.name, 'my_recall')
    self.assertEqual(len(r_obj2.variables), 2)
    self.assertEqual(r_obj2.thresholds, [0.4, 0.9])
    self.assertEqual(r_obj2.top_k, 15)
    self.assertEqual(r_obj2.class_id, 12)

  def test_value_is_idempotent(self):
    r_obj = metrics.Recall(thresholds=[0.3, 0.72])
    y_pred = random_ops.random_uniform(shape=(10, 3))
    y_true = random_ops.random_uniform(shape=(10, 3))
    update_op = r_obj.update_state(y_true, y_pred)
    self.evaluate(variables.variables_initializer(r_obj.variables))

    # Run several updates.
    for _ in range(10):
      self.evaluate(update_op)

    # Then verify idempotency.
    initial_recall = self.evaluate(r_obj.result())
    for _ in range(10):
      self.assertArrayNear(initial_recall, self.evaluate(r_obj.result()), 1e-3)

  def test_unweighted(self):
    r_obj = metrics.Recall()
    y_pred = constant_op.constant([1, 0, 1, 0], shape=(1, 4))
    y_true = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
    self.evaluate(variables.variables_initializer(r_obj.variables))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(0.5, self.evaluate(result))

  def test_unweighted_all_incorrect(self):
    r_obj = metrics.Recall(thresholds=[0.5])
    inputs = np.random.randint(0, 2, size=(100, 1))
    y_pred = constant_op.constant(inputs)
    y_true = constant_op.constant(1 - inputs)
    self.evaluate(variables.variables_initializer(r_obj.variables))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(0, self.evaluate(result))

  def test_weighted(self):
    r_obj = metrics.Recall()
    y_pred = constant_op.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    y_true = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    self.evaluate(variables.variables_initializer(r_obj.variables))
    result = r_obj(
        y_true,
        y_pred,
        sample_weight=constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
    weighted_tp = 3.0 + 1.0
    weighted_t = (2.0 + 3.0) + (4.0 + 1.0)
    expected_recall = weighted_tp / weighted_t
    self.assertAlmostEqual(expected_recall, self.evaluate(result))

  def test_div_by_zero(self):
    r_obj = metrics.Recall()
    y_pred = constant_op.constant([0, 0, 0, 0])
    y_true = constant_op.constant([0, 0, 0, 0])
    self.evaluate(variables.variables_initializer(r_obj.variables))
    result = r_obj(y_true, y_pred)
    self.assertEqual(0, self.evaluate(result))

  def test_unweighted_with_threshold(self):
    r_obj = metrics.Recall(thresholds=[0.5, 0.7])
    y_pred = constant_op.constant([1, 0, 0.6, 0], shape=(1, 4))
    y_true = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
    self.evaluate(variables.variables_initializer(r_obj.variables))
    result = r_obj(y_true, y_pred)
    self.assertArrayNear([0.5, 0.], self.evaluate(result), 0)

  def test_weighted_with_threshold(self):
    r_obj = metrics.Recall(thresholds=[0.5, 1.])
    y_true = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
    y_pred = constant_op.constant([[1, 0], [0.6, 0]],
                                  shape=(2, 2),
                                  dtype=dtypes.float32)
    weights = constant_op.constant([[1, 4], [3, 2]],
                                   shape=(2, 2),
                                   dtype=dtypes.float32)
    self.evaluate(variables.variables_initializer(r_obj.variables))
    result = r_obj(y_true, y_pred, sample_weight=weights)
    weighted_tp = 0 + 3.
    weighted_positives = (0 + 3.) + (4. + 0.)
    expected_recall = weighted_tp / weighted_positives
    self.assertArrayNear([expected_recall, 0], self.evaluate(result), 1e-3)

  def test_multiple_updates(self):
    r_obj = metrics.Recall(thresholds=[0.5, 1.])
    y_true = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
    y_pred = constant_op.constant([[1, 0], [0.6, 0]],
                                  shape=(2, 2),
                                  dtype=dtypes.float32)
    weights = constant_op.constant([[1, 4], [3, 2]],
                                   shape=(2, 2),
                                   dtype=dtypes.float32)
    self.evaluate(variables.variables_initializer(r_obj.variables))
    update_op = r_obj.update_state(y_true, y_pred, sample_weight=weights)
    for _ in range(2):
      self.evaluate(update_op)

    weighted_tp = (0 + 3.) + (0 + 3.)
    weighted_positives = ((0 + 3.) + (4. + 0.)) + ((0 + 3.) + (4. + 0.))
    expected_recall = weighted_tp / weighted_positives
    self.assertArrayNear([expected_recall, 0], self.evaluate(r_obj.result()),
                         1e-3)

  def test_unweighted_top_k(self):
    r_obj = metrics.Recall(top_k=3)
    y_pred = constant_op.constant([0.2, 0.1, 0.5, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    self.evaluate(variables.variables_initializer(r_obj.variables))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(0.5, self.evaluate(result))

  def test_weighted_top_k(self):
    r_obj = metrics.Recall(top_k=3)
    y_pred1 = constant_op.constant([0.2, 0.1, 0.4, 0, 0.2], shape=(1, 5))
    y_true1 = constant_op.constant([0, 1, 1, 0, 1], shape=(1, 5))
    self.evaluate(variables.variables_initializer(r_obj.variables))
    self.evaluate(
        r_obj(
            y_true1,
            y_pred1,
            sample_weight=constant_op.constant([[1, 4, 2, 3, 5]])))

    y_pred2 = constant_op.constant([0.2, 0.6, 0.4, 0.2, 0.2], shape=(1, 5))
    y_true2 = constant_op.constant([1, 0, 1, 1, 1], shape=(1, 5))
    result = r_obj(y_true2, y_pred2, sample_weight=constant_op.constant(3))

    tp = (2 + 5) + (3 + 3)
    positives = (4 + 2 + 5) + (3 + 3 + 3 + 3)
    expected_recall = tp / positives
    self.assertAlmostEqual(expected_recall, self.evaluate(result))

  def test_unweighted_class_id(self):
    r_obj = metrics.Recall(class_id=2)
    self.evaluate(variables.variables_initializer(r_obj.variables))

    y_pred = constant_op.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    self.assertAlmostEqual(0, self.evaluate(r_obj.false_negatives))

    y_pred = constant_op.constant([0.2, 0.1, 0, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(0.5, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    self.assertAlmostEqual(1, self.evaluate(r_obj.false_negatives))

    y_pred = constant_op.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 0, 0, 0], shape=(1, 5))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(0.5, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    self.assertAlmostEqual(1, self.evaluate(r_obj.false_negatives))

  def test_unweighted_top_k_and_class_id(self):
    r_obj = metrics.Recall(class_id=2, top_k=2)
    self.evaluate(variables.variables_initializer(r_obj.variables))

    y_pred = constant_op.constant([0.2, 0.6, 0.3, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    self.assertAlmostEqual(0, self.evaluate(r_obj.false_negatives))

    y_pred = constant_op.constant([1, 1, 0.9, 1, 1], shape=(1, 5))
    y_true = constant_op.constant([0, 1, 1, 0, 0], shape=(1, 5))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(0.5, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    self.assertAlmostEqual(1, self.evaluate(r_obj.false_negatives))

  def test_unweighted_top_k_and_threshold(self):
    r_obj = metrics.Recall(thresholds=.7, top_k=2)
    self.evaluate(variables.variables_initializer(r_obj.variables))

    y_pred = constant_op.constant([0.2, 0.8, 0.6, 0, 0.2], shape=(1, 5))
    y_true = constant_op.constant([1, 1, 1, 0, 1], shape=(1, 5))
    result = r_obj(y_true, y_pred)
    self.assertAlmostEqual(0.25, self.evaluate(result))
    self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    self.assertAlmostEqual(3, self.evaluate(r_obj.false_negatives))


@test_util.run_all_in_graph_and_eager_modes
class SensitivityAtSpecificityTest(test.TestCase, parameterized.TestCase):

  def test_config(self):
    s_obj = metrics.SensitivityAtSpecificity(
        0.4, num_thresholds=100, name='sensitivity_at_specificity_1')
    self.assertEqual(s_obj.name, 'sensitivity_at_specificity_1')
    self.assertLen(s_obj.variables, 4)
    self.assertEqual(s_obj.specificity, 0.4)
    self.assertEqual(s_obj.num_thresholds, 100)

    # Check save and restore config
    s_obj2 = metrics.SensitivityAtSpecificity.from_config(s_obj.get_config())
    self.assertEqual(s_obj2.name, 'sensitivity_at_specificity_1')
    self.assertLen(s_obj2.variables, 4)
    self.assertEqual(s_obj2.specificity, 0.4)
    self.assertEqual(s_obj2.num_thresholds, 100)

  def test_value_is_idempotent(self):
    s_obj = metrics.SensitivityAtSpecificity(0.7)
    y_pred = random_ops.random_uniform((10, 3),
                                       maxval=1,
                                       dtype=dtypes.float32,
                                       seed=1)
    y_true = random_ops.random_uniform((10, 3),
                                       maxval=2,
                                       dtype=dtypes.int64,
                                       seed=1)
    update_op = s_obj.update_state(y_true, y_pred)
    self.evaluate(variables.variables_initializer(s_obj.variables))

    # Run several updates.
    for _ in range(10):
      self.evaluate(update_op)

    # Then verify idempotency.
    initial_sensitivity = self.evaluate(s_obj.result())
    for _ in range(10):
      self.assertAlmostEqual(initial_sensitivity, self.evaluate(s_obj.result()),
                             1e-3)

  def test_unweighted_all_correct(self):
    s_obj = metrics.SensitivityAtSpecificity(0.7)
    inputs = np.random.randint(0, 2, size=(100, 1))
    y_pred = constant_op.constant(inputs, dtype=dtypes.float32)
    y_true = constant_op.constant(inputs)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))

  def test_unweighted_high_specificity(self):
    s_obj = metrics.SensitivityAtSpecificity(0.8)
    pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.45, 0.5, 0.8, 0.9]
    label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    y_pred = constant_op.constant(pred_values, dtype=dtypes.float32)
    y_true = constant_op.constant(label_values)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred)
    self.assertAlmostEqual(0.8, self.evaluate(result))

  def test_unweighted_low_specificity(self):
    s_obj = metrics.SensitivityAtSpecificity(0.4)
    pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
    label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    y_pred = constant_op.constant(pred_values, dtype=dtypes.float32)
    y_true = constant_op.constant(label_values)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred)
    self.assertAlmostEqual(0.6, self.evaluate(result))

  @parameterized.parameters([dtypes.bool, dtypes.int32, dtypes.float32])
  def test_weighted(self, label_dtype):
    s_obj = metrics.SensitivityAtSpecificity(0.4)
    pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
    label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    y_pred = constant_op.constant(pred_values, dtype=dtypes.float32)
    y_true = math_ops.cast(label_values, dtype=label_dtype)
    weights = constant_op.constant(weight_values)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred, sample_weight=weights)
    self.assertAlmostEqual(0.675, self.evaluate(result))

  def test_invalid_specificity(self):
    with self.assertRaisesRegexp(
        ValueError, r'`specificity` must be in the range \[0, 1\].'):
      metrics.SensitivityAtSpecificity(-1)

  def test_invalid_num_thresholds(self):
    with self.assertRaisesRegexp(ValueError, '`num_thresholds` must be > 0.'):
      metrics.SensitivityAtSpecificity(0.4, num_thresholds=-1)


@test_util.run_all_in_graph_and_eager_modes
class SpecificityAtSensitivityTest(test.TestCase, parameterized.TestCase):

  def test_config(self):
    s_obj = metrics.SpecificityAtSensitivity(
        0.4, num_thresholds=100, name='specificity_at_sensitivity_1')
    self.assertEqual(s_obj.name, 'specificity_at_sensitivity_1')
    self.assertLen(s_obj.variables, 4)
    self.assertEqual(s_obj.sensitivity, 0.4)
    self.assertEqual(s_obj.num_thresholds, 100)

    # Check save and restore config
    s_obj2 = metrics.SpecificityAtSensitivity.from_config(s_obj.get_config())
    self.assertEqual(s_obj2.name, 'specificity_at_sensitivity_1')
    self.assertLen(s_obj2.variables, 4)
    self.assertEqual(s_obj2.sensitivity, 0.4)
    self.assertEqual(s_obj2.num_thresholds, 100)

  def test_value_is_idempotent(self):
    s_obj = metrics.SpecificityAtSensitivity(0.7)
    y_pred = random_ops.random_uniform((10, 3),
                                       maxval=1,
                                       dtype=dtypes.float32,
                                       seed=1)
    y_true = random_ops.random_uniform((10, 3),
                                       maxval=2,
                                       dtype=dtypes.int64,
                                       seed=1)
    update_op = s_obj.update_state(y_true, y_pred)
    self.evaluate(variables.variables_initializer(s_obj.variables))

    # Run several updates.
    for _ in range(10):
      self.evaluate(update_op)

    # Then verify idempotency.
    initial_specificity = self.evaluate(s_obj.result())
    for _ in range(10):
      self.assertAlmostEqual(initial_specificity, self.evaluate(s_obj.result()),
                             1e-3)

  def test_unweighted_all_correct(self):
    s_obj = metrics.SpecificityAtSensitivity(0.7)
    inputs = np.random.randint(0, 2, size=(100, 1))
    y_pred = constant_op.constant(inputs, dtype=dtypes.float32)
    y_true = constant_op.constant(inputs)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred)
    self.assertAlmostEqual(1, self.evaluate(result))

  def test_unweighted_high_sensitivity(self):
    s_obj = metrics.SpecificityAtSensitivity(0.8)
    pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.45, 0.5, 0.8, 0.9]
    label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    y_pred = constant_op.constant(pred_values, dtype=dtypes.float32)
    y_true = constant_op.constant(label_values)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred)
    self.assertAlmostEqual(0.4, self.evaluate(result))

  def test_unweighted_low_sensitivity(self):
    s_obj = metrics.SpecificityAtSensitivity(0.4)
    pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
    label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    y_pred = constant_op.constant(pred_values, dtype=dtypes.float32)
    y_true = constant_op.constant(label_values)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred)
    self.assertAlmostEqual(0.6, self.evaluate(result))

  @parameterized.parameters([dtypes.bool, dtypes.int32, dtypes.float32])
  def test_weighted(self, label_dtype):
    s_obj = metrics.SpecificityAtSensitivity(0.4)
    pred_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
    label_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    y_pred = constant_op.constant(pred_values, dtype=dtypes.float32)
    y_true = math_ops.cast(label_values, dtype=label_dtype)
    weights = constant_op.constant(weight_values)
    self.evaluate(variables.variables_initializer(s_obj.variables))
    result = s_obj(y_true, y_pred, sample_weight=weights)
    self.assertAlmostEqual(0.4, self.evaluate(result))

  def test_invalid_sensitivity(self):
    with self.assertRaisesRegexp(
        ValueError, r'`sensitivity` must be in the range \[0, 1\].'):
      metrics.SpecificityAtSensitivity(-1)

  def test_invalid_num_thresholds(self):
    with self.assertRaisesRegexp(ValueError, '`num_thresholds` must be > 0.'):
      metrics.SpecificityAtSensitivity(0.4, num_thresholds=-1)


@test_util.run_all_in_graph_and_eager_modes
class AUCTest(test.TestCase):

  def setup(self):
    self.num_thresholds = 3
    self.y_pred = constant_op.constant([0, 0.5, 0.3, 0.9], dtype=dtypes.float32)
    self.y_true = constant_op.constant([0, 0, 1, 1])
    self.sample_weight = [1, 2, 3, 4]

    # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    # y_pred when threshold = 0 - 1e-7  : [1, 1, 1, 1]
    # y_pred when threshold = 0.5       : [0, 0, 0, 1]
    # y_pred when threshold = 1 + 1e-7  : [0, 0, 0, 0]

    # without sample_weight:
    # tp = np.sum([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]], axis=1)
    # fp = np.sum([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], axis=1)
    # fn = np.sum([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], axis=1)
    # tn = np.sum([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], axis=1)

    # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]

    # with sample_weight:
    # tp = np.sum([[0, 0, 3, 4], [0, 0, 0, 4], [0, 0, 0, 0]], axis=1)
    # fp = np.sum([[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], axis=1)
    # fn = np.sum([[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 3, 4]], axis=1)
    # tn = np.sum([[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]], axis=1)

    # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]

  def test_config(self):
    auc_obj = metrics.AUC(
        num_thresholds=100,
        curve='PR',
        summation_method='majoring',
        name='auc_1')
    self.assertEqual(auc_obj.name, 'auc_1')
    self.assertEqual(len(auc_obj.variables), 4)
    self.assertEqual(auc_obj.num_thresholds, 100)
    self.assertEqual(auc_obj.curve, metrics_utils.AUCCurve.PR)
    self.assertEqual(auc_obj.summation_method,
                     metrics_utils.AUCSummationMethod.MAJORING)
    old_config = auc_obj.get_config()
    self.assertDictEqual(old_config, json.loads(json.dumps(old_config)))

    # Check save and restore config.
    auc_obj2 = metrics.AUC.from_config(auc_obj.get_config())
    self.assertEqual(auc_obj2.name, 'auc_1')
    self.assertEqual(len(auc_obj2.variables), 4)
    self.assertEqual(auc_obj2.num_thresholds, 100)
    self.assertEqual(auc_obj2.curve, metrics_utils.AUCCurve.PR)
    self.assertEqual(auc_obj2.summation_method,
                     metrics_utils.AUCSummationMethod.MAJORING)
    new_config = auc_obj2.get_config()
    self.assertDictEqual(old_config, new_config)
    self.assertAllClose(auc_obj.thresholds, auc_obj2.thresholds)

  def test_config_manual_thresholds(self):
    auc_obj = metrics.AUC(
        num_thresholds=None,
        curve='PR',
        summation_method='majoring',
        name='auc_1',
        thresholds=[0.3, 0.5])
    self.assertEqual(auc_obj.name, 'auc_1')
    self.assertEqual(len(auc_obj.variables), 4)
    self.assertEqual(auc_obj.num_thresholds, 4)
    self.assertAllClose(auc_obj.thresholds, [0.0, 0.3, 0.5, 1.0])
    self.assertEqual(auc_obj.curve, metrics_utils.AUCCurve.PR)
    self.assertEqual(auc_obj.summation_method,
                     metrics_utils.AUCSummationMethod.MAJORING)
    old_config = auc_obj.get_config()
    self.assertDictEqual(old_config, json.loads(json.dumps(old_config)))

    # Check save and restore config.
    auc_obj2 = metrics.AUC.from_config(auc_obj.get_config())
    self.assertEqual(auc_obj2.name, 'auc_1')
    self.assertEqual(len(auc_obj2.variables), 4)
    self.assertEqual(auc_obj2.num_thresholds, 4)
    self.assertEqual(auc_obj2.curve, metrics_utils.AUCCurve.PR)
    self.assertEqual(auc_obj2.summation_method,
                     metrics_utils.AUCSummationMethod.MAJORING)
    new_config = auc_obj2.get_config()
    self.assertDictEqual(old_config, new_config)
    self.assertAllClose(auc_obj.thresholds, auc_obj2.thresholds)

  def test_value_is_idempotent(self):
    self.setup()
    auc_obj = metrics.AUC(num_thresholds=3)
    self.evaluate(variables.variables_initializer(auc_obj.variables))

    # Run several updates.
    update_op = auc_obj.update_state(self.y_true, self.y_pred)
    for _ in range(10):
      self.evaluate(update_op)

    # Then verify idempotency.
    initial_auc = self.evaluate(auc_obj.result())
    for _ in range(10):
      self.assertAllClose(initial_auc, self.evaluate(auc_obj.result()), 1e-3)

  def test_unweighted_all_correct(self):
    self.setup()
    auc_obj = metrics.AUC()
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_true)
    self.assertEqual(self.evaluate(result), 1)

  def test_unweighted(self):
    self.setup()
    auc_obj = metrics.AUC(num_thresholds=self.num_thresholds)
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred)

    # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
    # fp_rate = [2/2, 0, 0] = [1, 0, 0]
    # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
    # widths = [(1 - 0), (0 - 0)] = [1, 0]
    expected_result = (0.75 * 1 + 0.25 * 0)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_manual_thresholds(self):
    self.setup()
    # Verify that when specified, thresholds are used instead of num_thresholds.
    auc_obj = metrics.AUC(num_thresholds=2, thresholds=[0.5])
    self.assertEqual(auc_obj.num_thresholds, 3)
    self.assertAllClose(auc_obj.thresholds, [0.0, 0.5, 1.0])
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred)

    # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
    # fp_rate = [2/2, 0, 0] = [1, 0, 0]
    # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
    # widths = [(1 - 0), (0 - 0)] = [1, 0]
    expected_result = (0.75 * 1 + 0.25 * 0)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_weighted_roc_interpolation(self):
    self.setup()
    auc_obj = metrics.AUC(num_thresholds=self.num_thresholds)
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

    # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
    # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
    # fp_rate = [3/3, 0, 0] = [1, 0, 0]
    # heights = [(1 + 0.571)/2, (0.571 + 0)/2] = [0.7855, 0.2855]
    # widths = [(1 - 0), (0 - 0)] = [1, 0]
    expected_result = (0.7855 * 1 + 0.2855 * 0)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_weighted_roc_majoring(self):
    self.setup()
    auc_obj = metrics.AUC(
        num_thresholds=self.num_thresholds, summation_method='majoring')
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

    # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
    # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
    # fp_rate = [3/3, 0, 0] = [1, 0, 0]
    # heights = [max(1, 0.571), max(0.571, 0)] = [1, 0.571]
    # widths = [(1 - 0), (0 - 0)] = [1, 0]
    expected_result = (1 * 1 + 0.571 * 0)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_weighted_roc_minoring(self):
    self.setup()
    auc_obj = metrics.AUC(
        num_thresholds=self.num_thresholds, summation_method='minoring')
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

    # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
    # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
    # fp_rate = [3/3, 0, 0] = [1, 0, 0]
    # heights = [min(1, 0.571), min(0.571, 0)] = [0.571, 0]
    # widths = [(1 - 0), (0 - 0)] = [1, 0]
    expected_result = (0.571 * 1 + 0 * 0)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_weighted_pr_majoring(self):
    self.setup()
    auc_obj = metrics.AUC(
        num_thresholds=self.num_thresholds,
        curve='PR',
        summation_method='majoring')
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

    # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
    # precision = [7/(7+3), 4/4, 0] = [0.7, 1, 0]
    # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
    # heights = [max(0.7, 1), max(1, 0)] = [1, 1]
    # widths = [(1 - 0.571), (0.571 - 0)] = [0.429, 0.571]
    expected_result = (1 * 0.429 + 1 * 0.571)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_weighted_pr_minoring(self):
    self.setup()
    auc_obj = metrics.AUC(
        num_thresholds=self.num_thresholds,
        curve='PR',
        summation_method='minoring')
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

    # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
    # precision = [7/(7+3), 4/4, 0] = [0.7, 1, 0]
    # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
    # heights = [min(0.7, 1), min(1, 0)] = [0.7, 0]
    # widths = [(1 - 0.571), (0.571 - 0)] = [0.429, 0.571]
    expected_result = (0.7 * 0.429 + 0 * 0.571)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_weighted_pr_interpolation(self):
    self.setup()
    auc_obj = metrics.AUC(num_thresholds=self.num_thresholds, curve='PR')
    self.evaluate(variables.variables_initializer(auc_obj.variables))
    result = auc_obj(self.y_true, self.y_pred, sample_weight=self.sample_weight)

    # auc = (slope / Total Pos) * [dTP - intercept * log(Pb/Pa)]

    # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
    # P = tp + fp = [10, 4, 0]
    # dTP = [7-4, 4-0] = [3, 4]
    # dP = [10-4, 4-0] = [6, 4]
    # slope = dTP/dP = [0.5, 1]
    # intercept = (TPa+(slope*Pa) = [(4 - 0.5*4), (0 - 1*0)] = [2, 0]
    # (Pb/Pa) = (Pb/Pa) if Pb > 0 AND Pa > 0 else 1 = [10/4, 4/0] = [2.5, 1]
    # auc * TotalPos = [(0.5 * (3 + 2 * log(2.5))), (1 * (4 + 0))]
    #                = [2.416, 4]
    # auc = [2.416, 4]/(tp[1:]+fn[1:])
    expected_result = (2.416/7 + 4/7)
    self.assertAllClose(self.evaluate(result), expected_result, 1e-3)

  def test_invalid_num_thresholds(self):
    with self.assertRaisesRegexp(ValueError, '`num_thresholds` must be > 1.'):
      metrics.AUC(num_thresholds=-1)

    with self.assertRaisesRegexp(ValueError, '`num_thresholds` must be > 1.'):
      metrics.AUC(num_thresholds=1)

  def test_invalid_curve(self):
    with self.assertRaisesRegexp(ValueError,
                                 'Invalid AUC curve value "Invalid".'):
      metrics.AUC(curve='Invalid')

  def test_invalid_summation_method(self):
    with self.assertRaisesRegexp(
        ValueError, 'Invalid AUC summation method value "Invalid".'):
      metrics.AUC(summation_method='Invalid')


if __name__ == '__main__':
  test.main()
