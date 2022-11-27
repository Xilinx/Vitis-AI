# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils

# Abbreviated constants used in tests.
CHILDREN = core_encoder.EncoderKeys.CHILDREN
PARAMS = core_encoder.EncoderKeys.PARAMS
SHAPE = core_encoder.EncoderKeys.SHAPE
STATE = core_encoder.EncoderKeys.STATE
TENSORS = core_encoder.EncoderKeys.TENSORS
COMMUTE = core_encoder.EncoderKeys.COMMUTE

P1_VALS = test_utils.PlusOneEncodingStage.ENCODED_VALUES_KEY
P1_ADD_PARAM = test_utils.PlusOneEncodingStage.ADD_PARAM_KEY
T2_VALS = test_utils.TimesTwoEncodingStage.ENCODED_VALUES_KEY
T2_FACTOR_PARAM = test_utils.TimesTwoEncodingStage.FACTOR_PARAM_KEY
SL_VALS = test_utils.SimpleLinearEncodingStage.ENCODED_VALUES_KEY
RM_VALS = test_utils.ReduceMeanEncodingStage.ENCODED_VALUES_KEY
SIF_SIGNS = test_utils.SignIntFloatEncodingStage.ENCODED_SIGNS_KEY
SIF_INTS = test_utils.SignIntFloatEncodingStage.ENCODED_INTS_KEY
SIF_FLOATS = test_utils.SignIntFloatEncodingStage.ENCODED_FLOATS_KEY
PN_VALS = test_utils.PlusOneOverNEncodingStage.ENCODED_VALUES_KEY
PN_ITER_STATE = test_utils.PlusOneOverNEncodingStage.ITERATION_STATE_KEY
PN_ADD_PARAM = test_utils.PlusOneOverNEncodingStage.ADD_PARAM_KEY
AN_VALS = test_utils.AdaptiveNormalizeEncodingStage.ENCODED_VALUES_KEY
AN_FACTOR_PARAM = test_utils.AdaptiveNormalizeEncodingStage.FACTOR_PARAM_KEY
AN_FACTOR_STATE = test_utils.AdaptiveNormalizeEncodingStage.FACTOR_STATE_KEY
AN_NORM_UPDATE = test_utils.AdaptiveNormalizeEncodingStage.NORM_STATE_UPDATE_KEY


if tf.executing_eagerly():
  tf.compat.v1.disable_eager_execution()


class EncoderTest(tf.test.TestCase):

  def test_correct_structure(self):
    """Tests that structured objects look like what they should.

    This test creates the following encoding tree:
    SignIntFloatEncodingStage
        [SIF_SIGNS] -> TimesTwoEncodingStage
        [SIF_INTS] -> PlusOneEncodingStage
        [SIF_FLOATS] -> PlusOneOverNEncodingStage
            [PN_VALS] -> AdaptiveNormalizeEncodingStage
    And verifies that the structured objects created by the methods of `Encoder`
    are of the expected structure.
    """
    sif_stage = test_utils.SignIntFloatEncodingStage()
    times_two_stage = test_utils.TimesTwoEncodingStage()
    plus_one_stage = test_utils.PlusOneEncodingStage()
    plus_n_squared_stage = test_utils.PlusOneOverNEncodingStage()
    adaptive_normalize_stage = test_utils.AdaptiveNormalizeEncodingStage()

    encoder = core_encoder.EncoderComposer(sif_stage)
    encoder.add_child(times_two_stage, SIF_SIGNS)
    encoder.add_child(plus_one_stage, SIF_INTS)
    encoder.add_child(plus_n_squared_stage, SIF_FLOATS).add_child(
        adaptive_normalize_stage, PN_VALS)
    encoder = encoder.make()

    # Create all intermediary objects.
    x = tf.constant(1.0)
    initial_state = encoder.initial_state()
    encode_params, decode_params = encoder.get_params(initial_state)
    encoded_x, state_update_tensors, input_shapes = encoder.encode(
        x, encode_params)
    decoded_x = encoder.decode(encoded_x, decode_params, input_shapes)
    updated_state = encoder.update_state(initial_state, state_update_tensors)
    commuting_structure = encoder.commuting_structure

    # Verify the structure and naming of those objects is as expected.
    for state in [initial_state, updated_state]:
      tf.nest.assert_same_structure(
          {
              STATE: {},
              CHILDREN: {
                  SIF_INTS: {
                      STATE: {},
                      CHILDREN: {}
                  },
                  SIF_SIGNS: {
                      STATE: {},
                      CHILDREN: {}
                  },
                  SIF_FLOATS: {
                      STATE: {
                          PN_ITER_STATE: None
                      },
                      CHILDREN: {
                          PN_VALS: {
                              STATE: {
                                  AN_FACTOR_STATE: None
                              },
                              CHILDREN: {}
                          }
                      }
                  }
              }
          }, state)
    self.assertIn(
        'encoder_initial_state/' + sif_stage.name + '/' + SIF_FLOATS + '/' +
        plus_n_squared_stage.name + encoding_stage.INITIAL_STATE_SCOPE_SUFFIX,
        initial_state[CHILDREN][SIF_FLOATS][STATE][PN_ITER_STATE].name)
    self.assertIn(
        'encoder_initial_state/' + sif_stage.name + '/' + SIF_FLOATS + '/' +
        plus_n_squared_stage.name + '/' + PN_VALS + '/' +
        adaptive_normalize_stage.name +
        encoding_stage.INITIAL_STATE_SCOPE_SUFFIX, initial_state[CHILDREN]
        [SIF_FLOATS][CHILDREN][PN_VALS][STATE][AN_FACTOR_STATE].name)
    self.assertIn(
        'encoder_update_state/' + sif_stage.name + '/' + SIF_FLOATS + '/' +
        plus_n_squared_stage.name + encoding_stage.UPDATE_STATE_SCOPE_SUFFIX,
        updated_state[CHILDREN][SIF_FLOATS][STATE][PN_ITER_STATE].name)
    self.assertIn(
        'encoder_update_state/' + sif_stage.name + '/' + SIF_FLOATS + '/' +
        plus_n_squared_stage.name + '/' + PN_VALS + '/' +
        adaptive_normalize_stage.name +
        encoding_stage.UPDATE_STATE_SCOPE_SUFFIX, updated_state[CHILDREN]
        [SIF_FLOATS][CHILDREN][PN_VALS][STATE][AN_FACTOR_STATE].name)

    for params in [encode_params, decode_params]:
      tf.nest.assert_same_structure(
          {
              PARAMS: {},
              CHILDREN: {
                  SIF_INTS: {
                      PARAMS: {
                          P1_ADD_PARAM: None
                      },
                      CHILDREN: {}
                  },
                  SIF_SIGNS: {
                      PARAMS: {
                          T2_FACTOR_PARAM: None
                      },
                      CHILDREN: {}
                  },
                  SIF_FLOATS: {
                      PARAMS: {
                          PN_ADD_PARAM: None
                      },
                      CHILDREN: {
                          PN_VALS: {
                              PARAMS: {
                                  AN_FACTOR_PARAM: None
                              },
                              CHILDREN: {}
                          }
                      }
                  }
              }
          }, params)
      self.assertIn(
          'encoder_get_params/' + sif_stage.name + '/' + SIF_INTS + '/' +
          plus_one_stage.name + encoding_stage.GET_PARAMS_SCOPE_SUFFIX,
          params[CHILDREN][SIF_INTS][PARAMS][P1_ADD_PARAM].name)
      self.assertIn(
          'encoder_get_params/' + sif_stage.name + '/' + SIF_SIGNS + '/' +
          times_two_stage.name + encoding_stage.GET_PARAMS_SCOPE_SUFFIX,
          params[CHILDREN][SIF_SIGNS][PARAMS][T2_FACTOR_PARAM].name)
      self.assertIn(
          'encoder_get_params/' + sif_stage.name + '/' + SIF_FLOATS + '/' +
          plus_n_squared_stage.name + encoding_stage.GET_PARAMS_SCOPE_SUFFIX,
          params[CHILDREN][SIF_FLOATS][PARAMS][PN_ADD_PARAM].name)
      # Note: we do not check the value of
      # params[CHILDREN][SIF_FLOATS][CHILDREN][PN_VALS][PARAMS][AN_FACTOR_PARAM]
      # because the get_params method of adaptive_normalize_stage does not
      # modify the graph, only passes through the provided state tensor.

    tf.nest.assert_same_structure(
        {
            SIF_INTS: {
                P1_VALS: None
            },
            SIF_SIGNS: {
                T2_VALS: None
            },
            SIF_FLOATS: {
                PN_VALS: {
                    AN_VALS: None
                }
            }
        }, encoded_x)
    self.assertIn(
        'encoder_encode/' + sif_stage.name + '/' + SIF_INTS + '/' +
        plus_one_stage.name + encoding_stage.ENCODE_SCOPE_SUFFIX,
        encoded_x[SIF_INTS][P1_VALS].name)
    self.assertIn(
        'encoder_encode/' + sif_stage.name + '/' + SIF_SIGNS + '/' +
        times_two_stage.name + encoding_stage.ENCODE_SCOPE_SUFFIX,
        encoded_x[SIF_SIGNS][T2_VALS].name)
    self.assertIn(
        'encoder_encode/' + sif_stage.name + '/' + SIF_FLOATS + '/' +
        plus_n_squared_stage.name + '/' + PN_VALS + '/' +
        adaptive_normalize_stage.name + encoding_stage.ENCODE_SCOPE_SUFFIX,
        encoded_x[SIF_FLOATS][PN_VALS][AN_VALS].name)

    tf.nest.assert_same_structure(
        {
            TENSORS: {},
            CHILDREN: {
                SIF_INTS: {
                    TENSORS: {},
                    CHILDREN: {}
                },
                SIF_SIGNS: {
                    TENSORS: {},
                    CHILDREN: {}
                },
                SIF_FLOATS: {
                    TENSORS: {},
                    CHILDREN: {
                        PN_VALS: {
                            TENSORS: {
                                AN_NORM_UPDATE: None
                            },
                            CHILDREN: {}
                        }
                    }
                }
            }
        }, state_update_tensors)
    tf.nest.assert_same_structure(state_update_tensors,
                                  encoder.state_update_aggregation_modes)
    self.assertIn(
        'encoder_encode/' + sif_stage.name + '/' + SIF_FLOATS + '/' +
        plus_n_squared_stage.name + '/' + PN_VALS + '/' +
        adaptive_normalize_stage.name + encoding_stage.ENCODE_SCOPE_SUFFIX,
        state_update_tensors[CHILDREN][SIF_FLOATS][CHILDREN][PN_VALS][TENSORS]
        [AN_NORM_UPDATE].name)

    tf.nest.assert_same_structure(
        {
            SHAPE: None,
            CHILDREN: {
                SIF_INTS: {
                    SHAPE: None,
                    CHILDREN: {}
                },
                SIF_SIGNS: {
                    SHAPE: None,
                    CHILDREN: {}
                },
                SIF_FLOATS: {
                    SHAPE: None,
                    CHILDREN: {
                        PN_VALS: {
                            SHAPE: None,
                            CHILDREN: {}
                        }
                    }
                }
            }
        }, input_shapes)
    self.assertTrue(tf.is_tensor(decoded_x))
    self.assertIn('encoder_decode/', decoded_x.name)

    tf.nest.assert_same_structure(
        {
            COMMUTE: None,
            CHILDREN: {
                SIF_INTS: {
                    COMMUTE: None,
                    CHILDREN: {}
                },
                SIF_SIGNS: {
                    COMMUTE: None,
                    CHILDREN: {}
                },
                SIF_FLOATS: {
                    COMMUTE: None,
                    CHILDREN: {
                        PN_VALS: {
                            COMMUTE: None,
                            CHILDREN: {}
                        }
                    }
                }
            }
        }, commuting_structure)
    for item in tf.nest.flatten(commuting_structure):
      self.assertEqual(False, item)

  # A utility for tests testing commutation with sum works as expected.
  commutation_test_data = collections.namedtuple(
      'comm_test_data',
      ['x', 'encoded_x', 'decoded_x_before_sum', 'decoded_x_after_sum'])

  def _data_for_test_decode_split(self, encoder, x):
    encode_params, decode_params = encoder.get_params(encoder.initial_state())
    encoded_x, _, input_shapes = encoder.encode(x, encode_params)
    decoded_x_before_sum = encoder.decode_before_sum(encoded_x, decode_params,
                                                     input_shapes)
    decoded_x_after_sum = encoder.decode_after_sum(decoded_x_before_sum,
                                                   decode_params, 1,
                                                   input_shapes)
    data = self.commutation_test_data(x, encoded_x, decoded_x_before_sum,
                                      decoded_x_after_sum)
    return self.evaluate(data)

  def test_decode_split_commutes_with_sum_false_false(self):
    """Tests that splitting decode works as expected with commutes_with_sum.

    This test chains two encoding stages, first *does not* commute with sum, the
    second *does not*, either. Together, nothing should commute with sum.
    """
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneEncodingStage()).add_parent(
            test_utils.PlusOneEncodingStage(), P1_VALS).make()
    self.assertFalse(encoder.fully_commutes_with_sum)
    data = self._data_for_test_decode_split(encoder, tf.constant(3.0))

    # Test the encoding is as expected.
    self.assertEqual(data.x, data.decoded_x_after_sum)
    self.assertAllEqual({
        P1_VALS: {
            P1_VALS: data.x + 1.0 + 1.0
        },
    }, data.encoded_x)
    # Nothing commutes with sum - decoded_x_before_sum should be fully decoded.
    self.assertEqual(data.x, data.decoded_x_before_sum)
    self.assertEqual(
        {
            COMMUTE: False,
            CHILDREN: {
                P1_VALS: {
                    COMMUTE: False,
                    CHILDREN: {}
                }
            }
        }, encoder.commuting_structure)

  def test_decode_split_commutes_with_sum_false_true(self):
    """Tests that splitting decode works as expected with commutes_with_sum.

    This test chains two encoding stages, first *does not* commute with sum, the
    second *does*. Together, nothing should commute with sum.
    """
    encoder = core_encoder.EncoderComposer(
        test_utils.TimesTwoEncodingStage()).add_parent(
            test_utils.PlusOneEncodingStage(), P1_VALS).make()
    self.assertFalse(encoder.fully_commutes_with_sum)
    data = self._data_for_test_decode_split(encoder, tf.constant(3.0))

    # Test the encoding is as expected.
    self.assertEqual(data.x, data.decoded_x_after_sum)
    self.assertAllEqual({
        P1_VALS: {
            T2_VALS: (data.x + 1.0) * 2.0
        },
    }, data.encoded_x)
    # Nothing commutes with sum - decoded_x_before_sum should be fully decoded.
    self.assertEqual(data.x, data.decoded_x_before_sum)
    self.assertEqual(
        {
            COMMUTE: False,
            CHILDREN: {
                P1_VALS: {
                    COMMUTE: False,
                    CHILDREN: {}
                }
            }
        }, encoder.commuting_structure)

  def test_decode_split_commutes_with_sum_true_true(self):
    """Tests that splitting decode works as expected with commutes_with_sum.

    This test chains two encoding stages, first *does* commute with sum, the
    second *does*, too. Together, everything should commute with sum.
    """
    encoder = core_encoder.EncoderComposer(
        test_utils.TimesTwoEncodingStage()).add_parent(
            test_utils.TimesTwoEncodingStage(), T2_VALS).make()
    self.assertTrue(encoder.fully_commutes_with_sum)
    data = self._data_for_test_decode_split(encoder, tf.constant(3.0))

    # Test the encoding is as expected.
    self.assertEqual(data.x, data.decoded_x_after_sum)
    self.assertAllEqual({
        T2_VALS: {
            T2_VALS: data.x * 2.0 * 2.0
        },
    }, data.encoded_x)
    # Everyting commutes with sum - decoded_x_before_sum should be intact.
    self.assertAllEqual(data.encoded_x, data.decoded_x_before_sum)
    self.assertEqual(
        {
            COMMUTE: True,
            CHILDREN: {
                T2_VALS: {
                    COMMUTE: True,
                    CHILDREN: {}
                }
            }
        }, encoder.commuting_structure)

  def test_decode_split_commutes_with_sum_true_false(self):
    """Tests that splitting decode works as expected with commutes_with_sum.

    This test chains two encoding stages, first *does* commute with sum, the
    second *does not*. Together, only the first one should commute with sum.
    """
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneEncodingStage()).add_parent(
            test_utils.TimesTwoEncodingStage(), T2_VALS).make()
    self.assertFalse(encoder.fully_commutes_with_sum)
    data = self._data_for_test_decode_split(encoder, tf.constant(3.0))

    # Test the encoding is as expected.
    self.assertEqual(data.x, data.decoded_x_after_sum)
    self.assertAllEqual({
        T2_VALS: {
            P1_VALS: data.x * 2.0 + 1.0
        },
    }, data.encoded_x)
    # Only first part commutes with sum.
    self.assertAllEqual({T2_VALS: data.x * 2.0}, data.decoded_x_before_sum)
    self.assertEqual(
        {
            COMMUTE: True,
            CHILDREN: {
                T2_VALS: {
                    COMMUTE: False,
                    CHILDREN: {}
                }
            }
        }, encoder.commuting_structure)

  def test_decode_split_commutes_with_sum_true_false_true(self):
    """Tests that splitting decode works as expected with commutes_with_sum.

    This test chains three encoding stages, first *does* commute with sum, the
    second *does not*, and third *does*, again. Together, only the first one
    should commute with sum, and the rest should not.
    """
    encoder = core_encoder.EncoderComposer(
        test_utils.TimesTwoEncodingStage()).add_parent(
            test_utils.PlusOneEncodingStage(), P1_VALS).add_parent(
                test_utils.TimesTwoEncodingStage(), T2_VALS).make()
    self.assertFalse(encoder.fully_commutes_with_sum)
    data = self._data_for_test_decode_split(encoder, tf.constant(3.0))

    # Test the encoding is as expected.
    self.assertEqual(data.x, data.decoded_x_after_sum)
    self.assertAllEqual({
        T2_VALS: {
            P1_VALS: {
                T2_VALS: (data.x * 2.0 + 1.0) * 2.0
            }
        },
    }, data.encoded_x)
    # Only first part commutes with sum.
    self.assertAllEqual({T2_VALS: data.x * 2.0}, data.decoded_x_before_sum)
    self.assertEqual(
        {
            COMMUTE: True,
            CHILDREN: {
                T2_VALS: {
                    COMMUTE: False,
                    CHILDREN: {
                        P1_VALS: {
                            COMMUTE: False,
                            CHILDREN: {}
                        },
                    }
                }
            }
        }, encoder.commuting_structure)

  def test_commutes_with_sum(self):
    """Tests that commutativity works, provided appropriate num_summands."""
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneEncodingStage()).add_parent(
            test_utils.SimpleLinearEncodingStage(2.0, 3.0), SL_VALS).make()

    x = tf.constant(3.0)
    encode_params, decode_params = encoder.get_params(encoder.initial_state())
    encoded_x, _, input_shapes = encoder.encode(x, encode_params)
    decoded_x_before_sum = encoder.decode_before_sum(encoded_x, decode_params,
                                                     input_shapes)
    # Trivial summing of the encoded - and partially decoded - values.
    part_decoded_and_summed_x = tf.nest.map_structure(lambda x: x + x + x,
                                                      decoded_x_before_sum)
    num_summands = 3
    decoded_x_after_sum = encoder.decode_after_sum(part_decoded_and_summed_x,
                                                   decode_params, num_summands,
                                                   input_shapes)
    data = self.evaluate(
        self.commutation_test_data(x, encoded_x, decoded_x_before_sum,
                                   decoded_x_after_sum))
    self.assertEqual(3.0, data.x)
    expected_encoded_x = {SL_VALS: {P1_VALS: (data.x * 2.0 + 3.0) + 1.0}}
    self.assertAllEqual(expected_encoded_x, data.encoded_x)
    expected_decoded_x_before_sum = {SL_VALS: data.x * 2.0 + 3.0}
    self.assertAllEqual(expected_decoded_x_before_sum,
                        data.decoded_x_before_sum)
    self.assertEqual(9.0, data.decoded_x_after_sum)

  def test_decode_needs_input_shape(self):
    """Tests that encoder works with stages that need input shape for decode.

    This test chains two stages with this property.
    """
    encoder = core_encoder.EncoderComposer(
        test_utils.ReduceMeanEncodingStage()).add_parent(
            test_utils.ReduceMeanEncodingStage(), RM_VALS).make()
    x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    encode_params, decode_params = encoder.get_params(encoder.initial_state())
    encoded_x, _, input_shapes = encoder.encode(x, encode_params)
    decoded_x = encoder.decode(encoded_x, decode_params, input_shapes)
    encoded_x, decoded_x = self.evaluate([encoded_x, decoded_x])

    self.assertAllEqual([3.0] * 5, decoded_x)
    self.assertAllEqual({RM_VALS: {RM_VALS: 3.0}}, encoded_x)

  def test_decode_needs_input_shape_unknown_input_shape(self):
    """Tests that encoder works with stages that need input shape for decode.

    This test chains two stages with this property, and provides an input with
    statically unknown shape information.
    """
    encoder = core_encoder.EncoderComposer(
        test_utils.ReduceMeanEncodingStage()).add_parent(
            test_utils.ReduceMeanEncodingStage(), RM_VALS).make()
    x = test_utils.get_tensor_with_random_shape()
    encode_params, decode_params = encoder.get_params(encoder.initial_state())
    encoded_x, _, input_shapes = encoder.encode(x, encode_params)
    decoded_x = encoder.decode(encoded_x, decode_params, input_shapes)
    assert x.shape.as_list()[0] is None  # Validate the premise of the test.
    x, decoded_x = self.evaluate([x, decoded_x])

    # Assert shape is correctly recovered, and finctionality is as expected.
    self.assertAllEqual(x.shape, decoded_x.shape)
    self.assertAllClose([x.mean()] * len(x), decoded_x)

  def test_tree_encoder(self):
    """Tests that the encoder works as a proper tree, not only a chain."""
    encoder = core_encoder.EncoderComposer(
        test_utils.SignIntFloatEncodingStage())
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_SIGNS)
    encoder.add_child(test_utils.PlusOneEncodingStage(), SIF_INTS)
    encoder.add_child(test_utils.PlusOneEncodingStage(), SIF_FLOATS).add_child(
        test_utils.TimesTwoEncodingStage(), P1_VALS)
    encoder = encoder.make()

    x = tf.constant([0.0, 0.1, -0.1, 0.9, -0.9, 1.6, -2.2])
    encode_params, decode_params = encoder.get_params(encoder.initial_state())
    encoded_x, _, input_shapes = encoder.encode(x, encode_params)
    decoded_x = encoder.decode(encoded_x, decode_params, input_shapes)
    x, encoded_x, decoded_x = self.evaluate([x, encoded_x, decoded_x])

    self.assertAllClose(x, decoded_x)
    expected_encoded_x = {
        SIF_SIGNS: {
            T2_VALS: np.array([0.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
        },
        SIF_INTS: {
            P1_VALS: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0])
        },
        SIF_FLOATS: {
            P1_VALS: {
                T2_VALS: np.array([2.0, 2.2, 2.2, 3.8, 3.8, 3.2, 2.4])
            }
        }
    }
    self.assertAllClose(expected_encoded_x, encoded_x)

  def test_encoder_is_reusable(self):
    """Tests that the same encoder can be used to encode multiple objects."""
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneEncodingStage()).add_parent(
            test_utils.TimesTwoEncodingStage(), T2_VALS).make()
    x_vals = [tf.random.normal(shape) for shape in [(3,), (3, 4), (3, 4, 5)]]
    for x in x_vals:
      encode_params, decode_params = encoder.get_params(
          encoder.initial_state())
      encoded_x, _, input_shapes = encoder.encode(x, encode_params)
      decoded_x = encoder.decode(encoded_x, decode_params, input_shapes)
      x, encoded_x, decoded_x = self.evaluate([x, encoded_x, decoded_x])

      self.assertAllClose(x, decoded_x)
      self.assertAllClose({T2_VALS: {P1_VALS: x * 2.0 + 1.0}}, encoded_x)

  def test_adaptive_stage(self):
    """Tests composition of two adaptive encoding stages."""
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneOverNEncodingStage()).add_parent(
            test_utils.PlusOneOverNEncodingStage(), PN_VALS).make()
    x = tf.constant(1.0)
    state = encoder.initial_state()

    for i in range(1, 5):
      initial_state = state
      encode_params, decode_params = encoder.get_params(state)
      encoded_x, state_update_tensors, input_shapes = encoder.encode(
          x, encode_params)
      decoded_x = encoder.decode(encoded_x, decode_params, input_shapes)
      state = encoder.update_state(initial_state, state_update_tensors)
      data = self.evaluate(
          test_utils.TestData(x, encoded_x, decoded_x, initial_state,
                              state_update_tensors, state))

      expected_initial_state = {
          STATE: {
              PN_ITER_STATE: i
          },
          CHILDREN: {
              PN_VALS: {
                  STATE: {
                      PN_ITER_STATE: i
                  },
                  CHILDREN: {}
              }
          }
      }
      expected_state_update_tensors = {
          TENSORS: {},
          CHILDREN: {
              PN_VALS: {
                  TENSORS: {},
                  CHILDREN: {}
              }
          }
      }

      self.assertAllClose(data.x, data.decoded_x)
      self.assertAllEqual(expected_initial_state, data.initial_state)
      self.assertDictEqual(expected_state_update_tensors,
                           data.state_update_tensors)
      self.assertAllClose(data.x + 2 * 1 / i, data.encoded_x[PN_VALS][PN_VALS])

  def test_adaptive_stage_using_state_update_tensors(self):
    """Tests adaptive encoding stage with state update tensors."""
    encoder = core_encoder.EncoderComposer(
        test_utils.AdaptiveNormalizeEncodingStage()).add_parent(
            test_utils.PlusOneEncodingStage(), P1_VALS).make()
    x = tf.constant(1.0)
    state = encoder.initial_state()

    for _ in range(1, 5):
      initial_state = state
      encode_params, decode_params = encoder.get_params(state)
      encoded_x, state_update_tensors, input_shapes = encoder.encode(
          x, encode_params)
      decoded_x = encoder.decode(encoded_x, decode_params, input_shapes)
      state = encoder.update_state(initial_state, state_update_tensors)
      data = self.evaluate(
          test_utils.TestData(x, encoded_x, decoded_x, initial_state,
                              state_update_tensors, state))

      self.assertAllClose(data.x, data.decoded_x)
      self.assertLessEqual(
          data.initial_state[CHILDREN][P1_VALS][STATE][AN_FACTOR_STATE], 1.0)
      self.assertEqual(
          data.state_update_tensors[CHILDREN][P1_VALS][TENSORS][AN_NORM_UPDATE],
          2.0)
      self.assertLessEqual(data.encoded_x[P1_VALS][AN_VALS], 2.0)


class EncoderComposerTest(tf.test.TestCase):

  def test_add_parent(self):
    encoder = core_encoder.EncoderComposer(
        test_utils.ReduceMeanEncodingStage()).add_parent(
            test_utils.PlusOneEncodingStage(), P1_VALS).add_parent(
                test_utils.TimesTwoEncodingStage(), T2_VALS).make()

    self.assertIsInstance(encoder, core_encoder.Encoder)
    self.assertIsInstance(encoder.stage._wrapped_stage,
                          test_utils.TimesTwoEncodingStage)
    self.assertIsInstance(encoder.children[T2_VALS], core_encoder.Encoder)
    self.assertIsInstance(encoder.children[T2_VALS].stage._wrapped_stage,
                          test_utils.PlusOneEncodingStage)
    self.assertIsInstance(encoder.children[T2_VALS].children[P1_VALS],
                          core_encoder.Encoder)
    self.assertIsInstance(
        encoder.children[T2_VALS].children[P1_VALS].stage._wrapped_stage,
        test_utils.ReduceMeanEncodingStage)

  def test_add_child(self):
    encoder = core_encoder.EncoderComposer(test_utils.TimesTwoEncodingStage())
    encoder.add_child(test_utils.PlusOneEncodingStage(), T2_VALS).add_child(
        test_utils.ReduceMeanEncodingStage(), P1_VALS)
    encoder = encoder.make()

    self.assertIsInstance(encoder, core_encoder.Encoder)
    self.assertIsInstance(encoder.stage._wrapped_stage,
                          test_utils.TimesTwoEncodingStage)
    self.assertIsInstance(encoder.children[T2_VALS], core_encoder.Encoder)
    self.assertIsInstance(encoder.children[T2_VALS].stage._wrapped_stage,
                          test_utils.PlusOneEncodingStage)
    self.assertIsInstance(encoder.children[T2_VALS].children[P1_VALS],
                          core_encoder.Encoder)
    self.assertIsInstance(
        encoder.children[T2_VALS].children[P1_VALS].stage._wrapped_stage,
        test_utils.ReduceMeanEncodingStage)

  def test_add_child_semantics(self):
    composer = core_encoder.EncoderComposer(test_utils.TimesTwoEncodingStage())
    composer.add_child(test_utils.PlusOneEncodingStage(), T2_VALS)
    encoder_1 = composer.make()
    encoder_2 = core_encoder.EncoderComposer(
        test_utils.TimesTwoEncodingStage()).add_child(
            test_utils.PlusOneEncodingStage(), T2_VALS).make()

    # Assert that these produce different trees. The add_child method returns
    # the newly created node, and thus the make creates only the child node.
    self.assertNotEqual(encoder_1.children.keys(), encoder_2.children.keys())

  def test_constructor_raises(self):
    with self.assertRaises(TypeError):
      core_encoder.EncoderComposer('not an encoding stage')

  def test_add_child_parent_bad_key_raises(self):
    encoder = core_encoder.EncoderComposer(test_utils.TimesTwoEncodingStage())
    with self.assertRaises(KeyError):
      encoder.add_child(test_utils.PlusOneEncodingStage(), '___bad_key')
    with self.assertRaises(KeyError):
      encoder.add_parent(test_utils.PlusOneEncodingStage(), '___bad_key')

  def test_add_child_repeat_key_raises(self):
    encoder = core_encoder.EncoderComposer(test_utils.TimesTwoEncodingStage())
    encoder.add_child(test_utils.PlusOneEncodingStage(), T2_VALS)
    with self.assertRaises(KeyError):
      encoder.add_child(test_utils.PlusOneEncodingStage(), T2_VALS)


def _get_test_commuting_structure():
  return {
      COMMUTE: True,
      CHILDREN: {
          'key_1': {
              COMMUTE: True,
              CHILDREN: {
                  'key_4': {
                      COMMUTE: False,
                      CHILDREN: {}
                  }
              }
          },
          'key_2': {
              COMMUTE: False,
              CHILDREN: {},
          },
          'key_3': {
              COMMUTE: True,
              CHILDREN: {},
          }
      }
  }


class UtilsTest(tf.test.TestCase):
  """Test for utilities created in core_encoder.py."""

  def test_split_params_by_commuting_structure(self):
    commuting_structure = _get_test_commuting_structure()
    params = {
        PARAMS: {
            'param_a': 0.0,
            'param_b': (-1.0, 1.0)
        },
        CHILDREN: {
            'key_1': {
                PARAMS: {
                    'param_c': None
                },
                CHILDREN: {
                    'key_4': {
                        PARAMS: {
                            'param_e': (-1.0, 1.0)
                        },
                        CHILDREN: {}
                    }
                }
            },
            'key_2': {
                PARAMS: {},
                CHILDREN: {},
            },
            'key_3': {
                PARAMS: {
                    'param_d': [1, 2]
                },
                CHILDREN: {},
            }
        }
    }
    expected_before_sum_params = {
        PARAMS: {
            'param_a': None,
            'param_b': None
        },
        CHILDREN: {
            'key_1': {
                PARAMS: {
                    'param_c': None
                },
                CHILDREN: {
                    'key_4': {
                        PARAMS: {
                            'param_e': (-1.0, 1.0)
                        },
                        CHILDREN: {}
                    }
                }
            },
            'key_2': {
                PARAMS: {},
                CHILDREN: {},
            },
            'key_3': {
                PARAMS: {
                    'param_d': None
                },
                CHILDREN: {},
            }
        }
    }
    expected_after_sum_params = {
        PARAMS: {
            'param_a': 0.0,
            'param_b': (-1.0, 1.0)
        },
        CHILDREN: {
            'key_1': {
                PARAMS: {
                    'param_c': None
                },
                CHILDREN: {
                    'key_4': {
                        PARAMS: {
                            'param_e': None
                        },
                        CHILDREN: {}
                    }
                }
            },
            'key_2': {
                PARAMS: {},
                CHILDREN: {},
            },
            'key_3': {
                PARAMS: {
                    'param_d': [1, 2]
                },
                CHILDREN: {},
            }
        }
    }

    before_sum_params, after_sum_params = (
        core_encoder.split_params_by_commuting_structure(
            params, commuting_structure))
    self.assertEqual(expected_before_sum_params, before_sum_params)
    self.assertEqual(expected_after_sum_params, after_sum_params)

  def test_split_shapes_by_commuting_structure(self):
    commuting_structure = _get_test_commuting_structure()
    shapes = {
        SHAPE: (5, 5),
        CHILDREN: {
            'key_1': {
                SHAPE: None,
                CHILDREN: {
                    'key_4': {
                        SHAPE: (1, 4),
                        CHILDREN: {}
                    }
                }
            },
            'key_2': {
                SHAPE: (1, 2),
                CHILDREN: {},
            },
            'key_3': {
                SHAPE: (1, 3),
                CHILDREN: {},
            }
        }
    }
    expected_before_sum_shapes = {
        SHAPE: None,
        CHILDREN: {
            'key_1': {
                SHAPE: None,
                CHILDREN: {
                    'key_4': {
                        SHAPE: (1, 4),
                        CHILDREN: {}
                    }
                }
            },
            'key_2': {
                SHAPE: (1, 2),
                CHILDREN: {},
            },
            'key_3': {
                SHAPE: None,
                CHILDREN: {},
            }
        }
    }
    expected_after_sum_shapes = {
        SHAPE: (5, 5),
        CHILDREN: {
            'key_1': {
                SHAPE: None,
                CHILDREN: {
                    'key_4': {
                        SHAPE: None,
                        CHILDREN: {}
                    }
                }
            },
            'key_2': {
                SHAPE: None,
                CHILDREN: {},
            },
            'key_3': {
                SHAPE: (1, 3),
                CHILDREN: {},
            }
        }
    }

    before_sum_shapes, after_sum_shapes = (
        core_encoder.split_shapes_by_commuting_structure(
            shapes, commuting_structure))
    self.assertEqual(expected_before_sum_shapes, before_sum_shapes)
    self.assertEqual(expected_after_sum_shapes, after_sum_shapes)


if __name__ == '__main__':
  tf.test.main()
