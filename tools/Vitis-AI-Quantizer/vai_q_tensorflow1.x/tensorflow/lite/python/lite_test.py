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
"""Tests for lite.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from absl.testing import parameterized
import numpy as np

from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.variables import global_variables_initializer as _global_variables_initializer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.training_util import write_graph


class TestModels(test_util.TensorFlowTestCase):

  def assertValidDebugInfo(self, debug_info):
    """Verify the DebugInfo is valid."""
    file_names = set()
    for file_path in debug_info.files:
      file_names.add(os.path.basename(file_path))
    # To make the test independent on how the nodes are created, we only assert
    # the name of this test file.
    self.assertIn('lite_test.py', file_names)
    self.assertNotIn('lite_v2_test.py', file_names)


class FromConstructor(TestModels):

  # Tests invalid constructors using a dummy value for the GraphDef.
  def testInvalidConstructor(self):
    message = ('If input_tensors and output_tensors are None, both '
               'input_arrays_with_shape and output_arrays must be defined.')

    # `output_arrays` is not defined.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter(
          None, None, [], input_arrays_with_shape=[('input', [3, 9])])
    self.assertEqual(message, str(error.exception))

    # `input_arrays_with_shape` is not defined.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter(None, [], None, output_arrays=['output'])
    self.assertEqual(message, str(error.exception))

  # Tests valid constructors using a dummy value for the GraphDef.
  def testValidConstructor(self):
    converter = lite.TFLiteConverter(
        None,
        None,
        None,
        input_arrays_with_shape=[('input', [3, 9])],
        output_arrays=['output'])
    self.assertFalse(converter._has_valid_tensors())
    self.assertEqual(converter.get_input_arrays(), ['input'])

    with self.assertRaises(ValueError) as error:
      converter._set_batch_size(1)
    self.assertEqual(
        'The batch size cannot be set for this model. Please use '
        'input_shapes parameter.', str(error.exception))

    converter = lite.TFLiteConverter(None, ['input_tensor'], ['output_tensor'])
    self.assertTrue(converter._has_valid_tensors())


class FromSessionTest(TestModels, parameterized.TestCase):

  def testFloat(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testString(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.string)
      out_tensor = array_ops.reshape(in_tensor, shape=[2, 2])
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.string_, input_details[0]['dtype'])
    self.assertTrue(([4] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('Reshape', output_details[0]['name'])
    self.assertEqual(np.string_, output_details[0]['dtype'])
    self.assertTrue(([2, 2] == output_details[0]['shape']).all())
    # TODO(b/122659643): Test setting/getting string data via the python
    # interpreter API after support has been added.

  def testQuantization(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {
        'inputA': (0., 1.),
        'inputB': (0., 1.)
    }  # mean, std_dev
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((1., 0.),
                     input_details[0]['quantization'])  # scale, zero_point

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.uint8, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((1., 0.),
                     input_details[1]['quantization'])  # scale, zero_point

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertTrue(output_details[0]['quantization'][0] > 0)  # scale

  def testQuantizationInvalid(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {'inputA': (0., 1.)}  # mean, std_dev
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'Quantization input stats are not available for input tensors '
        '\'inputB\'.', str(error.exception))

  def testIntermediateInputArray(self):
    """Convert a model from an intermediate input array."""
    with ops.Graph().as_default():
      in_tensor_init = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      in_tensor_final = in_tensor_init + in_tensor_init
      out_tensor = in_tensor_final + in_tensor_final
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor_final],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('add', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add_1', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testSizeNoneInvalid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test None as shape.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual('Provide an input shape for input array \'Placeholder\'.',
                     str(error.exception))

  def testScalarValid(self):
    # Construct a graph using a scalar (empty shape) input.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test conversion with the scalar input shape.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([] == input_details[0]['shape']).all())

    # Validate inference using the scalar inputs/outputs.
    test_input = np.array(4.0, dtype=np.float32)
    expected_output = np.array(8.0, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testSizeInvalid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, None, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test invalid shape. None after 1st dimension.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'None is only supported in the 1st dimension. Tensor '
        '\'Placeholder\' has invalid shape \'[1, None, 16, 3]\'.',
        str(error.exception))

  def testBatchSizeValid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testBatchSizeNonZero(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[None, 4], dtype=dtypes.float32, name='input1')
      in_tensor_2 = array_ops.placeholder(
          shape=[4, 10], dtype=dtypes.float32, name='input2')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2)
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('input1', input_details[0]['name'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())
    self.assertEqual('input2', input_details[1]['name'])
    self.assertTrue(([4, 10] == input_details[1]['shape']).all())

  def testFreezeGraph(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      var = variable_scope.get_variable(
          'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + var
      sess = session.Session()
      sess.run(_global_variables_initializer())

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testGraphviz(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.output_format = lite_constants.GRAPHVIZ_DOT
    graphviz_output = converter.convert()
    self.assertTrue(graphviz_output)

  def testDumpGraphviz(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    graphviz_dir = self.get_temp_dir()
    converter.dump_graphviz_dir = graphviz_dir
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure interpreter is able to allocate and check graphviz data.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    num_items_graphviz = len(os.listdir(graphviz_dir))
    self.assertTrue(num_items_graphviz)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    graphviz_dir = self.get_temp_dir()
    converter.dump_graphviz_dir = graphviz_dir
    converter.dump_graphviz_video = True
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure graphviz folder has more data after using video flag.
    num_items_graphviz_video = len(os.listdir(graphviz_dir))
    self.assertTrue(num_items_graphviz_video > num_items_graphviz)

  def testInferenceInputType(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_input_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())

  def testDefaultRangesStats(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    converter.default_ranges_stats = (0, 6)  # min, max
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertTrue(output_details[0]['quantization'][0] > 0)  # scale

  def testPostTrainingQuantizeDeprecatedAttribute(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    self.assertFalse(quantized_converter.post_training_quantize)

    quantized_converter.post_training_quantize = True
    self.assertTrue(quantized_converter.post_training_quantize)
    self.assertEqual(quantized_converter.optimizations, [lite.Optimize.DEFAULT])

    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

  def testPostTrainingQuantize(self):
    np.random.seed(0)
    with ops.Graph().as_default():
      # We need the tensor to have more than 1024 elements for quantize_weights
      # to kick in. Thus, the [33, 33] shape.
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1],
                                                        [out_tensor])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

    # Ensure that the quantized weights tflite model is smaller.
    self.assertTrue(len(quantized_tflite) < len(float_tflite))

  def _getCalibrationQuantizeModel(self):
    np.random.seed(0)
    inp = array_ops.placeholder(
        dtype=dtypes.float32, shape=(1, 5, 5, 3), name='input')
    conv = nn_ops.conv2d(
        inp,
        filter=array_ops.ones([3, 3, 3, 16]),
        strides=[1, 1, 1, 1],
        padding='SAME')
    output = nn_ops.relu(conv, name='output')

    def calibration_gen():
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]

    return (inp, output, calibration_gen)

  def testPostTrainingCalibrateAndQuantize(self):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getCalibrationQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite), len(float_tflite))

  def testCalibrateAndQuantizeBuiltinInt8(self):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getCalibrationQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert model by specifying target spec (instead of optimizations), since
    # when targeting an integer only backend, quantization is mandatory.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite), len(float_tflite))

  @parameterized.named_parameters(
      # Quantize to Float16 even if rep data provided.
      ('UseRepresentativeData', True, False, True, False, False),
      # Quantize to Float16 if no rep data provided.
      ('NoRepresentativeData', False, False, True, False, False),
      # Post training quantization if both rep data and int8 included.
      ('UseRepresentativeDataIncludeInt8', True, True, False, False, True),
      # Error if no rep data and int8 included.
      ('NoRepresentativeDataIncludeInt8', False, True, False, True, False))
  def testQuantizeFloat16(self, use_rep_data, include_int8,
                          is_float16_quantized, is_error,
                          is_post_training_quantized):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getCalibrationQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)
    interpreter = Interpreter(model_content=float_tflite)
    interpreter.allocate_tensors()
    self.assertEqual(interpreter.get_tensor_details()[0]['name'], 'Conv2D_bias')
    self.assertEqual(interpreter.get_tensor_details()[0]['dtype'],
                     lite.constants.FLOAT)
    # Convert model to quantized version
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_types = [lite.constants.FLOAT16]
    if include_int8:
      quantized_converter.target_spec.supported_types.append(
          lite.constants.INT8)
    if use_rep_data:
      quantized_converter.representative_dataset = calibration_gen

    if is_error:
      with self.assertRaises(ValueError) as error:
        quantized_converter.convert()
      self.assertEqual(
          'representative_dataset is required when specifying '
          'TFLITE_BUILTINS_INT8 or INT8 supported types.', str(error.exception))

    else:
      quantized_tflite = quantized_converter.convert()
      self.assertTrue(quantized_tflite)
      interpreter = Interpreter(model_content=quantized_tflite)
      interpreter.allocate_tensors()
      self.assertEqual(interpreter.get_tensor_details()[0]['name'],
                       'Conv2D_bias')

      if is_float16_quantized:
        # Verify that bias constant is float16 type.
        self.assertEqual(interpreter.get_tensor_details()[0]['dtype'],
                         lite.constants.FLOAT16)
      elif is_post_training_quantized:
        # Verify that bias constants is int32 type.
        self.assertEqual(interpreter.get_tensor_details()[0]['dtype'],
                         lite.constants.INT32)
      else:
        raise ValueError('Invalid test options.')

  def testInvalidQuantizeFloat16(self):
    with ops.Graph().as_default():
      inp, output, _ = self._getCalibrationQuantizeModel()
      sess = session.Session()

    # Specify float16 quantization
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_types = [lite.constants.FLOAT16]
    # Specifiy only int8 builtin ops
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    with self.assertRaises(ValueError) as error:
      quantized_converter.convert()
    self.assertEqual(
        'TFLITE_BUILTINS_INT8 requires smallest supported type to be INT8.',
        str(error.exception))

  def testInvalidPostTrainingQuantize(self):
    np.random.seed(0)
    with ops.Graph().as_default():
      # We need the tensor to have more than 1024 elements for quantize_weights
      # to kick in. Thus, the [33, 33] shape.
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    # Attempt to convert to quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    # Restricting to int8 type only
    quantized_converter.target_spec.supported_types = [lite.constants.INT8]
    # A representative dataset is required for full fixed point quantization.
    with self.assertRaises(ValueError) as error:
      quantized_converter.convert()
    self.assertEqual(
        'representative_dataset is required when specifying '
        'TFLITE_BUILTINS_INT8 or INT8 supported types.', str(error.exception))

  def testPostTrainingCalibrateAndQuantizeFloatNotAllowed(self):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getCalibrationQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_converter.target_spec.supported_types = [lite.constants.INT8]
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)
    # Ensure that restricting supported types to int8 forces
    # all fixed point ops/tensors in converter.
    self.assertTrue(quantized_converter._is_int8_target_required())

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite), len(float_tflite))

  def testPostTrainingCalibrateAndQuantizeInt8Inputs(self):
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getCalibrationQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite = float_converter.convert()
    self.assertTrue(float_tflite)

    # Convert quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.inference_input_type = lite_constants.INT8
    quantized_converter.inference_output_type = lite_constants.INT8
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite = quantized_converter.convert()
    self.assertTrue(quantized_tflite)

    # The input and output types should be int8.
    interpreter = Interpreter(model_content=quantized_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual(np.int8, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual(np.int8, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertTrue(len(quantized_tflite) < len(float_tflite))

  def testFloatTocoConverter(self):
    """Tests deprecated test TocoConverter."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure the interpreter is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testMultipleOutputNodeNames(self):
    """Tests converting a graph with an op that have multiple outputs."""
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.float32)
      out0, out1, out2, out3 = array_ops.split(
          input_tensor, [1, 1, 1, 1], axis=0)
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [input_tensor],
                                                  [out0, out1, out2, out3])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    interpreter.set_tensor(input_details[0]['index'],
                           np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    self.assertEqual(4, len(output_details))
    self.assertEqual(1.0, interpreter.get_tensor(output_details[0]['index']))
    self.assertEqual(2.0, interpreter.get_tensor(output_details[1]['index']))
    self.assertEqual(3.0, interpreter.get_tensor(output_details[2]['index']))
    self.assertEqual(4.0, interpreter.get_tensor(output_details[3]['index']))

  @test_util.run_in_graph_and_eager_modes
  def testFunctions(self):
    """Tests tf.function in 1.X."""

    @def_function.function
    def plus_placeholder(x, placeholder):
      return x + placeholder

    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1], name='input')
      variable_node = variables.Variable(1.0, name='variable_node')
      defun_node = plus_placeholder(variable_node, placeholder)
      output_node = math_ops.multiply(defun_node, 2.0, name='output_node')

      # Initialize variables in the model.
      sess = session.Session()
      sess.run(variables.variables_initializer([variable_node]))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [placeholder],
                                                  [output_node])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output_node', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testInferenceInputOutputTypeFloatDefault(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())

  def testInferenceInputOutputTypeQuantizedUint8Default(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())

  def testReusingConverterWithDifferentPostTrainingQuantization(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])

    converter.post_training_quantize = True
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    converter.post_training_quantize = False
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  def testResizingIntermediateDynamicTensor(self):
    # This is a regression test for the case where shape of dynamic output
    # tensors changes between invocations.
    # See also https://github.com/tensorflow/tensorflow/issues/26549
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(shape=[1, 1], dtype=dtypes.float32)
      input2_tensor = array_ops.placeholder(shape=[1], dtype=dtypes.float32)

      # The bug is triggered only when dynamic tensor is intermediate. Putting
      # some other ops around it.
      neg = math_ops.negative(input2_tensor)
      padding = array_ops.placeholder(shape=[2, 2], dtype=dtypes.int32)
      output_tensor = array_ops.pad(input_tensor, padding) + neg

      sess = session.Session()

    converter = lite.TFLiteConverter.from_session(
        sess, [input_tensor, padding, input2_tensor], [output_tensor])
    tflite_model = converter.convert()

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[1]['index'],
                           np.array([[1, 1], [1, 1]], dtype=np.int32))
    interpreter.invoke()

    # Without the fix, invocation will fail when changing the shape of
    # intermediate dynamic tensors.
    interpreter.set_tensor(input_details[1]['index'],
                           np.array([[2, 2], [2, 2]], dtype=np.int32))
    interpreter.invoke()

  def testGraphDebugInfo(self):
    """Test a session has debug info captured."""

    @def_function.function
    def plus_placeholder(x, placeholder):
      return x + placeholder

    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1], name='input')
      variable_node = variables.Variable(1.0, name='variable_node')
      defun_node = plus_placeholder(variable_node, placeholder)
      output_node = math_ops.multiply(defun_node, 2.0, name='output_node')

      # Initialize variables in the model.
      sess = session.Session()
      sess.run(variables.variables_initializer([variable_node]))

    converter = lite.TFLiteConverter.from_session(sess, [placeholder],
                                                  [output_node])
    converter.convert()
    self.assertValidDebugInfo(converter._debug_info)

    # Check the add node in the inlined function is included.
    func = sess.graph.as_graph_def().library.function[0].signature.name
    self.assertIn((func + 'add'), converter._debug_info.traces)


class FromFrozenGraphFile(test_util.TensorFlowTestCase):

  def testFloat(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testFloatWithShapesArray(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, ['Placeholder'], ['add'],
        input_shapes={'Placeholder': [1, 16, 16, 3]})
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())

  def testFreezeGraph(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      var = variable_scope.get_variable(
          'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + var
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Ensure the graph with variables cannot be converted.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                             ['add'])
    self.assertEqual('Please freeze the graph using freeze_graph.py.',
                     str(error.exception))

  def testPbtxt(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pbtxt')
    write_graph(sess.graph_def, '', graph_def_file, True)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testInvalidFileNotFound(self):
    with self.assertRaises(IOError) as error:
      lite.TFLiteConverter.from_frozen_graph('invalid_file', ['Placeholder'],
                                             ['add'])
    self.assertEqual('File \'invalid_file\' does not exist.',
                     str(error.exception))

  def testInvalidFileBadData(self):
    graph_def_file = os.path.join(self.get_temp_dir(), 'invalid_file')
    with gfile.Open(graph_def_file, 'wb') as temp_file:
      temp_file.write('bad data')
      temp_file.flush()

    # Attempts to convert the invalid model.
    with self.assertRaises(IOError) as error:
      lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                             ['add'])
    self.assertEqual(
        'Unable to parse input file \'{}\'.'.format(graph_def_file),
        str(error.exception))

  def testFloatTocoConverter(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testGraphDebugInfo(self):
    """Test a frozen graph doesn't have debug info captured."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    converter.convert()
    # GraphDebugInfo should be none for frozen graph.
    self.assertTrue(not converter._debug_info)


class FromFrozenGraphObjectDetection(test_util.TensorFlowTestCase):

  def _initObjectDetectionArgs(self):
    # Initializes the arguments required for the object detection model.
    # Looks for the model file which is saved in a different location internally
    # and externally.
    filename = resource_loader.get_path_to_datafile('testdata/tflite_graph.pb')
    if not os.path.exists(filename):
      filename = os.path.join(
          resource_loader.get_root_dir_with_all_resources(),
          '../tflite_mobilenet_ssd_quant_protobuf/tflite_graph.pb')
      if not os.path.exists(filename):
        raise IOError("File '{0}' does not exist.".format(filename))

    self._graph_def_file = filename
    self._input_arrays = ['normalized_input_image_tensor']
    self._output_arrays = [
        'TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1',
        'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3'
    ]
    self._input_shapes = {'normalized_input_image_tensor': [1, 300, 300, 3]}

  def testTFLiteGraphDef(self):
    # Tests the object detection model that cannot be loaded in TensorFlow.
    self._initObjectDetectionArgs()

    converter = lite.TFLiteConverter.from_frozen_graph(self._graph_def_file,
                                                       self._input_arrays,
                                                       self._output_arrays,
                                                       self._input_shapes)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('normalized_input_image_tensor', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 300, 300, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(4, len(output_details))
    self.assertEqual('TFLite_Detection_PostProcess', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 10, 4] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual('TFLite_Detection_PostProcess:1',
                     output_details[1]['name'])
    self.assertTrue(([1, 10] == output_details[1]['shape']).all())
    self.assertEqual('TFLite_Detection_PostProcess:2',
                     output_details[2]['name'])
    self.assertTrue(([1, 10] == output_details[2]['shape']).all())
    self.assertEqual('TFLite_Detection_PostProcess:3',
                     output_details[3]['name'])
    self.assertTrue(([1] == output_details[3]['shape']).all())

  def testTFLiteGraphDefMissingShape(self):
    # Tests invalid cases for the model that cannot be loaded in TensorFlow.
    self._initObjectDetectionArgs()

    # Missing `input_shapes`.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_frozen_graph(self._graph_def_file,
                                             self._input_arrays,
                                             self._output_arrays)
    self.assertEqual('input_shapes must be defined for this model.',
                     str(error.exception))

  def testTFLiteGraphDefInvalidShape(self):
    # Tests invalid cases for the model that cannot be loaded in TensorFlow.
    self._initObjectDetectionArgs()

    # `input_shapes` does not contain the names in `input_arrays`.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_frozen_graph(
          self._graph_def_file,
          self._input_arrays,
          self._output_arrays,
          input_shapes={'invalid-value': [1, 19]})
    self.assertEqual(
        'input_shapes must contain a value for each item in input_array.',
        str(error.exception))


class FromSavedModelTest(TestModels):

  def _createSavedModel(self, shape):
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor_1 = array_ops.placeholder(
            shape=shape, dtype=dtypes.float32, name='inputB')
        in_tensor_2 = array_ops.placeholder(
            shape=shape, dtype=dtypes.float32, name='inputA')
        out_tensor = in_tensor_1 + in_tensor_2
        inputs = {'x': in_tensor_1, 'y': in_tensor_2}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def testSimpleModel(self):
    """Test a SavedModel."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testNoneBatchSize(self):
    """Test a SavedModel, with None in input tensor's shape."""
    saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])

    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testOrderInputArrays(self):
    """Test a SavedModel ordering of input arrays."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputB', 'inputA'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testSubsetInputArrays(self):
    """Test a SavedModel with a subset of the input array names of the model."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Check case where input shape is given.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir,
        input_arrays=['inputA'],
        input_shapes={'inputA': [1, 16, 16, 3]})

    # Since we only partially specify the input, this is not allowed.
    with self.assertRaises(ConverterError):
      _ = converter.convert()

    # Check case where input shape is None.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputA'], input_shapes={'inputA': None})

    # Since we only partially specify the input, this is not allowed.
    with self.assertRaises(ConverterError):
      _ = converter.convert()

  def testSimpleModelTocoConverter(self):
    """Test a SavedModel with deprecated TocoConverter."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testGraphDebugInfo(self):
    """Test a SavedModel has debug info captured."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.convert()
    self.assertValidDebugInfo(converter._debug_info)


class MyAddLayer(keras.layers.Layer):

  def __init__(self, increment, **kwargs):
    super(MyAddLayer, self).__init__(**kwargs)
    self._increment = increment

  def call(self, inputs):
    return inputs + self._increment

  def get_config(self):
    config = super(MyAddLayer, self).get_config()
    config['increment'] = self._increment
    return config


class FromKerasFile(TestModels, parameterized.TestCase):

  def setUp(self):
    super(FromKerasFile, self).setUp()
    self._keras_file = None
    self._custom_objects = None
    if not context.executing_eagerly():
      keras.backend.clear_session()

  def tearDown(self):
    if self._keras_file:
      os.remove(self._keras_file)
    super(FromKerasFile, self).tearDown()

  def _getSequentialModel(self, include_custom_layer=False):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    if include_custom_layer:
      model.add(MyAddLayer(1.0))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    try:
      fd, self._keras_file = tempfile.mkstemp('.h5')
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

    if include_custom_layer:
      self._custom_objects = {'MyAddLayer': MyAddLayer}

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testSequentialModel(self, test_context):
    """Test a Sequential tf.keras model with default inputs."""
    with test_context():
      self._getSequentialModel()

      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      tflite_model = converter.convert()
      self.assertTrue(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('dense_input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 3, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testCustomLayer(self, test_context):
    """Test a Sequential tf.keras model with default inputs."""
    with test_context():
      self._getSequentialModel(include_custom_layer=True)

      converter = lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, custom_objects=self._custom_objects)
      tflite_model = converter.convert()
      self.assertTrue(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(
        self._keras_file, custom_objects=self._custom_objects)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def testSequentialModelInputArray(self):
    """Test a Sequential tf.keras model testing input arrays argument."""
    ops.disable_eager_execution()
    self._getSequentialModel()

    # Invalid input array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_arrays=['invalid-input'])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

    # Valid input array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_arrays=['dense_input'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  def testSequentialModelInputShape(self):
    """Test a Sequential tf.keras model testing input shapes argument."""
    self._getSequentialModel()

    # Passing in shape of invalid input array raises error.
    with self.assertRaises(ValueError) as error:
      converter = lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_shapes={'invalid-input': [2, 3]})
    self.assertEqual(
        "Invalid tensor 'invalid-input' found in tensor shapes map.",
        str(error.exception))

    # Passing in shape of valid input array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_shapes={'dense_input': [2, 3]})
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check input shape from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('dense_input', input_details[0]['name'])
    self.assertTrue(([2, 3] == input_details[0]['shape']).all())

  def testSequentialModelOutputArray(self):
    """Test a Sequential tf.keras model testing output arrays argument."""
    ops.disable_eager_execution()
    self._getSequentialModel()

    # Invalid output array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, output_arrays=['invalid-output'])
    self.assertEqual("Invalid tensors 'invalid-output' were found.",
                     str(error.exception))

    # Valid output array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, output_arrays=['time_distributed/Reshape_1'])
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testFunctionalModel(self, test_context):
    """Test a Functional tf.keras model with default inputs."""
    with test_context():
      inputs = keras.layers.Input(shape=(3,), name='input')
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(
          loss=keras.losses.MSE,
          optimizer='sgd',
          metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      model.predict(x)
      fd, self._keras_file = tempfile.mkstemp('.h5')
      try:
        keras.models.save_model(model, self._keras_file)
      finally:
        os.close(fd)

      # Convert to TFLite model.
      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      tflite_model = converter.convert()
      self.assertTrue(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def testFunctionalModelMultipleInputs(self):
    """Test a Functional tf.keras model with multiple inputs and outputs."""
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')
    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])
    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.mae],
        loss_weights=[1., 0.5])

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))
    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))
    model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])

    model.predict([input_a_np, input_b_np], batch_size=5)
    fd, self._keras_file = tempfile.mkstemp('.h5')
    try:
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('input_a', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEqual('input_b', input_details[1]['name'])
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertTrue(([1, 3] == input_details[1]['shape']).all())
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertTrue(([1, 4] == output_details[1]['shape']).all())
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testFunctionalSequentialModel(self):
    """Test a Functional tf.keras model containing a Sequential model."""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model = keras.models.Model(model.input, model.output)

    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    model.predict(x)
    fd, self._keras_file = tempfile.mkstemp('.h5')
    try:
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('dense_input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 3] == input_details[0]['shape']).all())
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 3, 3] == output_details[0]['shape']).all())
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def testSequentialModelTocoConverter(self):
    """Test a Sequential tf.keras model with deprecated TocoConverter."""
    self._getSequentialModel()

    converter = lite.TocoConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testGraphDebugInfo(self, test_context):
    """Test a Sequential tf.keras model has debug info captured."""
    with test_context():
      self._getSequentialModel()
      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      converter.convert()
      self.assertValidDebugInfo(converter._debug_info)


class GrapplerTest(TestModels):

  def testConstantFolding(self):
    ops.disable_eager_execution()
    # Constant folding handles the tf.broadcast_to operation which was not
    # supported by the TFLite at the time this test was added.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[3, 3], dtype=dtypes.float32)
      y_const = constant_op.constant([1., 2., 3.])
      y_broadcast = gen_array_ops.broadcast_to(y_const, [3, 3])
      out_tensor = math_ops.matmul(in_tensor, y_broadcast, name='output')
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([3, 3] == input_details[0]['shape']).all())

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([3, 3] == output_details[0]['shape']).all())


class ImportOpsUtilTest(test_util.TensorFlowTestCase):

  def testGetPotentiallySupportedOps(self):
    self.assertIsNotNone(lite.get_potentially_supported_ops())


if __name__ == '__main__':
  test.main()
