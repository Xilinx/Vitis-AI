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
"""TensorFlow Lite Python Interface: Sanity check."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import io
import sys
import numpy as np
import six

from tensorflow.lite.python import interpreter as interpreter_wrapper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class InterpreterTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())
    self.assertEqual((0.0, 0), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())
    self.assertEqual((0.0, 0), output_details[0]['quantization'])

    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    expected_output = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testUint8(self):
    model_path = resource_loader.get_path_to_datafile(
        'testdata/permute_uint8.tflite')
    with io.open(model_path, 'rb') as model_file:
      data = model_file.read()

    interpreter = interpreter_wrapper.Interpreter(model_content=data)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())
    self.assertEqual((1.0, 0), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())
    self.assertEqual((1.0, 0), output_details[0]['quantization'])

    test_input = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    expected_output = np.array([[4, 3, 2, 1]], dtype=np.uint8)
    interpreter.resize_tensor_input(input_details[0]['index'],
                                    test_input.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testString(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/gather_string.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.string_, input_details[0]['dtype'])
    self.assertTrue(([10] == input_details[0]['shape']).all())
    self.assertEqual((0.0, 0), input_details[0]['quantization'])
    self.assertEqual('indices', input_details[1]['name'])
    self.assertEqual(np.int64, input_details[1]['dtype'])
    self.assertTrue(([3] == input_details[1]['shape']).all())
    self.assertEqual((0.0, 0), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.string_, output_details[0]['dtype'])
    self.assertTrue(([3] == output_details[0]['shape']).all())
    self.assertEqual((0.0, 0), output_details[0]['quantization'])

    test_input = np.array([1, 2, 3], dtype=np.int64)
    interpreter.set_tensor(input_details[1]['index'], test_input)

    test_input = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    expected_output = np.array([b'b', b'c', b'd'])
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


class InterpreterTestErrorPropagation(test_util.TensorFlowTestCase):

  def testInvalidModelContent(self):
    with self.assertRaisesRegexp(ValueError,
                                 'Model provided has model identifier \''):
      interpreter_wrapper.Interpreter(model_content=six.b('garbage'))

  def testInvalidModelFile(self):
    with self.assertRaisesRegexp(
        ValueError, 'Could not open \'totally_invalid_file_name\''):
      interpreter_wrapper.Interpreter(
          model_path='totally_invalid_file_name')

  def testInvokeBeforeReady(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    with self.assertRaisesRegexp(RuntimeError,
                                 'Invoke called on model that is not ready'):
      interpreter.invoke()

  def testInvalidModelFileContent(self):
    with self.assertRaisesRegexp(
        ValueError, '`model_path` or `model_content` must be specified.'):
      interpreter_wrapper.Interpreter(model_path=None, model_content=None)

  def testInvalidIndex(self):
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    interpreter.allocate_tensors()
    # Invalid tensor index passed.
    with self.assertRaisesRegexp(ValueError, 'Tensor with no shape found.'):
      interpreter._get_tensor_details(4)


class InterpreterTensorAccessorTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    self.interpreter.allocate_tensors()
    self.input0 = self.interpreter.get_input_details()[0]['index']
    self.initial_data = np.array([[-1., -2., -3., -4.]], np.float32)

  def testTensorAccessor(self):
    """Check that tensor returns a reference."""
    array_ref = self.interpreter.tensor(self.input0)
    np.copyto(array_ref(), self.initial_data)
    self.assertAllEqual(array_ref(), self.initial_data)
    self.assertAllEqual(
        self.interpreter.get_tensor(self.input0), self.initial_data)

  def testGetTensorAccessor(self):
    """Check that get_tensor returns a copy."""
    self.interpreter.set_tensor(self.input0, self.initial_data)
    array_initial_copy = self.interpreter.get_tensor(self.input0)
    new_value = np.add(1., array_initial_copy)
    self.interpreter.set_tensor(self.input0, new_value)
    self.assertAllEqual(array_initial_copy, self.initial_data)
    self.assertAllEqual(self.interpreter.get_tensor(self.input0), new_value)

  def testBase(self):
    self.assertTrue(self.interpreter._safe_to_run())
    _ = self.interpreter.tensor(self.input0)
    self.assertTrue(self.interpreter._safe_to_run())
    in0 = self.interpreter.tensor(self.input0)()
    self.assertFalse(self.interpreter._safe_to_run())
    in0b = self.interpreter.tensor(self.input0)()
    self.assertFalse(self.interpreter._safe_to_run())
    # Now get rid of the buffers so that we can evaluate.
    del in0
    del in0b
    self.assertTrue(self.interpreter._safe_to_run())

  def testBaseProtectsFunctions(self):
    in0 = self.interpreter.tensor(self.input0)()
    # Make sure we get an exception if we try to run an unsafe operation
    with self.assertRaisesRegexp(
        RuntimeError, 'There is at least 1 reference'):
      _ = self.interpreter.allocate_tensors()
    # Make sure we get an exception if we try to run an unsafe operation
    with self.assertRaisesRegexp(
        RuntimeError, 'There is at least 1 reference'):
      _ = self.interpreter.invoke()
    # Now test that we can run
    del in0  # this is our only buffer reference, so now it is safe to change
    in0safe = self.interpreter.tensor(self.input0)
    _ = self.interpreter.allocate_tensors()
    del in0safe  # make sure in0Safe is held but lint doesn't complain


class InterpreterDelegateTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._delegate_file = resource_loader.get_path_to_datafile(
        'testdata/test_delegate.so')
    self._model_file = resource_loader.get_path_to_datafile(
        'testdata/permute_float.tflite')

    # Load the library to reset the counters.
    library = ctypes.pydll.LoadLibrary(self._delegate_file)
    library.initialize_counters()

  def _TestInterpreter(self, model_path, options=None):
    """Test wrapper function that creates an interpreter with the delegate."""
    # TODO(b/137299813): Enable when we fix for mac
    if sys.platform == 'darwin': return
    delegate = interpreter_wrapper.load_delegate(self._delegate_file, options)
    return interpreter_wrapper.Interpreter(
        model_path=model_path, experimental_delegates=[delegate])

  def testDelegate(self):
    """Tests the delegate creation and destruction."""
    # TODO(b/137299813): Enable when we fix for mac
    if sys.platform == 'darwin': return

    interpreter = self._TestInterpreter(model_path=self._model_file)
    lib = interpreter._delegates[0]._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

    del interpreter

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 1)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

  def testMultipleInterpreters(self):
    # TODO(b/137299813): Enable when we fix for mac
    if sys.platform == 'darwin': return

    delegate = interpreter_wrapper.load_delegate(self._delegate_file)
    lib = delegate._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)

    interpreter_a = interpreter_wrapper.Interpreter(
        model_path=self._model_file, experimental_delegates=[delegate])

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

    interpreter_b = interpreter_wrapper.Interpreter(
        model_path=self._model_file, experimental_delegates=[delegate])

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

    del delegate
    del interpreter_a

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

    del interpreter_b

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 1)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

  def testDestructionOrder(self):
    """Make sure internal _interpreter object is destroyed before delegate."""
    # Track which order destructions were doned in
    # TODO(b/137299813): Enable when we fix for mac
    if sys.platform == 'darwin': return
    destructions = []
    def register_destruction(x):
      destructions.append(x)
      return 0
    # Make a wrapper for the callback so we can send this to ctypes
    delegate = interpreter_wrapper.load_delegate(self._delegate_file)
    prototype = ctypes.CFUNCTYPE(ctypes.c_int, (ctypes.c_char_p))
    destroy_callback = prototype(register_destruction)
    delegate._library.set_destroy_callback(destroy_callback)
    # Make an interpreter with the delegate
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'), experimental_delegates=[delegate])

    class InterpreterDestroyCallback(object):

      def __del__(self):
        register_destruction('interpreter')

    interpreter._interpreter.stuff = InterpreterDestroyCallback()
    # Destroy both delegate and interpreter
    del delegate
    del interpreter
    # check the interpreter was destroyed before the delegate
    self.assertEqual(destructions, ['interpreter', 'test_delegate'])

  def testOptions(self):
    # TODO(b/137299813): Enable when we fix for mac
    if sys.platform == 'darwin': return
    delegate_a = interpreter_wrapper.load_delegate(self._delegate_file)
    lib = delegate_a._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 0)

    delegate_b = interpreter_wrapper.load_delegate(
        self._delegate_file, options={
            'unused': False,
            'options_counter': 2
        })
    lib = delegate_b._library

    self.assertEqual(lib.get_num_delegates_created(), 2)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 2)

    del delegate_a
    del delegate_b

    self.assertEqual(lib.get_num_delegates_created(), 2)
    self.assertEqual(lib.get_num_delegates_destroyed(), 2)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 2)

  def testFail(self):
    # TODO(b/137299813): Enable when we fix for mac
    if sys.platform == 'darwin': return
    with self.assertRaisesRegexp(
        ValueError, 'Failed to load delegate from .*\nFail argument sent.'):
      interpreter_wrapper.load_delegate(
          self._delegate_file, options={'fail': 'fail'})


if __name__ == '__main__':
  test.main()
