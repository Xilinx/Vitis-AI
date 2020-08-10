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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


# Number of steps to train model.
TRAIN_STEPS = 1

CONFIG = tf.ConfigProto(device_count={"GPU": 0})


class UnidirectionalSequenceLstmTest(test_util.TensorFlowTestCase):

  def setUp(self):
    tf.reset_default_graph()
    # Import MNIST dataset
    self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Define constants
    # Unrolled through 28 time steps
    self.time_steps = 28
    # Rows of 28 pixels
    self.n_input = 28
    # Learning rate for Adam optimizer
    self.learning_rate = 0.001
    # MNIST is meant to be classified in 10 classes(0-9).
    self.n_classes = 10
    # Batch size
    self.batch_size = 16
    # Lstm Units.
    self.num_units = 16

  def buildLstmLayer(self):
    return tf.keras.layers.StackedRNNCells([
        tf.lite.experimental.nn.TFLiteLSTMCell(
            self.num_units, use_peepholes=True, forget_bias=1.0, name="rnn1"),
        tf.lite.experimental.nn.TFLiteLSTMCell(
            self.num_units, num_proj=8, forget_bias=1.0, name="rnn2"),
        tf.lite.experimental.nn.TFLiteLSTMCell(
            self.num_units // 2,
            use_peepholes=True,
            num_proj=8,
            forget_bias=0,
            name="rnn3"),
        tf.lite.experimental.nn.TFLiteLSTMCell(
            self.num_units, forget_bias=1.0, name="rnn4")
    ])

  def buildModel(self, lstm_layer, is_dynamic_rnn):
    """Build Mnist recognition model.

    Args:
      lstm_layer: The lstm layer either a single lstm cell or a multi lstm cell.
      is_dynamic_rnn: Use dynamic_rnn or not.

    Returns:
     A tuple containing:

     - Input tensor of the model.
     - Prediction tensor of the model.
     - Output class tensor of the model.
    """
    # Weights and biases for output softmax layer.
    out_weights = tf.Variable(
        tf.random_normal([self.num_units, self.n_classes]))
    out_bias = tf.Variable(tf.random_normal([self.n_classes]))

    # input image placeholder
    x = tf.placeholder(
        "float", [None, self.time_steps, self.n_input], name="INPUT_IMAGE")

    # x is shaped [batch_size,time_steps,num_inputs]
    if is_dynamic_rnn:
      lstm_input = tf.transpose(x, perm=[1, 0, 2])
      outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
          lstm_layer, lstm_input, dtype="float32")
      outputs = tf.unstack(outputs, axis=0)
    else:
      lstm_input = tf.unstack(x, self.time_steps, 1)
      outputs, _ = tf.nn.static_rnn(lstm_layer, lstm_input, dtype="float32")

    # Compute logits by multiplying outputs[-1] of shape [batch_size,num_units]
    # by the softmax layer's out_weight of shape [num_units,n_classes]
    # plus out_bias
    prediction = tf.matmul(outputs[-1], out_weights) + out_bias
    output_class = tf.nn.softmax(prediction, name="OUTPUT_CLASS")

    return x, prediction, output_class

  def trainModel(self, x, prediction, output_class, sess):
    """Train the model.

    Args:
      x: The input tensor.
      prediction: The prediction class tensor.
      output_class: The output tensor.
      sess: The graph session.
    """
    # input label placeholder
    y = tf.placeholder("float", [None, self.n_classes])
    # Loss function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # Optimization
    opt = tf.train.AdamOptimizer(
        learning_rate=self.learning_rate).minimize(loss)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(TRAIN_STEPS):
      batch_x, batch_y = self.mnist.train.next_batch(
          batch_size=self.batch_size, shuffle=False)

      batch_x = batch_x.reshape((self.batch_size, self.time_steps,
                                 self.n_input))
      sess.run(opt, feed_dict={x: batch_x, y: batch_y})

  def saveAndRestoreModel(self, lstm_layer, sess, saver, is_dynamic_rnn):
    """Saves and restores the model to mimic the most common use case.

    Args:
      lstm_layer: The lstm layer either a single lstm cell or a multi lstm cell.
      sess: Old session.
      saver: Saver created by tf.compat.v1.train.Saver()
      is_dynamic_rnn: Use dynamic_rnn or not.

    Returns:
      A tuple containing:

      - Input tensor of the restored model.
      - Prediction tensor of the restored model.
      - Output tensor, which is the softwmax result of the prediction tensor.
      - new session of the restored model.

    """
    model_dir = tempfile.mkdtemp()
    saver.save(sess, model_dir)

    # Reset the graph.
    tf.reset_default_graph()
    x, prediction, output_class = self.buildModel(lstm_layer, is_dynamic_rnn)

    new_sess = tf.compat.v1.Session(config=CONFIG)
    saver = tf.train.Saver()
    saver.restore(new_sess, model_dir)
    return x, prediction, output_class, new_sess

  def getInferenceResult(self, x, output_class, sess):
    """Get inference result given input tensor and output tensor.

    Args:
      x: The input tensor.
      output_class: The output tensor.
      sess: Current session.

    Returns:
     A tuple containing:

      - Input of the next batch, batch size is 1.
      - Expected output.

    """
    b1, _ = self.mnist.train.next_batch(batch_size=1)
    sample_input = np.reshape(b1, (1, self.time_steps, self.n_input))

    expected_output = sess.run(output_class, feed_dict={x: sample_input})
    return sample_input, expected_output

  def tfliteInvoke(self, sess, test_inputs, input_tensor, output_tensor):
    """Get tflite inference result.

    This method will convert tensorflow from session to tflite model then based
    on the inputs, run tflite inference and return the results.

    Args:
      sess: Current tensorflow session.
      test_inputs: The test inputs for tflite.
      input_tensor: The input tensor of tensorflow graph.
      output_tensor: The output tensor of tensorflow graph.

    Returns:
      The tflite inference result.
    """
    converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensor],
                                                     [output_tensor])
    tflite = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite)

    try:
      interpreter.allocate_tensors()
    except ValueError:
      assert False

    input_index = (interpreter.get_input_details()[0]["index"])
    interpreter.set_tensor(input_index, test_inputs)
    interpreter.invoke()
    output_index = (interpreter.get_output_details()[0]["index"])
    result = interpreter.get_tensor(output_index)
    # Reset all variables so it will not pollute other inferences.
    interpreter.reset_all_variables()
    return result

  def testStaticRnnMultiRnnCell(self):
    sess = tf.compat.v1.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildLstmLayer(), is_dynamic_rnn=False)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()
    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildLstmLayer(), sess, saver, is_dynamic_rnn=False)

    test_inputs, expected_output = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(new_sess, test_inputs, x, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))

  @test_util.enable_control_flow_v2
  def testDynamicRnnMultiRnnCell(self):
    sess = tf.compat.v1.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildLstmLayer(), is_dynamic_rnn=True)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()

    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildLstmLayer(), sess, saver, is_dynamic_rnn=True)

    test_inputs, expected_output = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(new_sess, test_inputs, x, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))


if __name__ == "__main__":
  test.main()
