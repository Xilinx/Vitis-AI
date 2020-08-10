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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import numpy as np
import tensorflow as tf

from tensorflow import flags

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.lite.experimental.examples.lstm.rnn import bidirectional_dynamic_rnn
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

FLAGS = flags.FLAGS

# Number of steps to train model.
TRAIN_STEPS = 1

CONFIG = tf.ConfigProto(device_count={"GPU": 0})


class BidirectionalSequenceRnnTest(test_util.TensorFlowTestCase):

  def __init__(self, *args, **kwargs):
    super(BidirectionalSequenceRnnTest, self).__init__(*args, **kwargs)
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
    # Rnn Units.
    self.num_units = 16

  def setUp(self):
    super(BidirectionalSequenceRnnTest, self).setUp()
    # Import MNIST dataset
    data_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.mnist = input_data.read_data_sets(data_dir, one_hot=True)

  def buildRnnLayer(self):
    return tf.keras.layers.StackedRNNCells([
        tf.lite.experimental.nn.TfLiteRNNCell(self.num_units, name="rnn1"),
        tf.lite.experimental.nn.TfLiteRNNCell(self.num_units, name="rnn2")
    ])

  def buildModel(self,
                 fw_rnn_layer,
                 bw_rnn_layer,
                 is_dynamic_rnn,
                 is_inference,
                 use_sequence_length=False):
    """Build Mnist recognition model.

    Args:
      fw_rnn_layer: The forward rnn layer either a single rnn cell or a multi
        rnn cell.
      bw_rnn_layer: The backward rnn layer either a single rnn cell or a multi
        rnn cell.
      is_dynamic_rnn: Use dynamic_rnn or not.
      use_sequence_length: Whether to use sequence length or not. Default to
        False.

    Returns:
     A tuple containing:

     - Input tensor of the model.
     - Prediction tensor of the model.
     - Output class tensor of the model.
    """
    # Weights and biases for output softmax layer.
    out_weights = tf.Variable(
        tf.random_normal([self.num_units * 2, self.n_classes]))
    out_bias = tf.Variable(tf.random_normal([self.n_classes]))

    batch_size = self.batch_size
    if is_inference:
      batch_size = 1
    # input image placeholder
    x = tf.placeholder(
        "float", [batch_size, self.time_steps, self.n_input],
        name="INPUT_IMAGE")

    sequence_length = None
    if use_sequence_length:
      sequence_length = [self.time_steps] * batch_size
    if is_dynamic_rnn:
      rnn_inputs = tf.transpose(x, [1, 0, 2])
      outputs, _ = bidirectional_dynamic_rnn(
          fw_rnn_layer,
          bw_rnn_layer,
          rnn_inputs,
          sequence_length,
          dtype="float32",
          time_major=True)
      fw_outputs, bw_outputs = outputs
      output = tf.concat([fw_outputs, bw_outputs], 2)
      output = tf.unstack(output, axis=0)
      output = output[-1]
    else:
      rnn_inputs = tf.unstack(x, self.time_steps, 1)
      # Sequence length is not supported for static since we don't have a
      # wrapper for it. At training phase, we can still have sequence_length,
      # but inference phase, we change it to None.
      if is_inference:
        sequence_length = None
      outputs, _, _ = tf.nn.static_bidirectional_rnn(
          fw_rnn_layer,
          bw_rnn_layer,
          rnn_inputs,
          dtype="float32",
          sequence_length=sequence_length)
      output = outputs[-1]

    # Compute logits by multiplying output of shape [batch_size,num_units*2]
    # by the softmax layer's out_weight of shape [num_units*2,n_classes]
    # plus out_bias
    prediction = tf.matmul(output, out_weights) + out_bias
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

  def saveAndRestoreModel(self,
                          fw_rnn_layer,
                          bw_rnn_layer,
                          sess,
                          saver,
                          is_dynamic_rnn,
                          use_sequence_length=False):
    """Saves and restores the model to mimic the most common use case.

    Args:
      fw_rnn_layer: The forward rnn layer either a single rnn cell or a multi
        rnn cell.
      bw_rnn_layer: The backward rnn layer either a single rnn cell or a multi
        rnn cell.
      sess: Old session.
      saver: Saver created by tf.compat.v1.train.Saver()
      is_dynamic_rnn: Use dynamic_rnn or not.
      use_sequence_length: Whether to use sequence length or not. Default to
        False.

    Returns:
      A tuple containing:

      - Input tensor of the restored model.
      - Prediction tensor of the restored model.
      - Output tensor, which is the softwmax result of the prediction tensor.
      - new session of the restored model.

    """
    model_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    saver.save(sess, model_dir)

    # Reset the graph.
    tf.reset_default_graph()
    x, prediction, output_class = self.buildModel(
        fw_rnn_layer, bw_rnn_layer, is_dynamic_rnn, True, use_sequence_length)

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

    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, test_inputs)
    interpreter.invoke()
    output_index = interpreter.get_output_details()[0]["index"]
    result = interpreter.get_tensor(output_index)
    # Reset all variables so it will not pollute other inferences.
    interpreter.reset_all_variables()
    return result

  def testStaticRnnMultiRnnCell(self):
    sess = tf.compat.v1.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildRnnLayer(), self.buildRnnLayer(), False, is_inference=False)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()
    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildRnnLayer(), self.buildRnnLayer(), sess, saver, False)

    test_inputs, expected_output = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(new_sess, test_inputs, x, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))

  def testStaticRnnMultiRnnCellWithSequenceLength(self):
    sess = tf.compat.v1.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildRnnLayer(),
        self.buildRnnLayer(),
        False,
        is_inference=False,
        use_sequence_length=True)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()
    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildRnnLayer(),
        self.buildRnnLayer(),
        sess,
        saver,
        False,
        use_sequence_length=True)

    test_inputs, expected_output = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(new_sess, test_inputs, x, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))

  @test_util.enable_control_flow_v2
  def testDynamicRnnMultiRnnCell(self):
    sess = tf.compat.v1.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildRnnLayer(), self.buildRnnLayer(), True, is_inference=False)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()
    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildRnnLayer(),
        self.buildRnnLayer(),
        sess,
        saver,
        is_dynamic_rnn=True)

    test_inputs, expected_output = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(new_sess, test_inputs, x, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))

  @test_util.enable_control_flow_v2
  def testDynamicRnnMultiRnnCellWithSequenceLength(self):
    sess = tf.compat.v1.Session(config=CONFIG)

    x, prediction, output_class = self.buildModel(
        self.buildRnnLayer(),
        self.buildRnnLayer(),
        True,
        is_inference=False,
        use_sequence_length=True)
    self.trainModel(x, prediction, output_class, sess)

    saver = tf.train.Saver()
    x, prediction, output_class, new_sess = self.saveAndRestoreModel(
        self.buildRnnLayer(),
        self.buildRnnLayer(),
        sess,
        saver,
        is_dynamic_rnn=True,
        use_sequence_length=True)

    test_inputs, expected_output = self.getInferenceResult(
        x, output_class, new_sess)

    result = self.tfliteInvoke(new_sess, test_inputs, x, output_class)
    self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))


if __name__ == "__main__":
  test.main()
