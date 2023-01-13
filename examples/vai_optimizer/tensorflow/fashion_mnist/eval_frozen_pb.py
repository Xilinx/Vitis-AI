# Copyright 2021 Xilinx Inc.
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

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from net import x_train, y_train, x_test, y_test

tf.app.flags.DEFINE_string('input_graph', '',
                           'TensorFlow \'GraphDef\' file to load.')

tf.app.flags.DEFINE_string('logits_name', None,
                           'The name of logits before softmax node.')
FLAGS = tf.app.flags.FLAGS

def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def

def main(_):

  from net import x_train, y_train, x_test, y_test
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:

    input_binary = False if 'txt' in FLAGS.input_graph else True
    input_graph_def = _parse_input_graph_proto(FLAGS.input_graph, input_binary)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)
    _ = importer.import_graph_def(input_graph_def,
                                  name="",
                                  input_map={"input_1:0": x_test})

    logits = graph.get_tensor_by_name(FLAGS.logits_name)
    top1, top1_update = tf.metrics.recall_at_k(y_test, logits, 1, name='precision')
    with tf.Session() as sess:
      var_list = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                   scope="precision")
      vars_initializer = tf.variables_initializer(var_list=var_list)
      sess.run(vars_initializer)
      sess.run(logits)
      sess.run(top1_update)
      top1 = sess.run(top1)
      print('top1: {}'.format(top1))

if __name__ == '__main__':
  tf.app.run()
