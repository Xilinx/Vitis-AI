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
from tensorflow.keras import backend as K
from tensorflow.python.framework import graph_util
from net import build_model

tf.app.flags.DEFINE_string('output_nodes', '', 'Output nodes of the given graph.')
tf.app.flags.DEFINE_string('graph_filename', '', 'Filename of the graph are saved to.')
tf.app.flags.DEFINE_string('save_dir', './', 'Directory where the graph file are saved to.')
FLAGS = tf.app.flags.FLAGS

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.keras.backend.set_learning_phase(0)

  model = build_model()
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy())

  graph_def = K.get_session().graph.as_graph_def()
  graph_def = graph_util.extract_sub_graph(graph_def, [FLAGS.output_nodes])
  tf.train.write_graph(graph_def,
                       FLAGS.save_dir,
                       FLAGS.graph_filename,
                       as_text=True)
  print("Finish export inference graph: {}".format(FLAGS.save_dir))

if __name__ == '__main__':
  tf.app.run()
