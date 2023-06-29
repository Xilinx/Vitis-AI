# Copyright 2019 Xilinx Inc.
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
from tensorflow.python.platform import gfile
import os

tf.app.flags.DEFINE_string('saved_model_dir', '', 'saved_model_dir')
tf.app.flags.DEFINE_string('output_node_names', '', 'output_node_names')
FLAGS = tf.app.flags.FLAGS


with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], FLAGS.saved_model_dir)
  summary_write = tf.summary.FileWriter("./logdir/", sess.graph)

  # We use a built-in TF helper to export variables to constants
  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, # The session is used to retrieve the weights
      tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
      FLAGS.output_node_names.split(",") # The output node names are used to select the usefull nodes
  ) 

  # Finally we serialize and dump the output graph to the filesystem
  output_graph="./frozen_model.pb"
  with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))

os.system('tensorboard --logdir ./logdir/ --port 6006')
