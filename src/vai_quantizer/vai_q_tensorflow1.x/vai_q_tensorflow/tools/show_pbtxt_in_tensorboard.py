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


import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format

tf.app.flags.DEFINE_string('pbtxt_file', '', 'input pbtxt file')
FLAGS = tf.app.flags.FLAGS

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()

graph_def = graph_pb2.GraphDef()
pb_str = gfile.FastGFile(FLAGS.pbtxt_file, "rb").read()
text_format.Parse(pb_str, graph_def)
_ = tf.import_graph_def(graphdef, name="")

summary_write = tf.summary.FileWriter("./logdir/", graph)

os.system('tensorboard --logdir ./logdir/ --port 6006')
