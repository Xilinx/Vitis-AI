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

tf.app.flags.DEFINE_string('pb_file', '', 'input pb file')
FLAGS = tf.app.flags.FLAGS

if not FLAGS.pb_file:
  print("Usage: python show_quantize_info.py --pb_file deploy_model.pb")
  exit()

graphdef = tf.GraphDef()
graphdef.ParseFromString(tf.gfile.FastGFile(FLAGS.pb_file, "rb").read())
for node in graphdef.node:
  print("Op: {}, Type: {}".format(node.name, node.op))
  for key in node.attr:
    if key in ['ipos', 'opos', 'wpos', 'bpos']:
      print("  {}: bit_width: {} quantize_pos: {}".format(key, node.attr[key].list.i[0], node.attr[key].list.i[1]))
