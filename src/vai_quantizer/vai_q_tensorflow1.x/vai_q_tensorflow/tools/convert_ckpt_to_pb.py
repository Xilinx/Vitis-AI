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

ckpt_dir_meta="./ckptdir/ResNet-L50.meta"
ckpt_dir="./ckptdir"
output_name="./resnet_50_model.pb"

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver=tf.train.import_meta_graph(ckpt_dir_meta)
    ckpt=tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess,ckpt.model_checkpoint_path)

    graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,['prob'])
    # tf.train.write_graph(graph,'./',output_name,as_text=False)
    tf.train.write_graph(graph,'./',output_name,as_text=True)
