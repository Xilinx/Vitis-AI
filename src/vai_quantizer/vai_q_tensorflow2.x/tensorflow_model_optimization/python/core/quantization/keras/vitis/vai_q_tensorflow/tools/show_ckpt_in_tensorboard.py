import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
# import tensorflow.contrib.decent_q

tf.app.flags.DEFINE_string('model_dir', '', 'model folder')
tf.app.flags.DEFINE_integer('port', '6006', 'port number')
tf.app.flags.DEFINE_string('logdir', './logdir/', 'log dir')
FLAGS = tf.app.flags.FLAGS

model_dir = FLAGS.model_dir

if not tf.gfile.Exists(model_dir):
    raise AssertionError(
        "Export directory doesn't exists. Please specify an export "
        "directory: %s" % model_dir)

# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True

# We start a session using a temporary fresh Graph
with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    for node in sess.graph_def.node:
      print(node.name, node.op)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

writer.flush()
os.system('tensorboard --logdir {} --port {}'.format(FLAGS.logdir, FLAGS.port))
