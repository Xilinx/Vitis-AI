import tensorflow as tf
from tensorflow.python.platform import gfile
import os

tf.app.flags.DEFINE_string('pb_file', '', 'input pb file')
FLAGS = tf.app.flags.FLAGS

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile(FLAGS.pb_file, "rb").read())
_ = tf.import_graph_def(graphdef, name="")


node_ops = []
for node in graphdef.node:
  if not node.op in node_ops:
    node_ops.append(node.op)

print(node_ops)
