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
