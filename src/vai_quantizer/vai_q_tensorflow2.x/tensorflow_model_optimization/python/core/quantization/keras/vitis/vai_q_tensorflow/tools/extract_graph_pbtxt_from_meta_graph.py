import tensorflow as tf
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.platform import gfile
from google.protobuf import text_format

tf.app.flags.DEFINE_string('input_meta', '', 'input meta graph')
tf.app.flags.DEFINE_string('output_graph', '', 'output graph')
FLAGS = tf.app.flags.FLAGS

input_meta_graph = FLAGS.input_meta
output_graph = FLAGS.output_graph

input_meta_graph_def = MetaGraphDef()
input_binary = True

mode = "rb" if input_binary else "r"
with gfile.FastGFile(input_meta_graph, mode) as f:
  if input_binary:
    input_meta_graph_def.ParseFromString(f.read())
  else:
    text_format.Merge(f.read(), input_meta_graph_def)

with gfile.GFile(output_graph, "w") as f:
  f.write(text_format.MessageToString(input_meta_graph_def.graph_def))

print("Loaded meta graph file '" + input_meta_graph)
print("Write graph to file '" + output_graph)
