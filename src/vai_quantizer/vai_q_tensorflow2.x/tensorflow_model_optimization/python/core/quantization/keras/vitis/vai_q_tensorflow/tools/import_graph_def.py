import tensorflow as tf
from tensorflow.python.platform import gfile

input_graph = "./quantize_results/deploy_model.pb"
graph_def = tf.GraphDef()
with gfile.GFile(input_graph, "rb") as f:
  graph_def.ParseFromString(f.read())

for node in graph_def.node:
  print(node)
  import pdb; pdb.set_trace()

