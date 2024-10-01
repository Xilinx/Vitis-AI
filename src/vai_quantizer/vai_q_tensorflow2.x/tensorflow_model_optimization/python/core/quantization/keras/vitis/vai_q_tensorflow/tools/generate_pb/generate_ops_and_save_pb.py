import os
import tensorflow as tf


def save_pb(dst_graph, graph_def):
  dst_dir = os.path.dirname(dst_graph)
  try:
    os.makedirs(dst_dir)
  except OSError as error:
    pass
    # print(error)
  with tf.gfile.GFile(dst_graph, mode='wb') as f:
    f.write(graph_def.SerializeToString())
  print("saing processed grapb pb to ", dst_graph)


g = tf.Graph()
with g.as_default():
  x = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
  x = tf.identity(x)
  x = tf.nn.relu(x)
  # w = tf.Variable(tf.random_normal([5, 5, 3, 32]))
  # x = tf.nn.conv2d(x, w, strides=[1, 3, 3, 1], padding='SAME')
  x = tf.math.sigmoid(x)
  x = tf.identity(x)
  x = tf.nn.relu(x)
  save_pb("graph.pb", g.as_graph_def())

