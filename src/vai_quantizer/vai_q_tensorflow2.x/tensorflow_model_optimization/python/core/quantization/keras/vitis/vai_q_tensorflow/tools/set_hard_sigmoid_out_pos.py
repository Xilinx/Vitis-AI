import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# from tensorflow.core.framework.tensor_pb2 import TensorProto
# from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
import tensorflow.contrib.decent_q


"""
remove nodes before and include `target_node` then set input of graph as a
placeholder node
"""

tf.app.flags.DEFINE_string('input_pb', '', 'input pb file')
tf.app.flags.DEFINE_string('dst_dir', 'results', 'save dir')

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

src_graph = FLAGS.input_pb
dst_graph = os.path.join(FLAGS.dst_dir, FLAGS.input_pb)

def save_pb(dst_graph, graph_def):
  dst_dir = os.path.dirname(dst_graph)
  try:
    os.makedirs(dst_dir)
  except OSError as error:
    pass
    # print(error)
  with gfile.GFile(dst_graph, mode='wb') as f:
    f.write(graph_def.SerializeToString())
  print("saing processed grapb pb to ", dst_graph)

def load_pb(src_graph):
  with gfile.FastGFile(src_graph,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

############
def get_shape(src_graph_def, output_nodes):
  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(src_graph_def, name='')
    print("INFO: Checking Float Graph...")
    input_tensors = [
        op.outputs[0] for op in graph.get_operations()
        if op.type == 'Placeholder'
    ]
    output_tensors = [ graph.get_tensor_by_name(output_nodes + ':0') ]

    with tf.Session(graph=graph) as sess:
      feed_dict = {input_tensors[0]: np.zeros([1, 224,224,3])}
      shape = sess.run(output_tensors, feed_dict)[0].shape
  return shape
############

def main():
  src_graph_def = load_pb(src_graph)
  name_to_node = {}
  for node in src_graph_def.node:
    name_to_node[node.name] = node


  ## find (Relu6, Add) pair
  for node in src_graph_def.node:
    if node.op == "FixNeuron":
      inp_node = name_to_node[node.input[0]]
      if inp_node.op == "Mul":
        mul_node = inp_node
        sigmoid = name_to_node[mul_node.input[0]]
        if sigmoid.op == "sigmoid":
          print(sigmoid.name, sigmoid.op)
          import pdb; pdb.set_trace()
          scale = name_to_node[mul_node.input[1]]

          val = np.ones([1], dtype=np.float32) * (6.0 * 2731.0 / 2**14)
          sigmoid.attr["value"].tensor.tensor_shape.CopyFrom(tf.TensorShape(np.shape(val)).as_proto())
          # scale.attr["value"].tensor.tensor_content = bytes(val)
          scale.attr["value"].tensor.tensor_content = val.tobytes()

  save_pb(dst_graph, src_graph_def)


if __name__ == '__main__':
  main()
