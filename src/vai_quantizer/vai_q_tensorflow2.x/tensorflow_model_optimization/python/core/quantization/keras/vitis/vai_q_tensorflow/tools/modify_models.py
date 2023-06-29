import os
import copy
# import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
# from tensorflow.core.framework.tensor_pb2 import TensorProto
# from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
import vai_q_tensorflow
from align_pos import align_pos
from rename_node_func import rename_node


"""
remove nodes before and include `target_node` then set input of graph as a
placeholder node
"""

tf.app.flags.DEFINE_string('input_pb', '', 'input pb file')
tf.app.flags.DEFINE_string('dst_dir', 'modified_models', 'save dir')
tf.app.flags.DEFINE_string('src_name', 'results', 'source node name')
tf.app.flags.DEFINE_string('target_name', 'results', 'target node name')

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

src_graph = FLAGS.input_pb
dst_graph = os.path.join(FLAGS.dst_dir, FLAGS.input_pb)
src_name = FLAGS.src_name
target_name = FLAGS.target_name

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


def main():
  src_graph_def = load_pb(src_graph)
  modified_graph_def = tf.GraphDef()

  name_to_align_pos = {}
  for node in src_graph_def.node:
    new_node = copy.deepcopy(node)
    ### reset input shape
    if node.name == src_name:
      print("reset node shape_attr {} ".format(src_name))
      new_node.attr["shape"].shape.Clear()

      ## TODO: need to reset according to current situation
      val = [1, 224, 224, 3]

      for v in val:
        new_node.attr["shape"].shape.dim.add(size=v)
    ### remove isolated node
    ## TODO: need to reset according to current situation
    elif node.name in ["batch/n"]:
      print("remove isolated node {} ".format(node.name))
      continue
    ### reset shape of reshape
    ## TODO: need to reset according to current situation
    elif node.name in ["InceptionV1/Logits/Predictions/Reshape/shape",  "InceptionV1/Logits/Predictions/Shape"]:
      print("reset shape of reshape node {} ".format(node.name))
      key = "value"

      ## TODO: need to reset according to current situation
      val = np.array([1, 1001], np.int32)

      new_node.attr[key].tensor.tensor_shape.CopyFrom(tf.TensorShape(val.shape).as_proto())
      new_node.attr[key].tensor.tensor_content = bytes(val)

    ### remove _output_shapes
    if "_output_shapes" in new_node.attr:
      new_node.attr.pop("_output_shapes")
    modified_graph_def.node.extend([new_node])


  ### align pos
  src_graph_def = modified_graph_def
  modified_graph_def = align_pos(src_graph_def, align_concat=True, align_maxpool=True, align_avgpool=True)

  ### rename node
  src_graph_def = modified_graph_def
  modified_graph_def = rename_node(src_graph_def,src_name, target_name)

  ### add shape info
  graph = tf.Graph()
  with graph.as_default():
    tf.graph_util.import_graph_def(modified_graph_def, name='')
    dst_graph_def = graph.as_graph_def(add_shapes=True)
  save_pb(dst_graph, dst_graph_def)


if __name__ == '__main__':
  main()
