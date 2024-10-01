import argparse
import sys, pickle

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client.session import Session
from tensorflow.python.platform import gfile

import tensorflow.contrib.decent_q

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("input_graph_def", \
                      # "./models/model_a.pb",
                      # "./quantize_results_a/decent_debug/update_old_batchnorms.pb",
                      "./quantize_results_a/decent_debug/fold_batchnorms.pb", \
                      # "./quantize_results_test_0/quantize_eval_model.pb",
                      # "./quantize_results_test_1/quantize_eval_model.pb",
                      "Path to input frozen graph(.pb)")
flags.DEFINE_string(
    "input_fn",
    "input_fn.calib_input",
    "")
flags.DEFINE_string("output_nodes",
    # "decoder/de_conv_block/conv2d_transpose/BiasAdd",
    # "resnet50/conv5_block3_3_conv/BiasAdd",
    "resnet50/conv1_conv/BiasAdd",
    # "conv1_conv/kernel/wquant",
    "the node to be eval")
flags.DEFINE_string( "save_name", "val.pickle", "the node to be eval")

FLAGS = flags.FLAGS

def check_node_names(graph_def, node_names):
  """Check if graph_def has node names"""
  if not isinstance(node_names, list):
    raise TypeError('node_names should be list(str)')

  node_list = []
  for node in graph_def.node:
    node_list.append(node.name)
  for node_name in node_names:
    if not node_name in node_list:
      raise NameError("Node '{}' not found in graph.".format(node_name))

def _parse_output_nodes(input_graph_def, output_nodes_str):
  """Parse output_nodes configurations"""
  output_nodes = []
  if output_nodes_str:
    output_nodes = output_nodes_str.split(",")
    check_node_names(input_graph_def, output_nodes)
  else:
    raise ValueError("No --output_nodes assigned.")
  return output_nodes

def _parse_input_graph(input_graph):
  """Parse input_graph configurations"""
  if input_graph == '':
    raise ValueError("No --input_graph assigned.")
  if not gfile.Exists(input_graph):
    raise ValueError("Input graph file '" + input_graph + "' does not exist!")
  graph_def = graph_pb2.GraphDef()
  with gfile.GFile(input_graph, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def

def _parse_input_fn(input_fn_str):
  """Parse input_fn configurations"""
  input_fn = None
  if input_fn_str == "":
    raise ValueError('No input_fn assigned.')
  else:
    try:
      sys.path.append('./')
      module = __import__(input_fn_str.rsplit('.', 1)[0], fromlist=True)
      input_fn = getattr(module, input_fn_str.rsplit('.', 1)[1])
    except Exception as e:
      raise ValueError('Fail to import input_fn, error: ', e)
  return input_fn

def main():
  graph = tf.Graph()
  with graph.as_default():
    input_graph_def = _parse_input_graph(FLAGS.input_graph_def)
    tf.graph_util.import_graph_def(input_graph_def, name='')
    output_nodes = _parse_output_nodes(input_graph_def, FLAGS.output_nodes)
    output_tensors = [graph.get_tensor_by_name(name + ':0') for name in output_nodes]

    with Session(graph=graph) as sess:
      input_fn = _parse_input_fn(FLAGS.input_fn)
      inputs = input_fn(iter=0)
      # import pdb; pdb.set_trace()
      input_name = "r_image"
      output_val = sess.run(output_tensors, {input_name+":0":inputs[input_name]})
      # import pdb; pdb.set_trace()
      print(output_val[0].shape)
      with open(FLAGS.save_name, 'wb') as handle:
        pickle.dump(output_val[0], handle)

if __name__ == '__main__':
  main()
