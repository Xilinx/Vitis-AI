"""
read frozen pb file and extract const variables then save it to ckpt file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
# from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = None


def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def


def save_const_to_ckpt(pb_file_name, save_path, create_global_step):
  input_graph_def = _parse_input_graph_proto(pb_file_name, True)
  # find read op which contain const weights
  for node in input_graph_def.node:
    # if "read" in node.name:
    if node.op == "Const":
      print("find node: ", node.name)
      # import pdb; pdb.set_trace()
      w_tensor = tensor_util.MakeNdarray(node.attr['value'].tensor)
      tf.Variable(initial_value=w_tensor, name=node.name,
              dtype=node.attr["dtype"].type, shape=w_tensor.shape)
  create_global_step = (create_global_step != 0)
  if create_global_step:
    tf.train.create_global_step()
  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, save_path)

def main(unused_argv):
  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
          "[--tensor_name=tensor_to_print] "
          "[--all_tensors] "
          "[--all_tensor_names] "
          "[--printoptions]")
    sys.exit(1)
  else:
    save_const_to_ckpt(FLAGS.file_name, FLAGS.save_path,
            FLAGS.create_global_step)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--file_name",
      type=str,
      default="",
      help="Frozen pb filename. ")
  parser.add_argument(
      "--save_path",
      type=str,
      default="saved_ckpt/converted.ckpt",
      help="Checkpoint filename. save const data in frozen pb file"
      "Note, if using Checkpoint V2 format, file_name is the "
      "shared prefix between all files in the checkpoint.")
  parser.add_argument(
      "--create_global_step",
      type=str,
      default=1,
      help="Checkpoint filename. save const data in frozen pb file"
      "Note, if using Checkpoint V2 format, file_name is the "
      "shared prefix between all files in the checkpoint.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
