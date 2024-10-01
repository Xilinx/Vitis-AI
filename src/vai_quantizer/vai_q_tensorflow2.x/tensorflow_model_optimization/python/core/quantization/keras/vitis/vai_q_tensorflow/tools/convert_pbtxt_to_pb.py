import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import os
import argparse
import pdb

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser()
parser.add_argument("--input_graph", type=str, default="", help="tensorflow pb file to load")
parser.add_argument("--output_graph", type=str, default="", help="tensorflow pbtxt file to save")
parser.add_argument("--remove_tensor", type=bool, default=False, help="remove tensor content")
FLAGS, uparsed = parser.parse_known_args()

if not gfile.Exists(FLAGS.input_graph):
    print("Input graph file '"+ FLAGS.input_graph + "' does not exist!")
    print("Usage: python convert_pb_to_pbtxt --input_graph xxx --output_graph xxx [--remove_tensor]")
    exit()

with tf.Session() as sess:
    f=tf.gfile.FastGFile(FLAGS.input_graph,'r')
    in_graph_def = tf.GraphDef()
    pb_str = gfile.FastGFile(FLAGS.input_graph, "rb").read()
    text_format.Parse(pb_str, in_graph_def)

    if FLAGS.remove_tensor:
      for node in in_graph_def.node:
          if node.op == "Const":
              node.attr["value"].tensor.ClearField("tensor_content")

    with tf.gfile.FastGFile(FLAGS.output_graph, 'w') as f:
      f.write(in_graph_def.SerializeToString())
