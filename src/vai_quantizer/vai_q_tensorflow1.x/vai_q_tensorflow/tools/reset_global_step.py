"""
a_new = 10
b_new = 30
work_dir = './merge_res_bn'
ckpt_name = 'model.ckpt'
a = tf.get_variable(name='a', initializer=tf.constant(1))
assign_a = a.assign(a_m)
b = tf.get_variable(name='b', initializer=tf.constant(3))
assign_b = b.assign(b_m)
c = a+b

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(work_dir, sess.graph)
  print(sess.run(a))
  print(sess.run(b))
  print(sess.run(c))
  sess.run(assign_a)
  sess.run(assign_b)
  print(sess.run(a))
  print(sess.run(b))
  print(sess.run(c))
  var_list=[a,b]
  saver = tf.train.Saver(var_list=var_list)
  saver.save(sess, os.path.join(work_dir, ckpt_name))

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys, os

import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

gpus_list = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus_list


FLAGS = None

def get_original_tensor_list(file_name):
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    original_tensor_dict = {}
    for op_name in sorted(var_to_shape_map):
      tf.logging.info("tensor_name: {}".format(op_name))
      original_tensor_dict[op_name] = reader.get_tensor(op_name)
    return original_tensor_dict
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    sys.exit(1)

def infer_bn_names(conv_name):
  """
  examples:

  conv_name = ssd1200/stage3_residul_block5/stage3_residul_block5_1/kernel
  ->ssd1200/stage3_residul_block5/stage3_residul_block5_bn1/beta
    ssd1200/stage3_residul_block5/stage3_residul_block5_bn1/gamma
    ssd1200/stage3_residul_block5/stage3_residul_block5_bn1/moving_mean
    ssd1200/stage3_residul_block5/stage3_residul_block5_bn1/moving_variance
  """
  if '_1' in conv_name and 'kernel' in conv_name and 'stage' in conv_name:
    org_suffix = '_1'
    new_suffix = '_bn1'
  elif '_2' in conv_name and 'kernel' in conv_name and 'stage' in conv_name:
    org_suffix = '_2'
    new_suffix = '_bn2'
  else:
    raise ValueError("Can not find suffix in conv op name")

  beta_name = conv_name.replace(org_suffix, new_suffix).replace('kernel', 'beta')
  gamma_name = conv_name.replace(org_suffix, new_suffix).replace('kernel', 'gamma')
  mean_name = conv_name.replace(org_suffix, new_suffix).replace('kernel', 'moving_mean')
  variance_name = conv_name.replace(org_suffix, new_suffix).replace('kernel', 'moving_variance')
  return beta_name, gamma_name, mean_name, variance_name

def merge_conv_bn(conv_name, original_tensor_dict):
  # conv kernel shape HWCN
  conv_kernel = original_tensor_dict[conv_name]
  beta_name, gamma_name, mean_name, variance_name = infer_bn_names(conv_name)
  beta = original_tensor_dict[beta_name]
  gamma = original_tensor_dict[gamma_name]
  mean = original_tensor_dict[mean_name]
  variance = original_tensor_dict[variance_name]
  # print(type(conv_kernel))
  # print(type(beta))
  tf.logging.debug(conv_kernel)
  tf.logging.debug(beta)
  tf.logging.debug(gamma)
  tf.logging.debug(mean)
  tf.logging.debug(variance)

  # TODO: compute merged W and b
  return

def main(unused_argv):
  # print(FLAGS.file_name)
  merge_conv_op_list = ["global_step"]
  # skip_op_list = []
  new_var_list = []
  model_dir = './finetune_dir_bak'
  ckpt_name = 'model.ckpt-0'
  original_tensor_dict = get_original_tensor_list(FLAGS.file_name)
  with tf.Session() as sess:
    for op_name, tensor in original_tensor_dict.items():
      tf.logging.info("{} {}".format(op_name, tensor.shape))
      if op_name in merge_conv_op_list:
        # new_tensor, bn_op_list = merge_conv_bn(op_name, original_tensor_dict)
        new_tensor = tf.Variable(0, name=op_name, dtype=tf.int64)
      else:
        new_tensor = tf.Variable(tensor, name=op_name)
      new_var_list.append(new_tensor)
    saver = tf.train.Saver(var_list=new_var_list)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(model_dir, ckpt_name))


if __name__ == "__main__":
  # tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.set_verbosity(tf.logging.DEBUG)
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--file_name",
      type=str,
      default="",
      help="Checkpoint filename. "
      "Note, if using Checkpoint V2 format, file_name is the "
      "shared prefix between all files in the checkpoint.")
  parser.add_argument(
      "--tensor_name",
      type=str,
      default="",
      help="Name of the tensor to inspect")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
