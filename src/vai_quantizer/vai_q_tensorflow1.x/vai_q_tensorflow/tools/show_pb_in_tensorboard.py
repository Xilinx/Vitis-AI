import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import tensorflow.contrib.decent_q

#  cpu_kernel_path = '/home/shengxiao/workspace/graffitist/kernels/quantize_ops.so'
#  gpu_kernel_path = '/home/shengxiao/workspace/graffitist/kernels/quantize_ops_cuda.so'

#  if tf.test.is_built_with_cuda() and tf.test.is_gpu_available(cuda_only=True):
  #  tf.load_op_library(gpu_kernel_path)
#  else:
  #  tf.load_op_library(cpu_kernel_path)

tf.app.flags.DEFINE_string('gpu', '0', 'model folder')
tf.app.flags.DEFINE_integer('port', '6006', 'port number')
tf.app.flags.DEFINE_string('input_pb', '', 'meta_graph file')
FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
FLAGS = tf.app.flags.FLAGS

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile(FLAGS.input_pb, "rb").read())
_ = tf.import_graph_def(graphdef, name="")

summary_write = tf.summary.FileWriter("./logdir/", graph)

os.system('tensorboard --logdir ./logdir/ --port {}'.format(FLAGS.port))
