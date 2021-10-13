import tensorflow as tf
import imp

lib_file = imp.find_module('nndct_kernels', __path__)[1]
kernels = tf.load_op_library(lib_file)

from tf_nndct.quantization.api import tf_quantizer
