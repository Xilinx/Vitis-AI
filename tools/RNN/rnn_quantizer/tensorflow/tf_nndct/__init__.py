
import copy as _copy
import sys as _sys
import tensorflow as tf
import imp
from nndct_shared.utils import NndctOption, option_util, NndctScreenLogger

lib_file = imp.find_module('nndct_kernels', __path__)[1]
kernels = tf.load_op_library(lib_file)

def _init_tf_nndct_module():
  _pop_out_argv = []
  argv = _copy.copy(_sys.argv)
  for cmd_pos, option in enumerate(_sys.argv[1:], 1):
    _pop_out_argv.extend(
        option_util.add_valid_nndct_option(argv, option, cmd_pos, 'tensorflow'))

  for item in _pop_out_argv:
    _sys.argv.remove(item)

  if NndctOption.nndct_option_list.value:
    print("Usage: python file [option]")
    print("Nndct options:")
    for option in option_util.get_all_options():
      print(option)
    _sys.exit()

  if NndctOption.nndct_help.value:
    pass
    _sys.exit()

_init_tf_nndct_module()

from tf_nndct.quantization.api import tf_quantizer
