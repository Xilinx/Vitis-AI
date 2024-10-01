# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Quantization API functions for tf.keras models."""

import os
import sys

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis import vai_q_tensorflow
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger


class PBQuantizer(object):
  """Quantize pb model using TF1 quantizer vai-q-tensorflow"""

  def __init__(self, float_model):
    self._float_model = float_model

  def _parse_input_frozen_graph(self, input_frozen_graph_file):
    """Parse input_frozen_graph configurations"""

    if input_frozen_graph_file == '':
      raise ValueError("No --input_frozen_graph assigned.")

    if not tf.io.gfile.exists(input_frozen_graph_file):
      raise ValueError("Input frozen graph file '" + input_frozen_graph_file +
                       "' does not exist!")

    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(input_frozen_graph_file, "rb") as f:
      graph_def.ParseFromString(f.read())

    return graph_def

  def _parse_input_fn(self, input_fn_str):
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

  def _parse_session_config(self, gpu_memory_fraction):
    """Parse session configurations"""

    s_config = tf.compat.v1.ConfigProto()
    s_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    # Disable graph optimizer and rewiter to make sure every quantize node works correctly
    # the next two operation have been moved into `quantize_frozen` and `dump` function
    # s_config.graph_options.optimizer_options.opt_level = -1
    # s_config.graph_options.rewrite_options.disable_meta_optimizer = True
    return s_config

  def quantize_model(self,
                     input_fn=None,
                     calib_steps=None,
                     input_shape=None,
                     input_layers=None,
                     output_layers=None,
                     gpu=None,
                     output_dir="./quantize_pb",
                     method=1,
                     gpu_memory_fraction=0.5,
                     do_cle=0,
                     ignore_nodes=[],
                     weight_bit=8,
                     activation_bit=8,
                     nodes_bit=[],
                     nodes_method=[],
                     target_type="",
                     calib_iter=1,
                     align_concat=0,
                     align_pool=0,
                     adjust_shift_bias=1,
                     adjust_shift_cut=1,
                     simulate_dpu=1,
                     scale_all_avgpool=1,
                     replace_relu6=1,
                     replace_sigmoid=0,
                     replace_softmax=0,
                     add_shapes=True,
                     keep_float_weight=True,
                     fold_constant=False,
                     fix_input_shape=False,
                     debug_mode=False,
                     skip_check=0,
                     dump_as_xir=False,
                     fuse_op_config="",
                     convert_datatype=0,
                     custom_op_set=set(),
                     onnx_opset=13,
                     output_format="pb"):

    if gpu is not None:
      os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    self._calib_input_fn = input_fn
    self._calib_iter = calib_steps
    self._input_shapes = input_shape
    self._input_nodes = input_layers
    self._output_nodes = output_layers

    #input_graph_def = self._parse_input_frozen_graph(self._float_model)
    input_graph_def = self._float_model
    q_config = vai_q_tensorflow.QuantizeConfig(
        input_nodes=self._input_nodes,
        output_nodes=self._output_nodes,
        input_shapes=self._input_shapes,
        ignore_nodes=ignore_nodes,
        weight_bit=weight_bit,
        activation_bit=activation_bit,
        nodes_bit=nodes_bit,
        nodes_method=nodes_method,
        method=method,
        target_type=target_type,
        calib_iter=self._calib_iter,
        output_dir=output_dir,
        align_concat=align_concat,
        align_pool=align_pool,
        adjust_shift_bias=adjust_shift_bias,
        adjust_shift_cut=adjust_shift_cut,
        simulate_dpu=simulate_dpu,
        scale_all_avgpool=scale_all_avgpool,
        do_cle=do_cle,
        replace_relu6=replace_relu6,
        replace_sigmoid=replace_sigmoid,
        replace_softmax=replace_softmax)

    if convert_datatype:
      vai_q_tensorflow.convert_datatype(input_graph_def, q_config,
                                        convert_datatype)
      return

    calib_input_fn = self._parse_input_fn(self._calib_input_fn)
    s_config = self._parse_session_config(gpu_memory_fraction)
    #s_config = _parse_session_config(flags.gpu_memory_fraction)
    vai_q_tensorflow.quantize_frozen(input_graph_def, calib_input_fn, q_config,
                                     s_config, skip_check, dump_as_xir,
                                     fuse_op_config, add_shapes,
                                     keep_float_weight, output_format,
                                     onnx_opset, custom_op_set, fold_constant,
                                     fix_input_shape, debug_mode)

  def dump_model(self,
                 input_graph_def=None,
                 input_fn=None,
                 output_dir=None,
                 max_dump_batches=None,
                 dump_float=None,
                 s_config=tf.compat.v1.ConfigProto(),
                 dump_input_tensors='',
                 get_weight_only=False,
                 fuse_op_config=None,
                 custom_op_set=None):

    if input_fn is None or output_dir is None or \
       max_dump_batches is None or dump_float is None:
      logger.error('Invalid inputs input_fn {} output_dir {} '
                   'max_dump_batches {} dump_float {} for PB Quantizer'.format(
                       input_fn, output_dir, max_dump_batches, dump_float))
    self._calib_input_fn = input_fn
    calib_input_fn = self._parse_input_fn(self._calib_input_fn)
    vai_q_tensorflow.dump(
        self._float_model,
        calib_input_fn,
        output_dir,
        max_dump_batches,
        dump_float,
        s_config=tf.compat.v1.ConfigProto(),
        dump_input_tensors='',
        get_weight_only=False,
        fuse_op_config=None,
        custom_op_set=None)
