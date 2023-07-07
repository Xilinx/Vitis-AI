

#
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
#

from lib2to3.pgen2.token import OP
from types import DynamicClassAttribute
from .option_def import Option

####################################
# Add Option list here
####################################
# The option name include two parts: prefix "nndct" and name seperated by "_"
class NndctOption(object):
  nndct_help = Option(name="help", dtype=bool, default=False, action="store_true", 
                      help="list all api usage description")

  nndct_quant_off = Option(name="quant_off", dtype=bool, default=False, action="store_true", 
                           help="disable quantization flow")

  nndct_option_list = Option(name="option_list", dtype=bool, default=False, action="store_true", 
                             help="list all the options in nndct")

  nndct_parse_debug = Option(name="parse_debug", dtype=int, default=0, env="NNDCT_PARSE_DEBUG",
                             help="logging graph, 1: torch raw graph, 2: nndct graph 3: nndct quant graph")

  nndct_logging_level = Option(name="logging_level", dtype=int, default=0, help="logging level")
  
  nndct_quant_mode = Option(name="quant_mode", dtype=int, default=0, 
                            help="quant mode, 1:calibration, 2:quantization")
  
  nndct_dump_float_format = Option(name="dump_float_format", dtype=int, default=0, 
                            help="deploy check data format, 0: bin, 1: txt")
  
  nndct_record_slow_mode = Option(name="record_slow_mode", dtype=bool, default=False, action="store_true", 
                                  help="record outputs every iteration")
  
  nndct_quant_opt = Option(name="quant_opt", dtype=int, default=3, help="quant opt level")
  
  nndct_relu6_replace = Option(name="relu6_replace", dtype=str, default='relu', help="relu6 replace operator")

  nndct_equalization = Option(name="equalization", dtype=bool, default=True, action="store_true", 
                              help="enable weights equalization")
  
  # nndct_wes = Option(name="weights_equalizing_shift", dtype=bool, default=False, action="store_true",
  #                    help="enable weights equalizing shift")
  
  # nndct_wes_in_cle = Option(name="weights_equalizing_shift in cle", dtype=bool, default=False, action="store_true",
  #                    help="enable weights equalizing shift in cle")

  nndct_param_corr = Option(name="param_corr", dtype=bool, default=True, action="store_true", 
                            help="enable parameter correction")

  nndct_param_corr_rate = Option(name="param_corr_rate", dtype=float, default=0.05, help="parameter correction rate")

  nndct_cv_app = Option(name="cv_app", dtype=bool, default=True, action="store_true", help="cv application")

  nndct_finetune_lr_factor = Option(name="finetune_lr_factor", dtype=float, default=0.01, help="finetune learning rate factor")

  nndct_partition_mode = Option(name="partition_mode", dtype=int, default=0, 
                       help="0: quant stub controled. 1: custom op controled")

  nndct_stat = Option(name="stat", dtype=int, default=0, help="quantizer statistic level")

  nndct_jit_script_mode = Option(name="jit_script_mode", dtype=bool, default=False, action="store_true", help="enable torch script parser")

  nndct_diffs_mode = Option(name="diffs_mode", dtype=str, default='mse', help="diffs_mode: mse, maxmin")

  nndct_ft_mode = Option(name="ft_mode", dtype=int, default=1, help="1: mix mode 0: cache mode")
  
  nndct_visualize = Option(name="visualize", dtype=bool, default=False, action="store_true", help="visualize tensors")

  nndct_dump_no_quant_part = Option(name="dump_no_quant_part", dtype=bool, default=False, action="store_true", help="dump no quantized nodes")

  nndct_max_fix_position = Option(name="max_fix_position", dtype=int, default=12, help="maximum of fix position")

  nndct_use_torch_quantizer = Option(name="use_torch_quantizer", dtype=bool, default=False, action="store_true", help="enable torch quantizer")
  
  nndct_jit_trace = Option(name="jit_trace", dtype=bool, default=False, action="store_true", env="NNDCT_JIT_TRACE", help="parse graph from script tracing")
  
  nndct_jit_script = Option(name="jit_script", dtype=bool, default=False, action="store_true", help="parse graph from script")
  
  nndct_calib_histogram_bins = Option(name="calib_histogram_bins", dtype=int, default=2048, help="calibration histogram bins number")

  nndct_mse_start_bin = Option(name="mse_start_bin", dtype=int, default=1536, help="mse calibration method start bin")
  
  nndct_mse_stride = Option(name="mse_stride", dtype=int, default=4, help="mse calibration method stride")
  
  nndct_entropy_start_bin = Option(name="entropy_start_bin", dtype=int, default=1536, help="entropy calibration method start bin")
  
  nndct_entropy_stride = Option(name="entropy_stride", dtype=int, default=16, help="entropy calibration method stride")
  
  nndct_convert_relu6_to_relu = Option(name="convert_relu6_to_relu", dtype=bool, default=False, help="convert relu6 to relu")

  nndct_convert_sigmoid_to_hsigmoid = Option(name="convert_sigmoid_to_hsigmoid", dtype=bool, default=False, action="store_true", help="convert sigmoid to hsigmoid")
  
  nndct_convert_silu_to_hswish = Option(name="convert_silu_to_hswish", dtype=bool, default=False, action="store_true", help="convert silu to hswish")
  
  nndct_keep_first_last_layer_accuracy = Option(name="keep_first_last_layer_accuracy", dtype=bool, default=False, help="keep accuracy of first and last layer")
  
  nndct_keep_add_layer_accuracy = Option(name="keep_add_layer_accuracy", dtype=bool, default=False, help="keep accuracy of add layer")
  
  nndct_avg_pool_approximate = Option(name="avg_pool_approximate", dtype=bool, default=True, action="store_true", help="enable average pooling approximate for dpu")
  
  nndct_leaky_relu_approximate = Option(name="leaky_relu_approximate", dtype=bool, default=True, action="store_true", help="enable leaky relu approximate for dpu")
  
  nndct_conv_bn_merge = Option(name="conv_bn_merge", dtype=bool, default=True, action="store_true", help="enable conv and bn merge")
  
  nndct_input_quant_only = Option(name="input_quant_only", dtype=bool, default=False, action="store_false", help="only quantize the input")
  
  nndct_tensorrt_strategy = Option(name="tensorrt_strategy", dtype=bool, default=False, action="store_true", help="use quantization strategy as tensorrt")
  
  nndct_tensorrt_quant_algo = Option(name="tensorrt_quant_algo", dtype=bool, default=False, action="store_true", help="use tensorrt quantization algorithm")

  nndct_calibration_local = Option(name="calibration_local", dtype=bool, default=True, action="store_true", help="calibration in local batch data")

  nndct_change_concat_input_fix = Option(name="change_concat_input_fix", dtype=bool, default=False, action="store_true", help="change concat input nodes fix point to be the same as concat output node")

  nndct_change_pool_input_fix = Option(name="change_pool_input_fix", dtype=bool, default=False, action="store_true", help="change pooling input nodes fix point to be the same as their output node")

  nndct_change_add_input_fix = Option(name="change_add_input_fix", dtype=bool, default=False, action="store_true", help="change add input nodes fix point to be the identical")
  
  nndct_export_jit = Option(name="export_jit", dtype=bool, default=False, action="store_true", env="NNDCT_EXPORT_JIT", help="export quant script by inserting fixneuron")

  nndct_deploy_check = Option(name="deploy_check", dtype=bool, default=False, action="store_true", help="dump deploy data in forward process")
  
  nndct_input_check = Option(name="input_check", dtype=bool, default=False, action="store_true", help="dump input float data in forward process")
  
  nndct_op_tanh_sigmoid_mode = Option(name="tanh_sigmoid_mode", dtype=str, default='quant_input_output', help="Tanh/sigmoid quantization mode: quant_input_output, table_look_up, simulation, aie2_lut_16bw")
  
  nndct_op_softmax_mode = Option(name="softmax_mode", dtype=str, default='quant_input_output', help="Softmax quantization mode: quant_input_output, hardware_pl, liyi, aie2_lut_16bw, bert_8bw")
  
  nndct_op_logsoftmax_mode = Option(name="logsoftmax_mode", dtype=str, default='quant_input_output', help="Logsoftmax quantization mode: quant_input_output, aie2_lut_16bw")

  nndct_op_gelu_mode = Option(name="gelu_mode", dtype=str, default='quant_input_output', help="GELU quantization mode: quant_input_output, dynamic_table")
  
  nndct_op_layernorm_mode = Option(name="layernorm_mode", dtype=str, default='quant_input_output', help="Layernorm quantization mode: quant_input_output, aie2_16bw, bert_8bw")

  nndct_ip_asr = Option(name="ip_asr", dtype=bool, default=False, action="store_true", help="asr quant method")
  
  nndct_ip_v70_bert = Option(name="ip_v70_bert", dtype=bool, default=False, action="store_true", help="bert v70 quant method")

  nndct_ip_v70_bert_qat = Option(name="ip_v70_bert_qat", dtype=bool, default=False, action="store_true", help="bert v70 quant method")

  nndct_use_old_inspector = Option(name="use_old_inspector", dtype=bool, default=False, action="store_true", env="NNDCT_USE_OLD_INSPECTOR", help="switch to old inspector")
  
  nndct_calib_before_finetune = Option(name="calib_before_finetune", dtype=bool, default=False, action="store_true", help="calibration before fast finetune")

  nndct_inspect_debug = Option(name="inspect_debug", dtype=bool, default=False, action="store_true", env="NNDCT_INSPECT_DEBUG", help="turn on inspector")

  nndct_op_instancenorm_mode = Option(name="instancenorm_mode", dtype=str, default='quant_input_output', help="Instancenorm quantization mode: quant_input_output, ipu_8bw")

  nndct_op_groupnorm_mode = Option(name="groupnorm_mode", dtype=str, default='quant_input_output', help="Groupnorm quantization mode: quant_input_output, ipu_8bw")
  
  nndct_check_onnx = Option(name="check_onnx", dtype=bool, default=False, action="store_true", help="check exported onnx")
  
  nndct_inspect_test = Option(name="inspect_test", dtype=bool, default=False, action="store_true", env="NNDCT_INSPECT_TEST", help="embed target related test in torch quantizer")

  nndct_target = Option(name="target", dtype=str, default="", env="NNDCT_TARGET", help="target name")
  
  nndct_traversal_graph_mode = Option(name="nndct_traversal_graph_mode", dtype=int, default=0, env="NNDCT_GRAPH_SEARCH",help="0: auto, 1:recursion, 2: iteration")
  

