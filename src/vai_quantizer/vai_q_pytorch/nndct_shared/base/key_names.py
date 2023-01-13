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

import sys

class GlobalMap(object):
  globalmap = {}

  def set_map(self, key, value):
    self.globalmap[key] = value

  def set(self, **keys):
    try:
      for key_, value_ in keys.items():
        self.globalmap[key_] = str(value_)
        print(key_ + ":" + str(value_))
    except BaseException as msg:
      print(msg)
      raise msg

  def del_map(self, key):
    try:
      del self.globalmap[key]
      return self.globalmap
    except KeyError:
      pass
      #print("key:'" + str(key) + "'  not found!")

  def get_ele(self, key):
    if key in self.globalmap:
      return self.globalmap[key]
    return None

  def get(self, *args):
    try:
      dic = {}
      for key in args:
        if len(args) == 1:
          dic = self.globalmap[key]
          print(key + ":" + str(dic))
        elif len(args) == 1 and args[0] == 'all':
          dic = self.globalmap
        else:
          dic[key] = self.globalmap[key]
      return dic
    except KeyError:
      print("key:'" + str(key) + "'  not found!")
      return 'Null_'

class NNDCT_KEYS(object):
  #basic names
  INFO_FLAG = "NNDCT_NOTE"
  WARN_FLAG = "NNDCT_WARN"
  DEBUG_FLAG = "NNDCT_DEBUG"
  ERROR_FLAG = "NNDCT_ERROR"
  VERBOSE_LEVEL = 'nndct_verbose_lvl'
  LOG_LEVEL = 'nndct_log_lvl'
  LOGGER = 'nndct_logger'
  SUFFIX_CONNECT = 'SUFFIX'

  #for debug
  COMPILER = 'nndct_compiler'
  OUTPUT_TO_NODE_MAP = 'output_to_node_map'
  NODE_TO_OUTPUT_MAP = 'node_to_output_map'

  #for Xgraph & Xnode
  XMODEL_SUFFIX = '.xmodel'
  XMODEL_IMAGE_SUFFIX = '.svg'
  XPARAM_SUFFIX = '.xparams'
  XPATTERN_SUFFIX = '.xpattern'
  XBLOBS_SUFFIX = '.xblobs'

  #for parsing/exporting
  TORCH_REFLECT_OPS_MAP = 'torch_reflect_ops_map'
  TORCH_PARSER_MAP = 'torch_parser_map'
  TORCH_SUPPORT_OPS_MAP = 'torch_support_ops_map'
  TORCH_PARAM_MAP = 'torch_parameters_name_map'
  TORCH_IR_ATTRS_MAP = 'torch_ir_attrs_map'
  TORCH_SCHEMA_OP_TABLE = 'torch_schema_op_table'
  NODE_CALLER_MAP = 'node_caller_map'
  CUSTOM_OP_ATTRS_MAP = 'custom_op_attrs_map'
  CUSTOM_TO_XIR_LIST = 'custom_to_xir_list'
  DEVICE = 'device'
  TORCH_SCRIPT_MODEL = 'torch_script_model'
  #for quantization module:
  QUANT_MODE = "quant_mode"
  QUANTIZER = "nndct_quantizer"
  QUANT_SUFFIX = '_quant.json'
  QUANT_DEVICE = "quant_device"
  QUANT_CONFIG = "quant_config"

  PARAM_SCAN_SCOPE = "ParamScan"
  BLOB_SCAN_SCOPE = "BlobScan"

  QUANT_PARAMSCAN_OPS_COLLECTION = "qaunt_paramscan_ops_collection"

  BLOB_PREFFIX = "Blob"
  MAX_SCAN_SUFFIX = SUFFIX_CONNECT + "maxscan"
  MIN_SCAN_SUFFIX = SUFFIX_CONNECT + "minscan"
  DIFFS_SCAN_SUFFIX = SUFFIX_CONNECT + "diffs"

  QUANTTABLE_VAR_SUFFIX = SUFFIX_CONNECT + "QuantTableVar"

  #for load module
  NNDCT_LOADER = 'nndct_loader'
  LOAD_FLAG = 'load_flag'
  ORGVARS_SUFFIX = '_OrgVars.json'
  ORGKERASMODEL_SUFFIX = '_OrgKerasModel.json'

  #for modification process
  MODIFIER = 'nndct_modifier'
  TRANS_SCOPE = 'TransScp'

  #for graph export
  IR_GRAPH = 'nndct_ir_graph'
  IR_NAME = 'nndct_ir_name'
  IR_EXPORT_TYPE = 'ir_export_type'

  #for training and controlling
  NRS_COLLECTION = "non_restorable_collection"
  NGTS_COLLECTION = "non_grad_tensor_collection"
  DEBUG_COLLECTION = "nndct_debug_collection"

  #for compile
  PARAMETER_FILE = 'NndctParameter'
  ISTRUCTION_FILE = 'NndctInstruction'
  WORKSPACE_PATH = 'NndctWorkspace'
  INPUT_FILE = 'NndctInput'
  DEVOP_PREFFIX = 'fpga_op_'
  FIX_OP_SUFFIX = '_fix'
  PRE_FIX_OP_SUFFIX = '_pre_fix'
  TRANSPOSE_OP_SUFFIX = '_t'
  #deploy
  DEPLOY_CHECK_DATA_FOLDER = 'deploy_check_data'

class NNDCT_OP(object):
  ADAPTIVEAVGPOOL2D = 'nndct_adaptive_avg_pool2d'
  ADD = 'nndct_elemwise_add'
  ADDMM = 'nndct_addmm'
  ANGLE = 'nndct_angle'
  ARANGE = 'nndct_arange'
  ARGMAX = 'nndct_argmax_no_dim'
  ARGMAX_DIM = 'nndct_argmax_dim'
  AVG_POOL = 'nndct_avgpool'
  BASIC_GRU = 'nndct_basic_gru'
  BASIC_LSTM = 'nndct_basic_lstm'
  BATCH_NORM = 'nndct_batch_norm'
  INSTANCE_NORM = 'nndct_instance_norm'
  BATCH_TO_SPACE_ND = 'nndct_batch_to_space_nd'
  BIAS_ADD = 'nndct_bias_add'
  BIDIRECTIONAL_RNN = 'nndct_bidirectional_rnn'
  BMM = 'nndct_bmm'
  BUFFER_GET_NEXT = 'nndct_buffer_get_next'
  BLOCK = 'nndct_block'
  CAST = 'nndct_cast'
  CEIL = 'nndct_ceil'
  CHANNEL_SCALE = 'nndct_channel_scale'
  CORRELATION1D_ELEMWISE = 'nndct_correlation1d_elemwise'
  CORRELATION2D_ELEMWISE = 'nndct_correlation2d_elemwise'
  COST_VOLUME = 'nndct_cost_volume'
  CHUNK = 'nndct_chunk'
  CLAMP = 'nndct_clamp'
  COMPLEX_ABS = 'nndct_complex_abs'
  CONCAT = 'nndct_concat'
  CONSTANT_WITH_RESHAPE = "constant_with_reshape"
  CONST = 'nndct_const'
  CONTIGUOUS = 'nndct_contiguous'
  CONV1D = 'nndct_conv1d'
  CONV2D = 'nndct_conv2d'
  CONV3D = 'nndct_conv3d'
  CONVTRANSPOSE2D = 'nndct_conv_transpose_2d'
  CONVTRANSPOSE3D = 'nndct_conv_transpose_3d'
  DENSE = 'nndct_dense'
  DEPTHWISE_CONV1D = 'nndct_depthwise_conv1d'
  DEPTHWISE_CONV2D = 'nndct_depthwise_conv2d'
  DEPTHWISE_CONV3D = 'nndct_depthwise_conv3d'
  DEPTHWISE_CONVTRANSPOSE2D = 'nndct_depthwise_conv_transpose_2d'
  DEPTHWISE_CONVTRANSPOSE3D = 'nndct_depthwise_conv_transpose_3d'
  DEQUANT_STUB = 'nndct_dequant_stub'
  DERIVE_LOOP_INDEX = 'nndct_derive_loop_index'
  DETACH = 'nndct_detach'
  DEVICE = 'nndct_device'
  DTYPE = 'nndct_dtype'
  DIV = 'nndct_elemwise_div'
  DROPOUT = 'nndct_dropout'
  EMBEDDING = 'nndct_embedding'
  EMBEDDING_BAG = 'nndct_embedding_bag'
  EMPTY = 'nndct_empty'
  EQUAL = 'nndct_equal'
  EXP = 'nndct_elemwise_exp'
  EXPAND = 'nndct_expand'
  EXPAND_AS = 'nndct_expand_as'
  FLATTEN = 'nndct_flatten'
  FLOOR = 'nndct_floor'
  FLOOR_DIV = 'nndct_floor_divide'
  FIX = 'nndct_fix'
  FPGA_OP = 'nndct_fpga_op'
  GATHER = 'nndct_gather'
  GELU = 'nndct_GELU'
  GENERIC = 'nndct_generic'
  GRID_SAMPLE = 'nndct_grid_sample'
  GROUP_NORM = 'nndct_group_norm'
  GRU = 'nndct_gru'
  HARDTANH = 'nndct_hardtanh'
  HSIGMOID = 'nndct_hsigmoid'
  HSWISH = 'nndct_hswish'
  IDENTITY = 'nndct_identity'
  IF = 'nndct_if'
  INDEX = 'nndct_index'
  INDEX_INPUT_INPLACE = 'nndct_index_put_inplace'
  INPLACE_COPY = 'nndct_copy_'
  INPUT = 'nndct_input'
  INPUT_WITH_DEFAULT = 'nndct_input_with_default'
  INT = 'nndct_int'
  INTERPOLATE = 'nndct_interpolate'
  IRFFT = 'nndct_irfft'
  ITER_GET_NEXT = 'nndct_iter_get_next'
  LAYER_NORM = 'nndct_layer_norm'
  LEAKY_RELU = 'nndct_leaky_relu'
  LENGTH = 'nndct_len'
  LINEAR = 'nndct_linear'
  LINEAR = 'nndct_linear'
  LIST = 'nndct_list'
  LIST_ADD = 'nndct_list_add'
  LOG = 'nndct_log'
  LOG_SOFTMAX = 'nndct_log_softmax'
  LOOP = 'nndct_loop'
  LSTM = 'nndct_lstm'
  LSTM_CELL = 'nndct_lstm_cell'
  MATMUL = 'nndct_matmul'
  MAX = 'nndct_max'
  MAX_POOL = 'nndct_maxpool'
  MAX_POOL1D = 'nndct_maxpool1d'
  MEAN = 'nndct_mean'
  MERGE = 'nndct_merge'
  MIN = 'nndct_min'
  MISH = 'nndct_mish'
  MULTIPLY = 'nndct_elemwise_mul'
  NANQUANTILE = 'nndct_nanquantile'
  NEG = 'nndct_neg'
  NOOP = 'nndct_noop'
  NORM = 'nndct_normalize'
  NOT_EQUAL = 'nndct_not_equal'
  NON_TENSOR_SUB = 'nndct_non_tensor_sub'
  ONE_HOT = 'nndct_one_hot'
  PACK = 'nndct_pack'
  PAD = 'nndct_pad'
  PERMUTE = 'nndct_permute'
  PIXEL_SHUFFLE = 'nndct_pixel_shuffle'
  PIXEL_UNSHUFFLE = 'nndct_pixel_unshuffle'
  PLACEHOLDER = 'nndct_placeholder'
  PRELU = 'nndct_prelu'
  QUANT_NEURON = 'nndct_quant_neuron'
  QUANT_STUB = 'nndct_quant_stub'
  QUANTILE = 'nndct_quantile'
  RANDOM_UNIFORM = 'nndct_random_uniform'
  RANGE = 'nndct_range'
  REALDIV = 'nndct_real_div'
  RELU = 'nndct_relu'
  RELU6 = 'nndct_relu6'
  RELUK = 'nndct_reluk'
  REORG = 'nndct_reorg'
  REPEAT = 'nndct_repeat'
  RESHAPE = 'nndct_reshape'
  RESIZE = 'nndct_resize'
  RESIZE_3D = 'nndct_resize_3d'
  RESIZE_NEAREST_3D = 'nndct_resize_nearest_3d'
  RETURN = 'nndct_return'
  RFFT = 'nndct_rfft'
  RNN = 'nndct_rnn'
  RNN_LAYER = 'nndct_rnn_layer'
  RSQRT = 'nndct_rsqrt'
  RSUB = 'nndct_rsub'
  REMAINDER = 'nndct_remainder'
  SCALAR_ADD = 'nndct_add'
  SCALAR_EQUAL = 'nndct_scalar_equal'
  SCALAR_LESS_THAN = 'nndct_scalar_lt'
  SCALAR_MUL = 'nndct_mul'
  SCALAR_SUB = 'nndct_sub'
  SCALAR_REMAINDER = 'nndct_scalar_remainder'
  SELECT = 'nndct_select'
  SHAPE = 'nndct_shape'
  SHAPE_AS_TENSOR = 'nndct_shape_as_tensor'
  SIGMOID = 'nndct_sigmoid'
  SIMPLE_RNN = 'nndct_simple_rnn'
  SLICE = 'nndct_slice'
  SLICE_TENSOR_INPLACE_COPY = 'nndct_slice_tensor_inplace_copy'
  SOFTMAX = 'nndct_softmax'
  SPACE_TO_BATCH_ND = 'nndct_space_to_batch_nd'
  SPARSE_SOFTMAX_CROSS_ENTROPY = 'nndct_sparse_softmax_cross_entropy_with_logits'
  SPLIT = 'nndct_split'
  SQUARE = 'nndct_square'
  SQUEEZE = 'nndct_squeeze'
  STACK = 'nndct_stack'
  STACKED_RNN_CELLS = 'nndct_stacked_rnn_cells'
  STFT = 'nndct_stft'
  STRIDED_SLICE = 'nndct_strided_slice'
  STRIDED_SLICE_INPLACE_COPY = 'nndct_strided_slice_inplace_copy'
  SUB = 'nndct_elementwise_sub'
  SUM = 'nndct_sum'
  TANH = 'nndct_tanh'
  TENSOR = 'nndct_tensor'
  TENSOR_ARRAY_GATHER = 'nndct_tensor_array_gather'
  TENSOR_TO_SCALAR = 'nndct_tensor_to_scalar'
  THRESHOLD = 'nndct_threshold'
  TILE = 'nndct_tile'
  TRANSPOSE = 'nndct_transpose'
  TUPLE = 'nndct_tuple'
  TUPLE_INPUT = 'nndct_tuple_input'
  TUPLE_INDEX = 'nndct_tuple_index'
  TUPLE_UNPACK = 'nndct_tuple_unpack'
  UNSQUEEZE = 'nndct_unsqueeze'
  UP_SAMPLING = 'nndct_up_sampling'
  ZEROS = 'nndct_zeros'
  UNIQUE_DIM = 'nndct_unique_dim'
  _UNIQUE2 = 'nndct_unique2'
  _UNIQUE = 'nndct_unique'
  

  
class NNDCT_PARAM(object):
  WEIGHT = 'weights'
  BIAS = 'bias'
  GAMMA = 'gamma'
  BETA = 'beta'
  VAR = 'var'
  MEAN = 'mean'

class FrameworkType(object):
  # Frontend types
  TORCH = 'torch'
  CAFFE = 'caffe'
  TENSORFLOW = 'tensorflow'
  TF_KERAS = 'tf_keras'

  # NNDCT as a bridge
  NNDCT = 'nndct'

class NNDCT_CONSTANT(object):
  INT_MAX = 2 ** 31 - 1
