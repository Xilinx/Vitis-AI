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

  #for quantization module:
  QUANT_MODE = "quant_mode"
  QUANTIZER = "nndct_quantizer"
  QUANT_SUFFIX = '_quant.json'
  QUANT_DEVICE = "quant_device"

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

  


# TODO(yuwang): Move to base_operation.py
class NNDCT_OP(object):
  ADAPTIVEAVGPOOL2D = 'adaptive_avg_pool2d'
  ADD = 'elemwise_add'
  ARANGE = 'arange'
  ARGMAX = 'argmax'
  AVG_POOL = 'avgpool'
  ADDMM = 'addmm'
  BASIC_GRU = 'basic_gru'
  BASIC_LSTM = 'basic_lstm'
  BATCH_NORM = 'batch_norm'
  BATCH_NORM1D = 'batch_norm_1d'
  BATCH_NORM3D = 'batch_norm_3d'
  BATCH_TO_SPACE_ND = 'batch_to_space_nd'
  BIAS_ADD = 'bias_add'
  BIDIRECTIONAL_RNN = 'bidirectional_rnn'
  BMM = "bmm"
  BUFFER_GET_NEXT = 'buffer_get_next'
  CAST = 'cast'
  CEIL = 'ceil'
  CHANNEL_SCALE = 'channel_scale'
  CHUNK = 'chunk'
  CLAMP = 'clamp'
  CONCAT = 'concat'
  CONST = 'const'
  CONTIGUOUS = 'contiguous'
  CONV1D = 'conv1d'
  CONV2D = 'conv2d'
  CONV3D = 'conv3d'
  CONVTRANSPOSE2D = 'conv_transpose_2d'
  CONVTRANSPOSE3D = 'conv_transpose_3d'
  DENSE = 'dense'
  DEPTHWISE_CONV2D = 'depthwise_conv2d'
  DEPTHWISE_CONV3D = 'depthwise_conv3d'
  DEPTHWISE_CONVTRANSPOSE2D = 'depthwise_conv_transpose_2d'
  DEPTHWISE_CONVTRANSPOSE3D = 'depthwise_conv_transpose_3d'
  DEQUANT_STUB = 'dequant_stub'
  DETACH = 'detach'
  DIV = 'elemwise_div'
  DERIVE_LOOP_INDEX = 'derive_loop_index'
  DROPOUT = 'dropout'
  EMBEDDING = 'embedding'
  EMBEDDING_BAG = 'embedding_bag'
  EMPTY = 'empty'
  EQUAL = 'equal'
  SCALAR_EQUAL = 'scalar_equal'
  EXP = 'elemwise_exp'
  EXPAND = 'expand'
  EXPAND_AS = 'expand_as'
  FLATTEN = 'flatten'
  FLOOR = 'floor'
  FLOOR_DIV = 'floor_divide'
  FPGA_OP = 'fpga_op'
  GATHER = 'gather'
  GENERIC = 'generic'
  GRID_SAMPLE = 'grid_sample'
  GRU = 'gru'
  HARDTANH = 'hardtanh'
  HSIGMOID = 'hsigmoid'
  HSWISH = 'hswish'
  IDENTITY = 'identity'
  IF = 'if'
  INDEX = 'index'
  INDEX_INPUT_INPLACE = 'index_put_inplace'
  INPLACE_COPY = 'copy_'
  INPUT = 'input'
  INPUT_WITH_DEFAULT = 'input_with_default'
  INT = 'int'
  INTERPOLATE = 'interpolate'
  ITER_GET_NEXT = 'iter_get_next'
  LAYER_NORM = 'layer_norm'
  LEAKY_RELU = 'leaky_relu'
  LENGTH = 'len'
  LINEAR = 'linear'
  LINEAR = 'linear'
  LIST = 'list'
  LIST_ADD = "list_add"
  LOG = 'log'
  LOOP = 'loop'
  LSTM = 'lstm'
  LSTM_CELL = 'lstm_cell'
  MATMUL = 'matmul'
  MAX = 'max'
  MAX_POOL = 'maxpool'
  MAX_POOL1D = 'maxpool1d'
  MEAN = 'mean'
  MERGE = 'merge'
  MIN = "min"
  MULTIPLY = 'elemwise_mul'
  NEG = 'neg'
  NOOP = 'noop'
  NORM = 'normalize'
  NOT_EQUAL = 'not_equal'
  ONE_HOT = 'one_hot'
  PACK = 'pack'
  PAD = 'pad'
  PERMUTE = 'permute'
  PIXEL_SHUFFLE = 'pixel_shuffle'
  PIXEL_UNSHUFFLE = 'pixel_unshuffle'
  PLACEHOLDER = 'placeholder'
  QUANT_NEURON = 'quant_neuron'
  QUANT_STUB = 'quant_stub'
  RANDOM_UNIFORM = 'random_uniform'
  RANGE = 'range'
  REALDIV = 'real_div'
  RELU = 'relu'
  RELU6 = 'relu6'
  RELUK = 'reluk'
  REPEAT = 'repeat'
  RESHAPE = 'reshape'
  RESIZE = 'resize'
  RESIZE_3D = 'resize_3d'
  RESIZE_NEAREST_3D = 'resize_nearest_3d'
  REORG = 'reorg'
  RNN = 'rnn'
  RNN_LAYER = 'rnn_layer'
  RSQRT = 'rsqrt'
  RSUB = 'rsub'
  SCALAR_ADD = "add"
  SCALAR_MUL = 'mul'
  SCALAR_SUB = 'sub'
  SCALAR_LESS_THAN = 'scalar_lt'
  SELECT = 'select'
  SHAPE = 'shape'
  SIGMOID = 'sigmoid'
  SIMPLE_RNN = 'simple_rnn'
  SLICE = 'slice'
  SLICE_TENSOR_INPLACE_COPY = 'slice_tensor_inplace_copy'
  SOFTMAX = 'softmax'
  SPACE_TO_BATCH_ND = 'space_to_batch_nd'
  SPARSE_SOFTMAX_CROSS_ENTROPY = 'sparse_softmax_cross_entropy_with_logits'
  SPLIT = 'split'
  SQUARE = 'square'
  SQUEEZE = 'squeeze'
  STACK = 'stack'
  STACKED_RNN_CELLS = 'stacked_rnn_cells'
  STRIDED_SLICE = 'strided_slice'
  STRIDED_SLICE_INPLACE_COPY = "strided_slice_inplace_copy"
  SUB = 'elementwise_sub'
  SUM = 'sum'
  TANH = 'tanh'
  TENSOR = 'tensor'
  TENSOR_ARRAY_GATHER = 'tensor_array_gather'
  TENSOR_TO_SCALAR = 'tensor_to_scalar'
  THRESHOLD = 'threshold'
  TILE = 'tile'
  TRANSPOSE = 'transpose'
  TUPLE_UNPACK = 'tuple_unpack'
  UNSQUEEZE = "unsqueeze"
  UP_SAMPLING = 'up_sampling'
  ZEROS = 'zeros'

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
