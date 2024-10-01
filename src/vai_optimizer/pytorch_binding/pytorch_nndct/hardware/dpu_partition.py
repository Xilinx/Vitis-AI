# Copyright 2022 Xilinx Inc.
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

from nndct_shared.nndct_graph import graph_searcher
from nndct_shared.base import NNDCT_OP
from .dpu_op_filter import filters, pattern_filters
from .device import DeviceInfo, DeviceType
class DPUPartition(object):
  def __init__(self, target):
    self._dpu_target = target.get_devices()[0].get_legacy_dpu_target()
    self._cpu_only_xir_type = [
      NNDCT_OP.PERMUTE,
      NNDCT_OP.SIGMOID,
      NNDCT_OP.TANH,
      NNDCT_OP.HARDTANH,
      NNDCT_OP.SUM,
      NNDCT_OP.STRIDED_SLICE,
      NNDCT_OP.NEG,
      NNDCT_OP.EXP,
      NNDCT_OP.SUB,
      NNDCT_OP.DIV,
      NNDCT_OP.RSUB,
    ]

    self._supported_nndct_type = [
      NNDCT_OP.INPUT,
      # NNDCT_OP.CONV1D,
      NNDCT_OP.CONV2D,
      NNDCT_OP.DEPTHWISE_CONV2D,
      NNDCT_OP.CONVTRANSPOSE2D,
      NNDCT_OP.AVG_POOL,
      NNDCT_OP.ADAPTIVEAVGPOOL2D,
      NNDCT_OP.MAX_POOL,
      # NNDCT_OP.MAX_POOL1D,
      NNDCT_OP.RELU,
      NNDCT_OP.LEAKY_RELU,
      NNDCT_OP.TANH,
      NNDCT_OP.SIGMOID,
      NNDCT_OP.DENSE,
      NNDCT_OP.MATMUL,
      NNDCT_OP.RESHAPE,
      NNDCT_OP.ADD,
      NNDCT_OP.FLATTEN,
      NNDCT_OP.CONCAT,
      NNDCT_OP.MULTIPLY,
      NNDCT_OP.STRIDED_SLICE,
      NNDCT_OP.RSUB,
      NNDCT_OP.SUB,
      NNDCT_OP.PAD,
      NNDCT_OP.RESIZE,
      NNDCT_OP.SOFTMAX,
      NNDCT_OP.PERMUTE,
      NNDCT_OP.CONST,
      NNDCT_OP.TENSOR,
      NNDCT_OP.RELU6,
      NNDCT_OP.MEAN,
      NNDCT_OP.BATCH_NORM,
      NNDCT_OP.QUANT_STUB,
      NNDCT_OP.MAX,
      NNDCT_OP.TRANSPOSE,
      NNDCT_OP.SQUEEZE,
      NNDCT_OP.ZEROS,
      NNDCT_OP.NEG,
      NNDCT_OP.DIV,
      NNDCT_OP.SUM,
      NNDCT_OP.HSIGMOID,
      NNDCT_OP.HSWISH,
      NNDCT_OP.PIXEL_SHUFFLE,
      NNDCT_OP.PIXEL_UNSHUFFLE,
      NNDCT_OP.CONV3D,
      NNDCT_OP.DEPTHWISE_CONV3D,
      NNDCT_OP.CONVTRANSPOSE3D,
      NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D,
      NNDCT_OP.RESIZE_3D,
      NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D
    ]

  def simple_allocate_op_device(self, graph):
    """only use partition pass of xcompile to set device info of operation"""
    for node in graph.nodes:
      if not node.in_nodes or node.target_device is not None:
        continue
      if node.in_quant_part is False:
        node.target_device = DeviceInfo(DeviceType.CPU)
        node.target_device.set_filter_message("Not in quantizable part.")
      elif node.op.type not in self._supported_nndct_type:
        node.target_device = DeviceInfo(DeviceType.CPU)
        node.target_device.set_filter_message(f"{node.op.type} can't be converted to XIR.")
      elif node.op.type in self._cpu_only_xir_type:
        node.target_device = DeviceInfo(DeviceType.CPU)
        node.target_device.set_filter_message(f"DPU does not support {node.op.type}")
      else:
        if node.op.type in filters:
          ret, filter_msg = filters[node.op.type](node, self._dpu_target)
          if ret:
            node.target_device = DeviceInfo(DeviceType.DPU)
          else:
            node.target_device = DeviceInfo(DeviceType.CPU)
            node.target_device.set_filter_message(filter_msg)
        else:
          node.target_device = DeviceInfo(DeviceType.DPU)
    
    # handle input node
    for node in graph.nodes:
      if not node.in_nodes:
        if node.in_quant_part is False:
          node.target_device = DeviceInfo(DeviceType.CPU)
          node.target_device.set_filter_message("Not in quantizable part.")
        elif any([cn.target_device.get_device_type() == DeviceType.DPU for cn in graph.children(node)]):
          if node.op.type in self._supported_nndct_type:
            node.target_device = DeviceInfo(DeviceType.DPU)
          else:
            node.target_device = DeviceInfo(DeviceType.CPU)
            node.target_device.set_filter_message(f"{node.op.type} can't be converted to XIR.")
        else:
          node.target_device = DeviceInfo(DeviceType.CPU)
          node.target_device.set_filter_message(f"All the children nodes are assigned to CPU.")

    for pattern_filter in pattern_filters:
      pattern_filter(graph, self._dpu_target)

          



      

