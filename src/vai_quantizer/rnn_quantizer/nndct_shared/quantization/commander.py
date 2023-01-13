

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

import math
from nndct_shared.utils import BaseCommander
from nndct_shared.base import NNDCT_OP
from nndct_shared import nndct_graph as graph_utils

class QuantConfigerCommander(BaseCommander):

  def create_commands(self):

    # def SoftFuseClamp(graph, quant_groups):
    #   return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CLAMP)

    def SoftFuseBatchSpaceNdToConv(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups,
                                  NNDCT_OP.BATCH_TO_SPACE_ND, NNDCT_OP.CONV2D)

    def SoftFuseConvToSpaceBatchNd(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CONV2D,
                                  NNDCT_OP.SPACE_TO_BATCH_ND)

    def SoftFuseHardtanh(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.HARDTANH)

    def SoftFuseRelu(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.RELU)

    def SoftFuseLeakyRelu(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.LEAKY_RELU)

    def SoftFuseRelu6(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.RELU6)
    
    def SoftFuseReluk(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.RELUK)
    
    def SoftFuseChannelScale(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CHANNEL_SCALE)
    
    def SoftFuseFlatten(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.FLATTEN)

    def SoftFuseSqueeze(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.SQUEEZE)

    def SoftFusePixelShuffle(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.PIXEL_SHUFFLE)

    def SoftFuseReshape(graph, quant_groups):

      def is_reshape_parent(node):
        if node.op.type == NNDCT_OP.SHAPE:
          return False
        elif node.op.type in [NNDCT_OP.MULTIPLY]:
          for p in graph.parents(node.name):
            return is_reshape_parent(p)
        else:
          return True

      for n in graph.nodes:
        if not n.in_quant_part:
          continue
        for p in graph.parents(n.name):
          if is_reshape_parent(p):
            if quant_groups[
                n.name][0] == n.name and n.op.type == NNDCT_OP.RESHAPE:
              start_node = quant_groups[p.name][0]
              groups = graph_utils.glue_group_members(graph, quant_groups,
                                                      start_node, n.name)
      return quant_groups

    def SoftFuseSplit(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.SPLIT)

    def SoftFuseStrideSlice(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.STRIDED_SLICE)

    def SoftFuseTranspose(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.TRANSPOSE)

    def SoftFuseTile(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.TILE)

    def SoftFuseUpSampling(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.UP_SAMPLING)

    def SoftFuseDropout(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.DROPOUT)

    def SoftFuseContiguous(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CONTIGUOUS)

    def SoftFuseChunk(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CHUNK)

    def SoftFusePermute(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.PERMUTE)

    def SoftFuseDivide(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.DIV)

    def SoftFuseExp(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.EXP)

    def SoftFuseExpand(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.EXPAND)

    def SoftFuseInplaceCopy(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.INPLACE_COPY)

    def SoftFuseRepeat(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.REPEAT)

    # def SoftFuseSelect(graph, quant_groups):
    #   return graph_utils.group_up(graph, quant_groups, NNDCT_OP.SELECT)
    
    def SoftFuseUnsqueeze(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.UNSQUEEZE)

    return locals()


        
