import math
from nndct_shared.utils import BaseCommander
from nndct_shared.base import NNDCT_OP
from nndct_shared import nndct_graph as graph_utils

class QuantConfigerCommander(BaseCommander):

  def create_commands(self):

    def SoftFuseClamp(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CLAMP)

    def SoftFuseBatchSpaceNdToConv(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups,
                                  NNDCT_OP.BATCH_TO_SPACE_ND, NNDCT_OP.CONV2D)

    def SoftFuseConvToPad(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CONV2D,
                                  NNDCT_OP.PAD)

    def SoftFuseConvToSpaceBatchNd(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.CONV2D,
                                  NNDCT_OP.SPACE_TO_BATCH_ND)

    def SoftFuseDWConvToPad(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups,
                                  NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.PAD)


    def SoftFuseHardtanh(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.HARDTANH)

    '''
    def SoftFuseMul(graph, quant_groups):

      def __set_mul_quant_members(graph, quant_groups, start_node, c_node):
        name_lst = quant_groups[start_node]
        flag = False
        for g in quant_groups[c_node]:
          if g not in name_lst:
            if g == c_node:
              flag = True
            if flag:
              name_lst.append(g)
        for n in name_lst:
          quant_groups[n] = name_lst

      for n in graph.nodes:
        if quant_groups[n.name][
            0] == n.name and n.op.type == NNDCT_OP.MULTIPLY and graph_utils.should_pass_by_elemwise_node(
                n, graph):
          for p in graph.parents(n.name):
            start_node = quant_groups[p.name][0]
            __set_mul_quant_members(graph, quant_groups, start_node, n.name)

      return quant_groups
    '''

    def SoftFuseReduceSum(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.SUM)

    def SoftFuseRelu(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.RELU)

    #def SoftFuseSigmoid(graph, quant_groups):
    #  return graph_utils.group_up(graph, quant_groups, NNDCT_OP.SIGMOID)

    def SoftFuseRelu6(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.RELU6)

    def SoftFuseFlatten(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.FLATTEN)

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
        for p in graph.parents(n.name):
          if is_reshape_parent(p):
            if quant_groups[
                n.name][0] == n.name and n.op.type == NNDCT_OP.RESHAPE:
              start_node = quant_groups[p.name][0]
              groups = graph_utils.glue_group_members(graph, quant_groups,
                                                      start_node, n.name)
      return quant_groups

    # def SoftFuseSoftmax(graph, quant_groups):
    #   return graph_utils.group_up(graph, quant_groups, NNDCT_OP.SOFTMAX)

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
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.DEVIDE)

    def SoftFuseExp(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.EXP)

    def SoftFuseExpand(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.EXPAND)

    def SoftFuseInplaceCopy(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.INPLACE_COPY)

    def SoftFuseRepeat(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.REPEAT)

    def SoftFuseSelect(graph, quant_groups):
      return graph_utils.group_up(graph, quant_groups, NNDCT_OP.SELECT)

    return locals()

