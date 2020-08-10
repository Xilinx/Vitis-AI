import numpy as np
from nndct_shared.utils import BaseCommander
from nndct_shared.base import NNDCT_DEBUG_LVL, NNDCT_OP
from nndct_shared import utils as nndct_utils

class OptimizeCommander(BaseCommander):

  def create_commands(self):

    def FuseBnToConv(ctx, graph, param_info):
      if not ctx.use_quant:
        return graph, param_info

      nodes_to_fuse = {}
      params_to_reset = {}

      def __fuse_batchnorm(conv_node, bn_node):
        ctx.debug("fuse conv {} and batch_node {} together".format(
            conv_node.name, bn_node.name))
        if conv_node.node_attr(conv_node.op.AttrName.BIAS_TERM):
          bias_param_name = conv_node.op.params[
              conv_node.op.ParamName.BIAS].name
          bias_data = conv_node.op.params[conv_node.op.ParamName.BIAS].data
        else:
          split_sym = nndct_utils.get_split_sym(ctx.model_type)
          if ctx.model_type == 'torch':
            # bias_param_name = '.'.join(conv_node.op.params[0].split(split_sym)[:-1]) + split_sym+'bias'
            bias_param_name = '.'.join(
                conv_node.op.params[conv_node.op.ParamName.WEIGHTS].name.split(
                    split_sym)[:-1]) + split_sym + 'bias'
          else:
            bias_param_name = conv_node.name + split_sym + 'bias'
          bias_data = 0
          conv_node.set_node_attr(conv_node.op.AttrName.BIAS_TERM, True)
          # conv_node.op.params.append(bias_param_name)
          if ctx.model_type == 'tensorflow':
            bias_name = conv_node.name + split_sym + 'BiasAdd'
            conv_node.add_alias(bias_name)
            conv_node.add_node_config('bias_configs', {
                'name': bias_name,
                'data_format': conv_node.node_attr('layout')
            })
          elif ctx.model_type == 'tf-keras':
            conv_node.set_node_config('use_bias', True)
          elif ctx.model_type == 'torch':
            conv_node.set_node_attr(conv_node.op.AttrName.BIAS_TERM, True)
          else:
            raise TypeError("model_type {} is currently not supported ".format(
                ctx.model_type))
        conv_weights = conv_node.op.params[conv_node.op.ParamName.WEIGHTS].data
        bn_gamma = bn_node.op.params[bn_node.op.ParamName.GAMMA].data
        bn_beta = bn_node.op.params[bn_node.op.ParamName.BETA].data
        bn_mean = bn_node.op.params[bn_node.op.ParamName.MOVING_MEAN].data
        bn_var = bn_node.op.params[bn_node.op.ParamName.MOVING_VAR].data
        # epsilon = bn_node.node_attr('epsilon')
        epsilon = bn_node.node_attr(bn_node.op.AttrName.EPSILON)
        if all(data is not None for data in
               [conv_weights, bias_data, bn_beta, bn_gamma, bn_mean, bn_var]):
          scale = bn_gamma / np.sqrt(bn_var + epsilon)
          offset = bn_beta - bn_mean * scale
          
          if conv_node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
            # [channel_multiplier, h, w, in_channels] -> [channel_multiplier, in_channles, h, w]
            conv_weights = conv_weights.transpose(0, 3, 1, 2)
            in_dim = conv_node.node_attr(conv_node.op.AttrName.IN_DIM)
            out_dim = conv_node.node_attr(conv_node.op.AttrName.OUT_DIM)
            kernel_size = conv_node.node_attr(conv_node.op.AttrName.KERNEL)
            channel_multiplier = int(out_dim / in_dim)
            # [channel_multiplier, in_channles, h, w] -> [channel_multiplier * in_channles,  1, h, w]
            conv_weights = conv_weights.reshape((channel_multiplier * in_dim, 1, *kernel_size))
            # [channel_multiplier * in_channles,  1, h, w] -> [h, w, 1, channel_multiplier * in_channles]
            new_conv_weights = conv_weights.transpose(2, 3, 1, 0) * scale
            # [h, w, 1, channel_multiplier * in_channles] -> [channel_multiplier * in_channles, 1, h, w]
            new_conv_weights = new_conv_weights.transpose(3, 2, 0, 1)
            # [channel_multiplier * in_channles, 1, h, w] -> [channel_multiplier, in_channles, h, w]
            new_conv_weights = new_conv_weights.reshape((channel_multiplier, in_dim, *kernel_size))
            # [channel_multiplier, in_channles, h, w] -> [channel_multiplier, h, w, in_channles]
            new_conv_weights = new_conv_weights.transpose(0, 2, 3, 1)
          else:
            new_conv_weights = conv_weights.transpose(2, 3, 1, 0) * scale
            new_conv_weights = new_conv_weights.transpose(3, 2, 0, 1)
                
          conv_node.op.set_param_from_data(
              conv_node.op.ParamName.WEIGHTS,
              new_conv_weights)
          conv_node.op.set_param_from_data(conv_node.op.ParamName.BIAS,
                                           bias_data * scale + offset,
                                           bias_param_name)

        else:
          kernel_tensor = conv_node.op.params[conv_node.op.ParamName.WEIGHTS]
          if conv_node.op.type == NNDCT_OP.CONV2D:
            shape = [kernel_tensor.shape[0]]
          elif conv_node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
            shape = [kernel_tensor.shape[1]]
          else:
            raise KeyError(
                "unexpected conv type {} found during FuseBnToConv".foramt(
                    conv_node.op.type))
          conv_node.op.set_param_from_des(conv_node.op.ParamName.BIAS, {
              'shape': shape,
              'dtype': kernel_tensor.dtype
          }, bias_param_name)
        bn_params = [v.name for v in bn_node.op.params.values()]
        param_info[conv_node.op.params[conv_node.op.ParamName.WEIGHTS].name] = [{
            'type': 'FuseBnToConvKernel',
            'conv_type': conv_node.op.type,
            'params': bn_params,
            'epsilon': epsilon,
            'center': bn_node.node_attr(bn_node.op.AttrName.CENTER),
            'scale': bn_node.node_attr(bn_node.op.AttrName.SCALE)
        }]

        param_info[conv_node.op.params[conv_node.op.ParamName.BIAS].name] = [{
            'type': 'FuseBnToConvBias',
            'conv_type': conv_node.op.type,
            'params': bn_params,
            'epsilon': epsilon,
            'center': bn_node.node_attr(bn_node.op.AttrName.CENTER),
            'scale': bn_node.node_attr(bn_node.op.AttrName.SCALE)
        }]

      #find fusable bathnorm node
      for n in graph.nodes:
        if n.op.type == NNDCT_OP.BATCH_NORM and len(graph.parents(n.name)) == 1:
          p_node = graph.parents(n.name)[0]
          if p_node.op.type in [
              NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D,
              NNDCT_OP.CONVTRANSPOSE2D
          ]:
            nodes_to_fuse[p_node] = n
            __fuse_batchnorm(p_node, n)
      for p_node, bn_node in nodes_to_fuse.items():
        ctx.fuse_Nndctnode(graph, p_node)
      return graph, param_info

    return locals()
