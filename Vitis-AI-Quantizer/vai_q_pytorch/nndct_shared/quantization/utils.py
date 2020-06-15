import copy
import numpy as np

from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP, NNDCT_DEBUG_LVL
from nndct_shared.algorithms import breadth_first_search_handler
from nndct_shared.nndct_graph import NndctGraphHolder
from nndct_shared import utils as nndct_utils
from .commander import QuantConfigerCommander
from .quant_ops import normal_quant_neuron

def quantize_data2int(data, bn, fp, method=2):
  return normal_quant_neuron(
      data, maxamps=[[2**(bn - 1)], [2**fp]], round_method=method, as_int=True)

def maybe_get_quantizer(quantizer=None):
  quantizer = quantizer or GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
  if quantizer:
    return quantizer.quant_mode, quantizer
  else:
    return GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_MODE), None

def is_quant_end_point(graph, node, quant_types):
  if len(graph.parents(node.name)) == 0:
    return False
  __QuantNodes = []

  def __check_end(node_name):
    if graph.node(node_name).op.type in quant_types:
      __QuantNodes.append(node_name)

  def __children_names(node_name):
    for c in graph.children(node_name):
      if len(__QuantNodes) >= 1:
        break
      yield c.name

  breadth_first_search_handler(
      node.name, generator=__children_names, handler=__check_end)
  return len(__QuantNodes) == 0

def get_flows_and_info(quant_mode,
                       quantizer,
                       node_name=None,
                       params=None,
                       inputs=None):
  node = quantizer.configer.get_Nndctnode(node_name, params, inputs)
  return None, quantizer.configer.quant_inputs(
      node, inputs, params), quantizer.configer.quant_output(node, params)

def process_inputs_and_params(node,
                              quant_mode,
                              quantizer,
                              inputs,
                              valid_inputs=None,
                              params=[],
                              param_names=[]):

  # calculate quantization step of input activation
  # and quantize it
  if valid_inputs:
    for idx in range(min([len(valid_inputs), len(inputs)])):
      if valid_inputs[idx]:
        if quant_mode in [1, 3]:
          quantizer.do_scan(
              inputs[idx],
              inputs[idx],
              inputs[idx],
              valid_inputs[idx],
              node,
              tensor_type='input')
        elif quant_mode == 2:
          inputs[idx] = quantizer.do_quantize(
              inputs[idx], valid_inputs[idx], node, tensor_type='input')

  # calculate quantization step of parameters
  # and quantize it
  if quant_mode in [1, 3]:
    for idx in range(len(params)):
      quantizer.do_scan(
          params[idx],
          params[idx],
          params[idx],
          param_names[idx],
          node,
          tensor_type='param')

  # only quantize parameters by pre-calculated step
  if quant_mode == 2:
    for idx in range(len(params)):
      params[idx] = quantizer.do_quantize(
          params[idx], param_names[idx], node, tensor_type='param')

  return inputs, params

def post_quant_process(node,
                       valid_output,
                       outputs=[],
                       maxmin=[],
                       quantizer=None):

  quant_mode, quantizer = maybe_get_quantizer(quantizer)
  if valid_output:
    output_name, is_quant_end = valid_output
    #print('qmode = %d, q_end: %d activation: %s' %
    #         (quant_mode, is_quant_end, output_name))
    if is_quant_end:
      if quant_mode in [1, 3]:
        for idx in range(len(outputs)):
          quantizer.do_scan(
              outputs[idx],
              maxmin[0],
              maxmin[1],
              output_name,
              node,
              tensor_type='output')
      elif quant_mode == 2:
        for idx in range(len(outputs)):
          outputs[idx] = quantizer.do_quantize(
              outputs[idx], output_name, node, tensor_type='output')

  return outputs

def default_pre_quant_process(node_name, inputs=[], params=[], quantizer=None):
  quant_mode, quantizer = maybe_get_quantizer(quantizer)
  quant_info, valid_output = None, None
  if quant_mode and quant_mode > 0:
    quant_info, valid_inputs, valid_output = get_flows_and_info(
        quant_mode, quantizer, node_name, inputs=inputs, params=params)
    inputs, params = process_inputs_and_params(quant_mode, quantizer, inputs,
                                               valid_inputs, params, params)
  return inputs, params, quant_info, valid_output

def get_amp_bnfps(bnfp):
  bn, fp = bnfp
  bn = 2**(bn - 1)
  if fp is not None:
    fp = 2**fp if fp > 0 else 1.0 / 2**(-fp)
  return [bn, fp]

def check_quant_config(q_conf):
  for pack in ['params', 'blobs']:
    assert pack in q_conf, "{} is not in quantization confige, please check!".format(
        pack)
    for k, v in q_conf[pack].items():
      assert isinstance(
          k, str), "{} in {} has type {}, not string, please check!".format(
              k, pack, type(k))
      assert isinstance(
          v, list), "{}[{}] has type {}, not list, please check!".format(
              pack, k, type(v))
      assert len(v) == 2, "{}[{}]={}, not length 2, please check!".format(
          pack, k, v)

class QuantInfoConfiger(NndctGraphHolder):

  def get_info(self, commands=None):
    commands = commands or [k for k in self.scan_commander]
    quant_groups = {n.name: [n.name] for n in self.Nndctgraph.nodes}
    while True:
      org_groups = copy.deepcopy(quant_groups)
      for c in commands:
        quant_groups = self.scan_commander[c](self.Nndctgraph, quant_groups)
      if org_groups == quant_groups:
        break
    for k, v in quant_groups.items():
      quant_groups[k] = sorted(v, key=lambda n: self.get_Nndctnode(n).idx)
    self.__QuantGroups = quant_groups
    # print(self.__QuantGroups)
    self.debug_details(
        self.__QuantGroups,
        'quant_groups',
        level=NNDCT_DEBUG_LVL.SPECIFIED_DETAILS)
    return self.__QuantGroups

  def fill_value(self, bitwidth_w, bitwidth_a, fragpos=None, lstm=False):
    config = {'params': {}, 'blobs': {}}
    for node in self.Nndctgraph.nodes:
      #print('---- Handling node %s type: %s:' % (node.name, node.op.type))
      if self.is_node_quantizable(node, lstm):
        for p in self.quant_node_params(node).values():
          config['params'][p.name] = [bitwidth_w, fragpos]
          #print('---- Add fix of param %s' % p.name)
        if self.__QuantGroups[node.name][-1] not in config['blobs']:
          config['blobs'][self.__QuantGroups[node.name][-1]] = [
              bitwidth_a, fragpos
          ]
          #print('---- Add fix of blob %s' % self.__QuantGroups[node.name][-1])
          #print(self.__QuantGroups[node.name])
      elif node in self.Nndctgraph.inputs :
        if any(self.is_node_quantizable(c, lstm) for c in self.Nndctgraph.children(node.name)) :
          config['blobs'][self.__QuantGroups[node.name][-1]] = [bitwidth_a, fragpos]
          #print('---- Add fix input blob %s' % self.__QuantGroups[node.name][-1])
      #else:
        #print('**** Ignore fix of %s' % self.__QuantGroups[node.name][0])
    return config

  @property
  def default_kwargs(self):
    default_kwargs = super().default_kwargs
    default_kwargs.update({'keep_all_graph_info': True})
    return default_kwargs

  @property
  def default_commanders(self):
    default_commanders = super().default_commanders
    default_commanders.update({QuantConfigerCommander: 'scan_commander'})
    return default_commanders
