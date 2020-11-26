

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

from collections import OrderedDict

from nndct_shared.nndct_graph import GraphSearcher
from nndct_shared.utils import (NndctDebugLogger, NndctOption,
                                PatternType)

from .opt_pass import OptPass
from .torch_graph import *
from .utils import *


class TorchGraphHandler(object):

  def __init__(self):
    self._transparent_ops = ["ListUnpack", "TupleUnpack"]
    
  def build_torch_graph(self, graph_name, module, input_args, train=False):
    self._module = module
    fw_graph, params = self._trace_graph_from_model(input_args, train)

    self._node_kinds = {node.kind().split(":")[-1] for node in fw_graph.nodes()}
    if NndctOption.nndct_parse_debug.value >= 1:
      NndctDebugLogger.write(f"jit graph:\n{fw_graph}")
      NndctDebugLogger.write(f"\nparsing nodes types:\n{self._node_kinds}")

    raw_graph = self._build_raw_graph(graph_name, fw_graph, params)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch raw graph:\n{raw_graph}")   
    opt_graph = self._opt_raw_graph(raw_graph)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch opt graph:\n{raw_graph}")
    return opt_graph

  @property
  def graph_nodes(self):
    for node in [self._graph.param_node()] + list(self._graph.nodes()):
      yield node

  def _trace_graph_from_model(self, input_args, train):
    graph, output = trace_and_get_graph_from_model(self._module, input_args,
                                                    train)
    graph = optimize_graph(graph)
    params = rename_graph_param_name(self._module, graph)
    return graph, params
      
  def _opt_raw_graph(self, raw_graph):
    graph_searcher = GraphSearcher(raw_graph)
    
    # shufflenet
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListUnpack"]), 
                                                        PatternType(pattern=["TupleUnpack"])])
    OptPass.unpack_ListUnpack_op(self, raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["slice"])])
    
    OptPass.slice_to_strided_slice(self, raw_graph, node_sets)
      
    # yolo_v3
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["select", "copy_"])])
    OptPass.select_to_slice_inplace_copy(self, raw_graph, node_sets)
    
    # 3d pointpillar
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "index_put_"])])
    OptPass.stride_slice_to_index_inplace_put(self, raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "copy_"])])
    OptPass.create_stride_slice_inplace_copy(self, raw_graph, node_sets)
    
    # nd(>2) linear (JIRA 2646)
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["matmul", "add"]),
                                                     PatternType(pattern=["matmul", "add_"])])
    OptPass.merge_matmul_with_add(self, raw_graph, node_sets)                                                

      
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListConstruct"]),
                                                        PatternType(pattern=["TupleConstruct"])])
    OptPass.pack_ListConstruct_op(self, raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["embedding_bag"])])
    OptPass.strip_reduantant_tensors_in_embedding_bag(self, raw_graph, node_sets)
    
    return raw_graph

  def _build_raw_graph(self, graph_name, fw_graph, params):
    raw_graph = TorchGraph.new_graph(graph_name)
    for fw_value in fw_graph.param_node().outputs():
      if unique_name(fw_value) not in params:
        input_node = TorchNode(fw_graph.param_node())
        value = TorchValue(fw_value)
        value.node = input_node
        input_node.add_output(value)
        raw_graph.add_node(input_node)
      else:
        value = TorchValue(fw_value)
        raw_graph.add_param_value(value)

    for fw_node in fw_graph.nodes():
      self.add_torch_node(raw_graph, fw_node)

    for ip in fw_graph.return_node().inputs():
      ret_value = raw_graph.get_blob_value_by_name(unique_name(ip))
      if ret_value.node.kind in ["TupleConstruct"]:
        for ip in ret_value.node.inputs:
          raw_graph.add_ret_value(ip)
        raw_graph.remove_node(ret_value.node)
      else:
        raw_graph.add_ret_value(ret_value)
    self._connect_nodes(raw_graph)

    return raw_graph

  def reconnect_nodes(self, raw_graph):
    for idx, node in enumerate(raw_graph.nodes):
      node.idx = idx
      node.clean_connection()
    self._connect_nodes(raw_graph)

  @staticmethod
  def _connect_nodes(raw_graph):
    for nodeA in raw_graph.nodes:
      for ip in nodeA.flatten_inputs:
        for nodeB in raw_graph.nodes:
          if nodeB is not nodeA and ip in nodeB.outputs:
            nodeB.add_out_node(nodeA)
            nodeA.add_in_node(ip.node)

  def add_torch_node(self, raw_graph, fw_node):
    inputs = OrderedDict()
    params = OrderedDict()
    for ip_name in (unique_name(ip) for ip in fw_node.inputs()):
      if ip_name in raw_graph.blobs_name():
        inputs[ip_name] = raw_graph.get_blob_value_by_name(ip_name)
      elif ip_name in raw_graph.param_names():
        params[ip_name] = raw_graph.get_param_value_by_name(ip_name)
      else:
        raise RuntimeError(f"{ip_name} not in raw_graph")

    if inputs:
      node = TorchNode(fw_node)
      for ip in fw_node.inputs():
        ip_value = inputs[unique_name(ip)] if unique_name(
            ip) in inputs else params[unique_name(ip)]
        node.add_input(ip_value)
      for op in fw_node.outputs():
        value = TorchValue(op)
        value.node = node
        node.add_output(value)
      raw_graph.add_node(node)

    elif params:
      if len(params) == 1:
        # %output_param = op(%param)
        raw_graph.add_param_alias(
            unique_name(list(fw_node.outputs())[0]),
            unique_name(list(fw_node.inputs())[0]))
      else:
        # %output_param = ListConstruct(%param1, %param2, ...)
        raw_graph.add_param_alias(
            unique_name(list(fw_node.outputs())[0]),
            [param.name for param in params.values()])

    else:
      const_value = TorchValue(list(fw_node.outputs())[0])
      if const_value.is_plain_value() or const_value.is_none():
        raw_graph.add_blob_value(const_value)
      else:
        const_node = TorchNode(fw_node)
        const_value.node = const_node
        const_node.add_output(const_value)
        raw_graph.add_node(const_node)
