from collections import OrderedDict

from nndct_shared.utils import NndctDebugLogger, NndctOption

from .torch_graph import *
from .utils import *


class TorchGraphHandler(object):

  def __init__(self):
    self._transparent_ops = ["ListUnpack"]

  def build_torch_graph(self, graph_name, module, input_args, train=False):
    self._module = module
    fw_graph, params = self._trace_graph_from_model(input_args, train)

    self._node_kinds = {node.kind().split(":")[-1] for node in fw_graph.nodes()}
    if NndctOption.nndct_parse_debug.value >= 1:
      NndctDebugLogger.write(f"torch raw graph:\n{fw_graph}")
      NndctDebugLogger.write(f"\nparsing nodes types:\n{self._node_kinds}")

    raw_graph = self._build_raw_graph(graph_name, fw_graph, params)
    self._opt_raw_graph(raw_graph)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch opt graph:\n{raw_graph}")
    return raw_graph

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
 

  def _clean_transparent_ops(self, raw_graph):

    def find_transparent_nodes(root, remove_nodes, has_processed):

      def dfs(node):
        if len(node.out_nodes) == 0:
          return
        for c_node in node.out_nodes:
          if c_node not in has_processed:
            has_processed.append(c_node)
            dfs(c_node)
            if c_node.kind in self._transparent_ops:
              remove_nodes.add(c_node)
              node.outputs = c_node.outputs

      dfs(root)

    remove_nodes = set()
    has_processed = []
    find_transparent_nodes(raw_graph.nodes[0], remove_nodes, has_processed)

    for r_node in list(remove_nodes):
      raw_graph.nodes.remove(r_node)

    self._reconnect_nodes(raw_graph)

  def _pack_ListConstruct_op(self, raw_graph):
    ListConstruct_nodes = set()
    for node in raw_graph.nodes:
      if node.kind in ["ListConstruct", "TupleConstruct"]:
        continue

      id2list = {}
      for i, ip in enumerate(node.inputs):
        if isinstance(ip, list):
          continue
        if ip.node and ip.node.kind in ["ListConstruct", "TupleConstruct"]:
          id2list[i] = ip.node.inputs
          ListConstruct_nodes.add(ip.node)
       

      for i, lst in id2list.items():
        node.inputs[i] = lst

    for node in ListConstruct_nodes:
      raw_graph.nodes.remove(node)

    self._reconnect_nodes(raw_graph)

  def _to_strided_slice(self, raw_graph):
    strided_slice_nodes = {}
    slice_nodes = []
    for node in raw_graph.nodes:
      if node.kind in ["slice"]:
        slice_nodes.append(node)
        if node.inputs[0].node.kind != "strided_slice":
          strided_node = TorchNode()
          strided_node.idx = node.idx
          strided_node.kind = "strided_slice"
          strided_node.name = node.name
          strided_node.add_input(node.inputs[0])
          strided_node.dtype = node.dtype
          for ip in node.inputs[1:]:
            strided_node.add_input([ip])
          strided_node.add_output(node.outputs[0])
          strided_node.outputs
          strided_slice_nodes[node.name] = strided_node
        else:
          node.name = node.inputs[0].node.name
          strided_node = strided_slice_nodes[node.inputs[0].node.name]
          for i, ip in enumerate(node.inputs[1:], 1):
            strided_node.inputs[i].append(ip)
          strided_node.outputs[0] = node.outputs[0]
          strided_node.outputs[0].node = strided_node
      else:
        continue

    for node in strided_slice_nodes.values():
      raw_graph.nodes[node.idx] = node

    for node in slice_nodes:
      if node in raw_graph.nodes:
        raw_graph.nodes.remove(node)

    self._reconnect_nodes(raw_graph)

  def _to_slice_inplace_copy(self, raw_graph):
    select_nodes = []
    for node in raw_graph.nodes:
      if node.kind == "copy_":
        if node.inputs[0].node.kind == "select":
          select_nodes.append(node.inputs[0].node)
          node.kind = "slice_tensor_inplace_copy"
          inputs = node.inputs[0].node.inputs[1:]
          node.inputs[0] = node.inputs[0].node.inputs[0]
          for ip in inputs:
            node.add_input(ip)

    for node in select_nodes:
      raw_graph.nodes.remove(node)

    self._reconnect_nodes(raw_graph)

  def _opt_raw_graph(self, raw_graph):
    if set(self._transparent_ops).issubset(self._node_kinds):
      self._clean_transparent_ops(raw_graph)

    if "slice" in self._node_kinds:
      self._to_strided_slice(raw_graph)

    # replace select op  and inplace_copy op with a slice_tensor_copy op  e.g. a[:,:,0] = b
    if {"select", "copy_"}.issubset(self._node_kinds):
      self._to_slice_inplace_copy(raw_graph)

    if "ListConstruct" in self._node_kinds or "TupleConstruct" in self._node_kinds:
      self._pack_ListConstruct_op(raw_graph)

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
        raw_graph.nodes.remove(ret_value.node)
      else:
        raw_graph.add_ret_value(ret_value)
    self._connect_nodes(raw_graph)

    return raw_graph

  def _reconnect_nodes(self, raw_graph):
    for idx, node in enumerate(raw_graph.nodes):
      node.idx = idx
      node.clean_connection()
    self._connect_nodes(raw_graph)

  def _connect_nodes(self, raw_graph):
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
