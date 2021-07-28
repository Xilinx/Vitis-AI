

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

import torch

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph, Node, Tensor

from .torch_op_def import *
from .parse_utils import _GRAPH_SCOPE_SYM, get_full_name


class _NodeCreator(object):

  def __init__(self):
    self._idx = 0

  def __call__(self,
               graph,
               node_name,
               op,
               num_out_tensors,
               shape=None,
               in_tensors=None,
               in_quant_part=True):
    
    node_name = get_full_name(graph.name, node_name)
    node = Node(node_name, op=op, dtype="float32", idx=self._idx, in_quant_part=in_quant_part)
    # print(f"{node.name} quant state: {node.in_quant_part}")
    for i in range(num_out_tensors):
      tensor = Tensor(name=f"{node_name}_{i}", node=node, shape=shape)
      node.out_tensors.append(tensor)

    if in_tensors:
      for tensor in in_tensors:
        node.in_tensors.append(tensor)
    graph.add_node(node)
    self._idx += 1

class NodeTransformer(object):
  r""" tansform node to graph"""

  def __call__(self, node):
    return getattr(self, node.op.type, "default")(node)

  @staticmethod
  def _connect_nodes(graph):
    for nodeA in graph.nodes:
      for input_tensor in nodeA.in_tensors:
        for nodeB in graph.nodes:
          if nodeB is not nodeA and input_tensor in nodeB.out_tensors:
            #nodeB.outputs.add(input_tensor.node.name)
            nodeB.add_out_node(nodeA.name)
            nodeA.add_in_node(input_tensor.node.name)

  def basic_lstm(self, node):
    graph_scope_name = node.name.split(_GRAPH_SCOPE_SYM)[0]
    node_creator = _NodeCreator()
    graphs = []
    bidirectional = node.node_attr(node.op.AttrName.BIDIRECTIONAL)
    lstm_direction = ["forward"]
    if bidirectional:
      lstm_direction = ["forward", "backward"]
      
    for i in range(node.node_attr(node.op.AttrName.NUM_LAYERS)):
      lstm_cell_pair = {}
      if i == 0:
        input_size = node.node_attr(node.op.AttrName.INPUT_SIZE)
      else:
        input_size = len(lstm_direction) * node.node_attr(
            node.op.AttrName.HIDDEN_SIZE)

      hidden_size = node.node_attr(node.op.AttrName.HIDDEN_SIZE)
      bias=True
      for direction in lstm_direction:
        if direction == "forward":
          w_ih = node.op.params[node.op.ParamName.WEIGHT_IH][i]
          w_hh = node.op.params[node.op.ParamName.WEIGHT_HH][i]
          if node.op.ParamName.BIAS in node.op.params:
            bias_hi = node.op.params[node.op.ParamName.BIAS][i]
          else:
            bias=False
        else:
          w_ih = node.op.params[node.op.ParamName.WEIGHT_IH_REVERSE][i]
          w_hh = node.op.params[node.op.ParamName.WEIGHT_HH_REVERSE][i]
          if node.op.ParamName.BIAS_REVERSE in node.op.params:
            bias_hi = node.op.params[node.op.ParamName.BIAS_REVERSE][i]
          else:
            bias=False
      
        # lstm_node_name = node.name.replace("/", "_")
       
        graph_name = f"{graph_scope_name}_StandardLstmCell_layer_{i}_{direction}"
        graph = Graph(graph_name=graph_name)
        lstm_cell_pair[direction] = graph

        w_ii = Tensor(get_full_name(graph.name, "weight_ii"))
        w_if = Tensor(get_full_name(graph.name, "weight_if"))
        w_ig = Tensor(get_full_name(graph.name, "weight_ig"))
        w_io = Tensor(get_full_name(graph.name, "weight_io"))
        w_ii.from_ndarray(w_ih.data[:hidden_size])
        w_if.from_ndarray(w_ih.data[hidden_size:2 * hidden_size])
        w_ig.from_ndarray(w_ih.data[2 * hidden_size:3 * hidden_size])
        w_io.from_ndarray(w_ih.data[3 * hidden_size:4 * hidden_size])

        w_hi = Tensor(get_full_name(graph.name, "weight_hi"))
        w_hf = Tensor(get_full_name(graph.name, "weight_hf"))
        w_hg = Tensor(get_full_name(graph.name, "weight_hg"))
        w_ho = Tensor(get_full_name(graph.name, "weight_ho"))
        w_hi.from_ndarray(w_hh.data[:hidden_size])
        w_hf.from_ndarray(w_hh.data[hidden_size:2 * hidden_size])
        w_hg.from_ndarray(w_hh.data[2 * hidden_size:3 * hidden_size])
        w_ho.from_ndarray(w_hh.data[3 * hidden_size:4 * hidden_size])

        bias_i = Tensor(get_full_name(graph.name, "bias_i"))
        bias_f = Tensor(get_full_name(graph.name, "bias_f"))
        bias_g = Tensor(get_full_name(graph.name, "bias_g"))
        bias_o = Tensor(get_full_name(graph.name, "bias_o"))

        if bias is True:
          bias_i.from_ndarray(bias_hi.data[:hidden_size])
          bias_f.from_ndarray(bias_hi.data[hidden_size:2 * hidden_size])
          bias_g.from_ndarray(bias_hi.data[2 * hidden_size:3 * hidden_size])
          bias_o.from_ndarray(bias_hi.data[3 * hidden_size:4 * hidden_size])
       
        op = TorchBaseOperation(NNDCT_OP.INPUT, NNDCT_OP.INPUT)
        op.set_config("input", "args[0]")
        shape = [1, input_size]
        node_creator(
            graph=graph,
            node_name="input_0",
            op=op,
            num_out_tensors=1,
            shape=shape)
        op = TorchBaseOperation(NNDCT_OP.INPUT, NNDCT_OP.INPUT)
        op.set_config("input", "args[1]")
        shape = [1, hidden_size]
        node_creator(
            graph=graph,
            node_name="input_1",
            op=op,
            num_out_tensors=1,
            shape=shape)
        op = TorchBaseOperation(NNDCT_OP.INPUT, NNDCT_OP.INPUT)
        op.set_config("input", "args[2]")
        shape = [1, hidden_size]
        node_creator(
            graph=graph,
            node_name="input_2",
            op=op,
            num_out_tensors=1,
            shape=shape)
        # y_i = w_ii * input_0 + bias_i + w_hi * input_1 
        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("out_features", hidden_size)
        op.set_config("in_features", input_size)
        op.set_param(op.ParamName.WEIGHTS, w_ii)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_i)
        node_creator(
            graph=graph,
            node_name="w_ii * input_0 + bias_i",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_0")).out_tensors)

        op = TorchLinear()
        op.set_config("bias", False)
        op.set_config("out_features", hidden_size)
        op.set_config("in_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_hi)
        # op.set_param(op.ParamName.BIAS, bias_i)
        node_creator(
            graph=graph,
            node_name="w_hi * input_1",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_1")).out_tensors)

        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "w_ii * input_0 + bias_i")).out_tensors[0])
        op.set_config("other",
                      graph.node(get_full_name(graph.name, "w_hi * input_1")).out_tensors[0])
        node_creator(
            graph=graph,
            node_name="y_i",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "w_ii * input_0 + bias_i")).out_tensors[0],
                graph.node(get_full_name(graph.name, "w_hi * input_1")).out_tensors[0]
            ])
        # y_f = w_if * input_0 + bias_f + w_hf * input_1
        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("in_features", input_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_if)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_f)
        node_creator(
            graph=graph,
            node_name="w_if * input_0 + bias_f",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_0")).out_tensors)

        op = TorchLinear()
        op.set_config("bias", False)
        op.set_config("in_features", hidden_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_hf)
        # op.set_param(op.ParamName.BIAS, bias_f)
        node_creator(
            graph=graph,
            node_name="w_hf * input_1",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_1")).out_tensors)

        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "w_if * input_0 + bias_f")).out_tensors[0])
        op.set_config("other",
                      graph.node(get_full_name(graph.name, "w_hf * input_1")).out_tensors[0])
        node_creator(
            graph=graph,
            node_name="y_f",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "w_if * input_0 + bias_f")).out_tensors[0],
                graph.node(get_full_name(graph.name, "w_hf * input_1")).out_tensors[0]
            ])

        # y_g = w_ig * input_0 + bias_g + w_hg * input_1
        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("in_features", input_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_ig)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_g)
        node_creator(
            graph=graph,
            node_name="w_ig * input_0 + bias_g",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_0")).out_tensors)

        op = TorchLinear()
        op.set_config("bias", False)
        op.set_config("in_features", hidden_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_hg)
        # op.set_param(op.ParamName.BIAS, bias_g)
        node_creator(
            graph=graph,
            node_name="w_hg * input_1",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_1")).out_tensors)

        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "w_ig * input_0 + bias_g")).out_tensors[0])
        op.set_config("other",
                      graph.node(get_full_name(graph.name, "w_hg * input_1")).out_tensors[0])
        node_creator(
            graph=graph,
            node_name="y_g",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "w_ig * input_0 + bias_g")).out_tensors[0],
                graph.node(get_full_name(graph.name, "w_hg * input_1")).out_tensors[0]
            ])

        # y_o = w_io * input_0 +  bias_o + w_ho * input_1
        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("in_features", input_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_io)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_o)
        node_creator(
            graph=graph,
            node_name="w_io * input_0 + bias_o",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_0")).out_tensors)

        op = TorchLinear()
        op.set_config("bias", False)
        op.set_config("in_features", hidden_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_ho)
        # op.set_param(op.ParamName.BIAS, bias_o)
        node_creator(
            graph=graph,
            node_name="w_ho * input_1",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "input_1")).out_tensors)

        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "w_io * input_0 + bias_o")).out_tensors[0])
        op.set_config("other",
                      graph.node(get_full_name(graph.name, "w_ho * input_1")).out_tensors[0])

        node_creator(
            graph=graph,
            node_name="y_o",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "w_io * input_0 + bias_o")).out_tensors[0],
                graph.node(get_full_name(graph.name, "w_ho * input_1")).out_tensors[0]
            ])

        # op = Split(optype=NNDCT_OP.SPLIT)
        # op.set_attr(op.AttrName.INPUT, graph.node("combine_2_linearity").out_tensors[0])
        # op.set_attr(op.AttrName.SPLIT_SIZE_OR_SECTIONS, hidden_size)
        # op.set_attr(op.AttrName.AXIS, 1)
        # node_creator(graph=graph,
        #               node_name="split_ifgo",
        #               op=op,
        #               num_out_tensors=4,
        #               in_tensors=graph.node("combine_2_linearity").out_tensors)

        op = TorchSigmoid()
        node_creator(
            graph=graph,
            node_name="it",
            op=op,
            num_out_tensors=1,
            in_tensors=[graph.node(get_full_name(graph.name, "y_i")).out_tensors[0]])

        op = TorchSigmoid()
        node_creator(
            graph=graph,
            node_name="ft",
            op=op,
            num_out_tensors=1,
            in_tensors=[graph.node(get_full_name(graph.name, "y_f")).out_tensors[0]])

        op = TorchTanh()
        node_creator(
            graph=graph,
            node_name="cct",
            op=op,
            num_out_tensors=1,
            in_tensors=[graph.node(get_full_name(graph.name, "y_g")).out_tensors[0]])

        op = TorchSigmoid()
        node_creator(
            graph=graph,
            node_name="ot",
            op=op,
            num_out_tensors=1,
            in_tensors=[graph.node(get_full_name(graph.name, "y_o")).out_tensors[0]])

        op = TorchMul()
        op.set_config("input", graph.node(get_full_name(graph.name, "it")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "cct")).out_tensors[0])

        node_creator(
            graph=graph,
            node_name="it*cct",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "it")).out_tensors[0],
                graph.node(get_full_name(graph.name, "cct")).out_tensors[0]
            ])

        op = TorchMul()
        op.set_config("input", graph.node(get_full_name(graph.name, "ft")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "input_2")).out_tensors[0])
        node_creator(
            graph=graph,
            node_name="ft*input_2",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "ft")).out_tensors[0],
                graph.node(get_full_name(graph.name, "input_2")).out_tensors[0]
            ])

        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "it*cct")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "ft*input_2")).out_tensors[0])
        node_creator(
            graph=graph,
            node_name="c_next",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "it*cct")).out_tensors[0],
                graph.node(get_full_name(graph.name, "ft*input_2")).out_tensors[0]
            ])

        op = TorchTanh()
        node_creator(
            graph=graph,
            node_name="c_temp",
            op=op,
            num_out_tensors=1,
            in_tensors=graph.node(get_full_name(graph.name, "c_next")).out_tensors)

        op = TorchMul()
        op.set_config("input", graph.node(get_full_name(graph.name, "ot")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "c_temp")).out_tensors[0])
        node_creator(
            graph=graph,
            node_name="h_next",
            op=op,
            num_out_tensors=1,
            in_tensors=[
                graph.node(get_full_name(graph.name, "ot")).out_tensors[0],
                graph.node(get_full_name(graph.name, "c_temp")).out_tensors[0]
            ])
        self._connect_nodes(graph)
        graph.add_end_tensor(graph.node(get_full_name(graph.name, "h_next")).out_tensors[0])
        graph.add_end_tensor(graph.node(get_full_name(graph.name, "c_next")).out_tensors[0])
      graphs.append(lstm_cell_pair)
    return graphs
  
  def basic_gru(self, node):
    graph_scope_name = node.name.split(_GRAPH_SCOPE_SYM)[0]
    node_creator = _NodeCreator()
    graphs = []
    bidirectional = node.node_attr(node.op.AttrName.BIDIRECTIONAL)
    lstm_direction = ["forward"]
    if bidirectional:
      lstm_direction = ["forward", "backward"]
    
    for i in range(node.node_attr(node.op.AttrName.NUM_LAYERS)):
      lstm_cell_pair = {}
      
      if i == 0:
        input_size = node.node_attr(node.op.AttrName.INPUT_SIZE)
      else:
        input_size = len(lstm_direction) * node.node_attr(node.op.AttrName.HIDDEN_SIZE)
      
      hidden_size = node.node_attr(node.op.AttrName.HIDDEN_SIZE)
      
      bias = True
      for direction in lstm_direction:
        if direction  == "forward":
          w_ih = node.op.params[node.op.ParamName.WEIGHT_IH][i]
          w_hh = node.op.params[node.op.ParamName.WEIGHT_HH][i]
          if node.op.ParamName.BIAS_IH and node.op.ParamName.BIAS_HH in node.op.params:
            bias_ih = node.op.params[node.op.ParamName.BIAS_IH][i]
            bias_hh = node.op.params[node.op.ParamName.BIAS_HH][i]
          else:
            bias = False
        else:
          w_ih = node.op.params[node.op.ParamName.WEIGHT_IH_REVERSE][i]
          w_hh = node.op.params[node.op.ParamName.WEIGHT_HH_REVERSE][i] 
          if node.op.ParamName.BIAS_IH_REVERSE and node.op.ParamName.BIAS_HH_REVERSE in node.op.params:
            bias_ih = node.op.params[node.op.ParamName.BIAS_IH_REVERSE][i]
            bias_hh = node.op.params[node.op.ParamName.BIAS_HH_REVERSE][i]
          else:
            bias = False
        # lstm_node_name = node.name.replace("/", "_")
        graph_name = f"{graph_scope_name}_StandardGruCell_layer_{i}_{direction}"
        graph = Graph(graph_name = graph_name)
        lstm_cell_pair[direction] = graph
        
        w_ii = Tensor(f"weight_ii")
        w_if = Tensor(f"weight_if")
        w_ig = Tensor(f"weight_ig")
        w_ii.from_ndarray(w_ih.data[:hidden_size])
        w_if.from_ndarray(w_ih.data[hidden_size:2 * hidden_size])
        w_ig.from_ndarray(w_ih.data[2 * hidden_size:3 * hidden_size])
        
        w_hi = Tensor(f"weight_hi")
        w_hf = Tensor(f"weight_hf")
        w_hg = Tensor(f"weight_hg")
        w_hi.from_ndarray(w_hh.data[:hidden_size])
        w_hf.from_ndarray(w_hh.data[hidden_size:2 * hidden_size])
        w_hg.from_ndarray(w_hh.data[2 * hidden_size:3 * hidden_size])
        
        bias_ii = Tensor(f"bias_ii")
        bias_if = Tensor(f"bias_if")
        bias_ig = Tensor(f"bias_ig")
        bias_hi = Tensor(f"bias_hi")
        bias_hf = Tensor(f"bias_hf")
        bias_hg = Tensor(f"bias_hg")
        if bias is True:
          bias_ii.from_ndarray(bias_ih.data[:hidden_size])
          bias_if.from_ndarray(bias_ih.data[hidden_size:2 * hidden_size])
          bias_ig.from_ndarray(bias_ih.data[2 * hidden_size:3 * hidden_size])
          bias_hi.from_ndarray(bias_hh.data[: hidden_size])
          bias_hf.from_ndarray(bias_hh.data[hidden_size:2 * hidden_size])
          bias_hg.from_ndarray(bias_hh.data[2 * hidden_size:3 * hidden_size])
        
        op = TorchBaseOperation(NNDCT_OP.INPUT, NNDCT_OP.INPUT)
        op.set_config("input", "args[0]")
        shape = [1, input_size]
        node_creator(graph=graph, 
                      node_name="input_0", 
                      op=op, 
                      num_out_tensors=1, 
                      shape=shape)
        op = TorchBaseOperation(NNDCT_OP.INPUT, NNDCT_OP.INPUT)
        op.set_config("input", "args[1]")
        shape = [1, hidden_size]
        node_creator(graph=graph, 
                      node_name="input_1", 
                      op=op, 
                      num_out_tensors=1, 
                      shape=shape)
        # y_i = w_ii * input_0 +bias_ii + w_hi * input_1 + bias_hi
        
        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("out_features", hidden_size)
        op.set_config("in_features", input_size)
        op.set_param(op.ParamName.WEIGHTS, w_ii)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_ii)
        node_creator(graph=graph, 
                      node_name="w_ii * input_0 + bias_ii", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=graph.node(get_full_name(graph.name, "input_0")).out_tensors)

        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("out_features", hidden_size)
        op.set_config("in_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_hi)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_hi)
        node_creator(graph=graph, 
                      node_name="w_hi * input_1 + bias_hi", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=graph.node(get_full_name(graph.name, "input_1")).out_tensors)
        
        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "w_ii * input_0 + bias_ii")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "w_hi * input_1 + bias_hi")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="y_i", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "w_ii * input_0 + bias_ii")).out_tensors[0],
                                  graph.node(get_full_name(graph.name, "w_hi * input_1 + bias_hi")).out_tensors[0]])
        # y_f = w_if * input_0 + w_hf * input_1 + bias_f
        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("in_features", input_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_if)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_if)
        node_creator(graph=graph, 
                      node_name="w_if * input_0 + bias_if", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=graph.node(get_full_name(graph.name, "input_0")).out_tensors)

        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("in_features", hidden_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_hf)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_hf)
        node_creator(graph=graph, 
                      node_name="w_hf * input_1 + bias_hf", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=graph.node(get_full_name(graph.name, "input_1")).out_tensors)
        
        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "w_if * input_0 + bias_if")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "w_hf * input_1 + bias_hf")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="y_f", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "w_if * input_0 + bias_if")).out_tensors[0],
                                  graph.node(get_full_name(graph.name, "w_hf * input_1 + bias_hf")).out_tensors[0]])
        
        op = TorchSigmoid()
        node_creator(graph=graph, 
                      node_name="it", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "y_i")).out_tensors[0]])
        
        op = TorchSigmoid()
        node_creator(graph=graph, 
                      node_name="ft", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "y_f")).out_tensors[0]])
	
        # y_g = w_ig * input_0 + bias_ig + it*(w_hg * input_1 + bias_hg)
        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("in_features", input_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_ig)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_ig)
        node_creator(graph=graph, 
                      node_name="w_ig * input_0 + bias_ig", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=graph.node(get_full_name(graph.name, "input_0")).out_tensors)

        op = TorchLinear()
        op.set_config("bias", bias)
        op.set_config("in_features", hidden_size)
        op.set_config("out_features", hidden_size)
        op.set_param(op.ParamName.WEIGHTS, w_hg)
        if bias is True:
          op.set_param(op.ParamName.BIAS, bias_hg)
        node_creator(graph=graph, 
                      node_name="w_hg * input_1 + bias_hg", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=graph.node(get_full_name(graph.name, "input_1")).out_tensors)
	
        op = TorchMul()
        op.set_config("input", graph.node(get_full_name(graph.name, "it")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "w_hg * input_1 + bias_hg")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="it*(w_hg * input_1 + bias_hg)", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "it")).out_tensors[0],
                                graph.node(get_full_name(graph.name, "w_hg * input_1 + bias_hg")).out_tensors[0]])
		      
        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "w_ig * input_0 + bias_ig")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "it*(w_hg * input_1 + bias_hg)")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="y_g", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "w_ig * input_0 + bias_ig")).out_tensors[0],
                                  graph.node(get_full_name(graph.name, "it*(w_hg * input_1 + bias_hg)")).out_tensors[0]])
        
        # op = Split(optype=NNDCT_OP.SPLIT)
        # op.set_attr(op.AttrName.INPUT, graph.node("combine_2_linearity").out_tensors[0])
        # op.set_attr(op.AttrName.SPLIT_SIZE_OR_SECTIONS, hidden_size)
        # op.set_attr(op.AttrName.AXIS, 1)
        # node_creator(graph=graph, 
        #               node_name="split_ifgo", 
        #               op=op, 
        #               num_out_tensors=4, 
        #               in_tensors=graph.node("combine_2_linearity").out_tensors)
        
        
        op = TorchTanh()
        node_creator(graph=graph, 
                      node_name="cct", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "y_g")).out_tensors[0]])
       
        op = TorchMul()
        op.set_config("input", graph.node(get_full_name(graph.name, "ft")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "cct")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="ft*cct", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "ft")).out_tensors[0],
                                graph.node(get_full_name(graph.name, "cct")).out_tensors[0]])
        
        op = TorchSub()
        op.set_config("input", graph.node(get_full_name(graph.name, "cct")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "ft*cct")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="cct-ft*cct", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "cct")).out_tensors[0],
    			      graph.node(get_full_name(graph.name, "ft*cct")).out_tensors[0]])
        
        op = TorchMul()
        op.set_config("input", graph.node(get_full_name(graph.name, "ft")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "input_1")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="ft*input_1", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "ft")).out_tensors[0],
    			 graph.node(get_full_name(graph.name, "input_1")).out_tensors[0]])
        
        op = TorchAdd()
        op.set_config("input", graph.node(get_full_name(graph.name, "cct-ft*cct")).out_tensors[0])
        op.set_config("other", graph.node(get_full_name(graph.name, "ft*input_1")).out_tensors[0])
        node_creator(graph=graph, 
                      node_name="h_next", 
                      op=op, 
                      num_out_tensors=1, 
                      in_tensors=[graph.node(get_full_name(graph.name, "cct-ft*cct")).out_tensors[0],
                                  graph.node(get_full_name(graph.name, "ft*input_1")).out_tensors[0]])
		      
        self._connect_nodes(graph)
        graph.add_end_tensor(graph.node(get_full_name(graph.name, "h_next")).out_tensors[0])
      graphs.append(lstm_cell_pair)
    return graphs
