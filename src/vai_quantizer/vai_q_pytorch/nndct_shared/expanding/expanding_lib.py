# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Mapping

from nndct_shared.nndct_graph.base_graph import Graph
from nndct_shared.nndct_graph.base_node import Node
from nndct_shared.expanding.op_modifier import op_modifier
from nndct_shared.expanding.spec import DataInsert, GenericStructuredExpanding, StructuredExpanding
from nndct_shared.expanding.op_modifier import op_modifier


def propagate_node_expanding(node, node_expand_desc: Mapping[str, StructuredExpanding]):
  node_expanding = node_expand_desc[node.name]
  assert isinstance(node_expanding, GenericStructuredExpanding), \
    "Variable node_expanding here has to be instance of GenericStructuredExpanding"
  input_expanding = node_expand_desc[node.in_nodes[0]]
  node_expanding.in_dim = input_expanding.out_dim
  node_expanding.out_dim = node_expanding.in_dim
  for insert in input_expanding.out_inserts:
    node_expanding.add_insert(insert)


def update_node_by_expanding(graph: Graph, node: Node, node_expand_desc: Mapping[str, StructuredExpanding]):
  op_type = node.op.type
  if op_type in op_modifier:
    mod_func = op_modifier.lookup(op_type)
    mod_func(graph, node, node_expand_desc)
  else:
    propagate_node_expanding(node, node_expand_desc)
