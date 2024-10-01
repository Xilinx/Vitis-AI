
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
from collections import OrderedDict
from nndct_shared.nndct_graph import Graph, Block, Node, Tensor
from pytorch_nndct.fx.convert_op import create_op
from pytorch_nndct.fx.translator_utils import convert_dtype, get_meta_info, convert_shape
import nndct_shared.utils.tensor_util as tensor_util
from nndct_shared.base.key_names import FrameworkType


# def long_name(gm, node: torch.fx.Node):
#     name = short_name(gm, node)
#     target = node.target
#     if node.op == "call_function":
#         return torch_get_name(
#             node.target, f"{getattr(target, '__module__', '')}.{name}"
#         )
#     elif node.op in ["placeholder", "call_method"]:
#         return name
#     elif node.op == "call_module":
#         target = gm.get_submodule(target).__class__
#         return f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')}"
#     elif node.op == "get_attr":
#         return name
#     elif node.op == "output":
#         return "output"
#     raise AssertionError("unreachable")


def short_name(gm, node: torch.fx.Node):
    if node.op == "call_function":
        return node.target.__name__
    elif node.op in ["placeholder", "call_method"]:
        return node.target
    elif node.op == "call_module":
        return gm.get_submodule(node.target).__class__.__name__
    elif node.op == "get_attr":
        return node.target
    elif node.op == "output":
        return "output"
    raise AssertionError(node.op)


def make_tensor(name, meta):
  tensor = Tensor(
    name=name, 
    shape=convert_shape(get_meta_info(meta, "shape")),
    dtype=convert_dtype(get_meta_info(meta, "dtype")),
    # device=get_tensor_meta_info(input_meta, "device"),
    requires_grad=get_meta_info(meta, "requires_grad"))
  
  return tensor

def convert_op_param(op):
  for param_name, tensor in op.params.items():
      tensor = tensor_util.convert_parameter_tensor_format(
          tensor, FrameworkType.TORCH, FrameworkType.NNDCT)

class GraphTranslator(torch.fx.Interpreter):
  """
  convert fx graph to nndct graph
  """
  
  def __init__(self, gm: torch.fx.GraphModule, graph_id):
    super().__init__(gm, garbage_collect_values=True)
    self.env = OrderedDict()  # {node : output tensor of node}
    self._nndct_graph = Graph(graph_name=f"{gm.__class__.__name__}_{graph_id}")
    
  
  def run(self, *args):
    return super().run(*args)
  
  def run_node(self, n):
    args, kwargs = self.fetch_args_kwargs_from_env(n)
    out = self._run_node(n, args, kwargs)
    return out

  def _run_node(self, fx_node, args, kwargs):
    if fx_node.target == "output":
      dtype = get_meta_info(fx_node.args[0][0].meta["tensor_meta"], "dtype")
    else:
      dtype = get_meta_info(fx_node.meta["tensor_meta"], "dtype")
    node = Node(name="::".join([self._nndct_graph.name, fx_node.name]), dtype=convert_dtype(dtype))
    node.owning_graph = self._nndct_graph
    for arg in args:
      if isinstance(arg, Tensor):
        node.add_in_tensor(arg)

    for _, v in kwargs.items():
      if isinstance(v, Tensor):
        node.add_in_tensor(v)

    op = create_op(self.module, fx_node.op, fx_node.target, args, kwargs)
    convert_op_param(op)
    for _, v in op.params.items():
      node.add_in_tensor(v)
    
    node.op = op
    if fx_node.op != "output":
      tensor = make_tensor(fx_node.name, fx_node.meta["tensor_meta"])
      node.add_out_tensor(tensor)
      return tensor

  def build(self, *args):
    
    _ = self.run(*args)
    topo_sorted_nodes = sorted(self._nndct_graph._nodes_by_id.items(), key=lambda x: x[0])
    top_block = Block(self._nndct_graph, None, topo_sorted_nodes[0][1], topo_sorted_nodes[-1][1])
    self._nndct_graph.set_top_block(top_block)
    for _, node in topo_sorted_nodes[1:-1]:
      node.owning_block = top_block
      self._nndct_graph.append_node(node)

    self._nndct_graph.connect_nodes()

  @property
  def graph(self):
    return self._nndct_graph
    
    
    




