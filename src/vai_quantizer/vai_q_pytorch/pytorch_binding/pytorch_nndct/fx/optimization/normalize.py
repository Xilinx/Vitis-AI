
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
import operator
import itertools
import torch
from pytorch_nndct.fx.optimization.utils import replace_node

LEAF_MODULES = {
  "Conv2d",
  "BatchNorm2d",
  "Linear"
}

NORMALIZE_METHOD_FUNCTION = {
  operator.add : torch.add,
  operator.iadd : torch.add
}



def always_true(*args, **kwargs):
  return True

class InliningTrace(torch.fx.Tracer):
  def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
    return False

def expand_module_call(prefix: str, graph: torch.fx.graph, module, args, kwargs):
  try:
    assert not kwargs
    arg_index = itertools.count()
    args_map = {}
    for node in InliningTrace().trace(module).nodes:
      if node.op == "placeholder":
        args_map[node] = args[next(arg_index)]
      elif node.op == "output":
        assert len(node.args) == 1
        return args_map[node.args[0]]
      elif node.op == "get_attr":
        args_map[node] = graph.get_attr(f"{prefix}{node.target}")
      else:
        args_map[node] = graph.node_copy(node, lambda n: args_map[n])
    raise AssertionError("unreachable")
  except Exception:
    print(f"Error while expanding {module.__class__.__name__}")
    raise
  

def normalize(gm: torch.fx.GraphModule):
  graph = gm.graph
  for node in graph.nodes:
    with graph.inserting_before(node):
      if node.op == "call_module":
        submod = gm.get_submodule(node.target)
        if submod.__class__.__name__ not in LEAF_MODULES:
          replace_node(graph, node, expand_module_call(f"{node.target}", graph, submod, node.args, node.kwargs))

      elif node.op in ["call_method", "call_function"]:
        if node.target in NORMALIZE_METHOD_FUNCTION:
          replace_node(graph, node, graph.call_function(NORMALIZE_METHOD_FUNCTION[node.target], node.args, node.kwargs))  

  graph.lint()
  gm.recompile()
  return gm