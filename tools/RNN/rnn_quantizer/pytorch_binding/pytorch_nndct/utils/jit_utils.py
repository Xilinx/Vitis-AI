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


import contextlib
import re
import itertools
import torch.jit
from torch.jit import _unique_state_dict
from torch.nn import ModuleList
from .schema import SchemaHelper, convert_type_str
from nndct_shared.utils import DeprecatedAPIError, NndctScreenLogger, NndctDebugLogger, NndctOption



@contextlib.contextmanager
def set_training(model, mode):
  r"""
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.
    """
  if mode is None:
    yield
    return
  old_mode = model.training
  if old_mode != mode:
    model.train(mode)
  try:
    yield
  finally:
    if old_mode != mode:
      model.train(old_mode)
      
def _optimize_graph_19(graph):
    from torch.onnx.utils import _split_tensor_list_constants
    # Inline everything
    torch._C._jit_pass_inline(graph)

    # Remove fork/wait nodes
    torch._C._jit_pass_inline_fork_wait(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_lower_all_tuples(graph)

    # we record now record some ops like ones/zeros
    # into a trace where we previously recorded constants
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
   
    torch._C._jit_pass_constant_propagation(graph)

    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_canonicalize_graph_fuser_ops(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_fuse_addmm(graph)
    torch._C._jit_pass_lint(graph)
  
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
  
    return graph
  
def _optimize_graph_17(graph):
  # Inline everything
  from torch.onnx.utils import _split_tensor_list_constants
  torch._C._jit_pass_inline(graph)

  # Remove fork/wait nodes
  torch._C._jit_pass_inline_fork_wait(graph)
  torch._C._jit_pass_lint(graph)
  torch._C._jit_pass_remove_inplace_ops(graph)

  # we record now record some ops like ones/zeros
  # into a trace where we previously recorded constants
  # use constant prop to maintain our current level of onnx support
  # without implementing symbolics for all of them
  torch._C._jit_pass_constant_propagation(graph)
  _split_tensor_list_constants(graph, graph)
  # run dce to eliminate dead parts of the graph that might have been
  # left behind by things like symbolic_override
  torch._C._jit_pass_dce(graph)
  torch._C._jit_pass_lint(graph)

  torch._C._jit_pass_canonicalize_graph_fuser_ops(graph)
  torch._C._jit_pass_lint(graph)

  torch._C._jit_pass_peephole(graph, True)
  torch._C._jit_pass_fuse_addmm(graph)
  torch._C._jit_pass_lint(graph)
  # graph is not a valid jit graph anymore because types have been replaced
  # (e.g. int with Tensor), so it now contains operators that don't actually
  # exist. We can't run normal dead code elimination because it'd fail trying
  # to look up if an operator has side effects, but we can run a dead code
  # elimination variant that doesn't need to look up if an op has side effects.
  torch._C._jit_pass_lint(graph)
  graph = torch._C._jit_pass_canonicalize(graph)
  torch._C._jit_pass_lint(graph)
  return graph


def get_torch_version():
  pattern = re.compile(r'[0-9].[0-9].[0-9]')
  version = re.findall(pattern, torch.__version__)
  version = ''.join(version[0].split('.'))
  return int(version)


def post_process_script_graph(graph):
  remove_redundant_ops(graph)


def remove_redundant_ops(graph):
  remove_nodes = []
  remove_nodes = _collect_tuple_index(graph, remove_nodes)
  remove_nodes = _collect_pack_unpack_pair(graph, remove_nodes)
  for node in remove_nodes:
    node.destroy()


def _collect_pack_unpack_pair(graph, remove_nodes=None):
  # TupleConstruct + TupleUnpack
  if remove_nodes is None:
    remove_nodes = []
    
  for _, node in get_fw_op_nodes(graph):
    if node_type(node) in ["prim::TupleConstruct", "prim::ListConstruct"]:
      use_by_other = False
      for user in _get_users(node):
        if node_type(user) in ["prim::TupleUnpack", "prim::ListUnpack"]:
          remove_nodes.append(user)
          for index, out in enumerate(node_outputs(user)):
            out.replaceAllUsesWith(node_inputs(node)[index])    
        else:
          use_by_other = True
      if not use_by_other:
        remove_nodes.append(node)
    elif has_block(node):
      for block in node_blocks(node):
        _collect_pack_unpack_pair(block, remove_nodes)
    
  return remove_nodes


def _collect_tuple_index(graph, remove_nodes=None):
  #  TupleConstruct + TupleIndex
  if remove_nodes is None:
    remove_nodes = []
    
  for _, node in get_fw_op_nodes(graph):
    if node_type(node) == "prim::TupleConstruct":
      use_by_other = False
      for user in _get_users(node):
        if node_type(user) == "prim::TupleIndex":
          index = node_inputs(user)[-1]
          if node_type(index.node()) == "prim::Constant":
            index_value = get_attr_value(index.node(), "value")
            node_outputs(user)[0].replaceAllUsesWith(node_inputs(node)[index_value])
            remove_nodes.append(user)
        else:
          use_by_other = True
      if not use_by_other:
        remove_nodes.append(node)
    elif has_block(node):
      for block in node_blocks(node):
        _collect_tuple_index(block, remove_nodes)
    
  return remove_nodes
  
def optimize_graph(graph, is_jit_graph=False):
  if get_torch_version() > 159 and get_torch_version() < 190:
    _optimize_graph_17(graph)
    return graph
  
  if get_torch_version() >= 190:
    _optimize_graph_19(graph)
    return graph
  
  if is_jit_graph:
    torch._C._jit_pass_inline(graph)
  # Remove fork/wait nodes
  torch._C._jit_pass_inline_fork_wait(graph)
  torch._C._jit_pass_dce(graph)
  torch._C._jit_pass_lint(graph)

  torch._C._jit_pass_remove_inplace_ops(graph)
  # we record now record some ops like ones/zeros
  # into a trace where we previously recorded constants
  # use constant prop to maintain our current level of onnx support
  # without implementing symbolics for all of them
  torch._C._jit_pass_constant_propagation(graph)
  # _split_tensor_list_constants(graph, graph)
  # run dce to eliminate dead parts of the graph that might have been
  # left behind by things like symbolic_override
  torch._C._jit_pass_dce(graph)
  torch._C._jit_pass_lint(graph)

  # torch._C._jit_pass_canonicalize_ops(graph)
  # torch._C._jit_pass_lint(graph)

  torch._C._jit_pass_peephole(graph, True)
  torch._C._jit_pass_lint(graph)

  torch._C._jit_pass_dce(graph)
  torch._C._jit_pass_lint(graph)
  if get_torch_version() < 150:
    torch._C._jit_pass_fixup_onnx_loops(graph)
  torch._C._jit_pass_lint(graph)
  graph = torch._C._jit_pass_canonicalize(graph)
  torch._C._jit_pass_lint(graph)
  return graph


_TRACED_STACK_MODULES = []
_HOOKS = []

def _get_trace_graph():
  if hasattr(torch.jit, "_get_trace_graph"):
    return torch.jit._get_trace_graph
  else:
    return torch.jit._trace._get_trace_graph

def _get_trace_map():
  if hasattr(torch.jit, "_trace_module_map"):
    return torch.jit._trace_module_map
  else:
    return torch.jit._trace._trace_module_map

def _init_trace_map(init_map):
  if hasattr(torch.jit, "_trace_module_map"):
    torch.jit._trace_module_map = init_map
  else:
    torch.jit._trace._trace_module_map = init_map

def _init_trace_state():
  _init_trace_map({})
  _TRACED_STACK_MODULES.clear()

def _tracing_name(module):
  if not _TRACED_STACK_MODULES:
      return None

  def _extract_child_recursively(module, prefix='', name_to_child=None):
    if name_to_child is None:
      name_to_child = {}

    name_to_child[prefix] = module
    # If the child is a ModuleList,
    # we need to use the child in the list as the actual child.
    if isinstance(module, ModuleList):
      for index, child in enumerate(module):
        # Add missing hierarchy to the name so that
        # we can fetch the corresponding module by the name.
        # Note that the odd child name is meant to match the
        # string concatenatation in _set_trace_module_map.
        if prefix:
          child_name = f"{prefix}]/ModuleList[{index}"
        else:
          child_name = f"ModuleList[{index}"
        _extract_child_recursively(child, child_name, name_to_child)
    else:
      for name, child in module.named_children():
        if prefix:
          child_name = f"{prefix}]/{child._get_name()}[{name}"
        else:
          child_name = name
        _extract_child_recursively(child, child_name, name_to_child)
    return name_to_child

  parent = _TRACED_STACK_MODULES[-1]
  name_to_child = _extract_child_recursively(parent)

  for name, child in name_to_child.items():
    if child is module:
      return name
  return None

def _set_trace_module_map(module):
  name = _tracing_name(module)
  if name:
    _get_trace_map()[module] = f"{module._get_name()}[{name}]"
  else:
    _get_trace_map()[module] = f"{module._get_name()}"


def _traced_module_pre(module, inputs):
  _set_trace_module_map(module)
  _TRACED_STACK_MODULES.append(module)


def _traced_module_post(module, inputs, outputs):
  _TRACED_STACK_MODULES.pop()


def _register_trace_fn(module):
  if _traced_module_pre not in module._forward_pre_hooks.values():
    _HOOKS.append(module.register_forward_pre_hook(_traced_module_pre))

  if _traced_module_post not in module._forward_hooks.values():
    _HOOKS.append(module.register_forward_hook(_traced_module_post))


def _remove_trace_fn():
  for handle in _HOOKS:
    handle.remove()


def trace_and_get_graph_from_model(model, args, training):

  orig_state_dict_keys = _unique_state_dict(model).keys()

  # By default, training=False, which is good because running a model in
  # training mode could result in internal buffers getting updated, dropout
  # getting applied, etc.  If you really know what you're doing, you
  # can turn training=True (or None, to preserve whatever the original
  # training mode was.)
  with set_training(model, training):
    if hasattr(torch.jit, "get_trace_graph"):
      trace, torch_out = torch.jit.get_trace_graph(model, args)
      graph = trace.graph()
    else:
      old_map = _get_trace_map()
      _init_trace_state()
      model.apply(_register_trace_fn)
      graph, torch_out = _get_trace_graph()(model, args)
      _remove_trace_fn()
      _init_trace_map(old_map)
      # torch.jit._trace_module_map = old_map

    if orig_state_dict_keys != _unique_state_dict(model).keys():
      raise RuntimeError("state_dict changed after running the tracer; "
                         "something weird is happening in your model!")

    return graph, torch_out


def rename_graph_param_name(model, graph):
  state_dict = _unique_state_dict(model)
  graph_inputs = list(graph.inputs())
  user_input_num = len(graph_inputs) - len(state_dict)
  param_names = list(state_dict.keys())
  params = []
  for i, inp in enumerate(graph_inputs):
    if i >= user_input_num:
      set_unique_name(inp, param_names[i - user_input_num])
      params.append(unique_name(inp))
    else:
      set_unique_name(inp, 'input_' + str(i))
  return params


def set_unique_name(value: torch.Value, name: str):
  if hasattr(value, 'setUniqueName'):  # torch1.1
    value.setUniqueName(name)
  elif hasattr(value, 'setDebugName'):  # torch1.2
    value.setDebugName(name)
  else:
    raise DeprecatedAPIError('setDebugName', value.__class__.__name__)


def unique_name(value: torch.Value):
  if hasattr(value, 'uniqueName'):  # torch1.1
    return value.uniqueName()
  elif hasattr(value, 'debugName'):  # torch1.2
    return value.debugName()
  else:
    raise DeprecatedAPIError('debugName', value.__class__.__name__)


def get_attr_value(node: torch.Node, attr_name: str):
  sel = node.kindOf(attr_name)
  if sel in ['ival']:
    return node.output().toIValue()
  else:
    return getattr(node, sel)(attr_name)


def get_node_output_name(node: torch.Node):
  assert node.outputsSize() == 1
  return unique_name(node.output())


def get_node_outputs_name(node: torch.Node):
  return [unique_name(o) for o in node.outputs()]


def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """

    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = _get_users(current)

        if not users or terminate(users):
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """

    def terminate(users):
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]
        return len(next_attrs) == 0

    return get_use_chains(root_getattr_node, terminate)


def _get_users(node):
    return [use.user for use in _get_uses(node)]


def _get_uses(node):
    uses = []
    for output in node.outputs():
        uses += output.uses()
    return uses

def getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert len(attribute_names) == 1
    attr_name = node.s(attribute_names[0])
    return attr_name


def getattr_full_name(getattrs):
    return ".".join([getattr_attr_name(node) for node in getattrs])


def get_fw_op_nodes(graph):
  ops = []
  for node in graph.nodes():
    if node.outputsSize() == 0:
      continue
    if node.outputsSize() > 1:
      node_name = "_".join(get_node_outputs_name(node))
    else:
      node_name = get_node_output_name(node)
    
    if node.kind() != "prim::GetAttr":
      ops.append((node_name, node))
  
  return ops



def node_type(node):
  return node.kind()

def node_inputs(node):
  return [inp for inp in node.inputs()]


def node_outputs(node):
  return [out for out in node.outputs()]


def node_blocks(node):
  return [block for block in node.blocks()]

def has_block(node):
  return False if not list(node.blocks()) else True



def get_fw_graph_inputs(graph):
  for inp in graph.inputs():
    if unique_name(inp) == "self":
      continue
    else:
      yield inp

def get_fw_graph_ret_value(graph):
  if isinstance(graph, torch._C.Block):
    for ret in graph.returnNode().inputs():
      yield ret
  else:
    for ret in graph.return_node().inputs():
      yield ret

def should_construct_dynamic_list(list_construct_node):
    # if this list is element-accessed or modified at runtime, generate List ADT
    return len(list(list_construct_node.inputs())) == 0



def find_builtin(fn):
  if get_torch_version() < 150:
    return torch.jit._find_builtin(fn)
  else:
    return torch.jit._builtins._find_builtin(fn)

def modules_containing_builtins():
  if get_torch_version() < 150:
    return torch.jit._modules_containing_builtins
  else:
    return torch.jit._builtins._modules_containing_builtins

def builtin_ops():
  if get_torch_version() <= 120:
    import math
    import torch.backends.cudnn as cudnn
    import warnings
    from torch.nn.modules.utils import _single, _pair, _triple, _quadruple, _list_with_default
    _wait = torch._C.wait
    _unwrap_optional = torch.jit._unwrap_optional

    builtin_ops = [
        # Pairs of (function, op_name)
        (_list_with_default, "aten::list_with_default"),
        (_pair, "aten::_pair"),
        (_quadruple, "aten::_quadruple"),
        (_single, "aten::_single"),
        (_triple, "aten::_triple"),
        (_unwrap_optional, "aten::_unwrap_optional"),
        (_wait, 'aten::wait'),
        (cudnn.is_acceptable, "aten::cudnn_is_acceptable"),
        (math.ceil, "aten::ceil"),
        (math.copysign, "aten::copysign"),
        (math.erf, "aten::erf"),
        (math.erfc, "aten::erfc"),
        (math.exp, "aten::exp"),
        (math.expm1, "aten::expm1"),
        (math.fabs, "aten::fabs"),
        (math.floor, "aten::floor"),
        (math.gamma, "aten::gamma"),
        (math.lgamma, "aten::lgamma"),
        (math.log, "aten::log"),
        (math.log10, "aten::log10"),
        (math.log1p, "aten::log1p"),
        (math.pow, "aten::pow"),
        (math.sqrt, "aten::sqrt"),
        (math.isnan, "aten::isnan"),
        (math.asinh, "aten::asinh"),
        (math.atanh, "aten::atanh"),
        (math.cosh, "aten::cosh"),
        (math.sinh, "aten::sinh"),
        (math.tanh, "aten::tanh"),
        (math.acos, "aten::acos"),
        (math.asin, "aten::asin"),
        (math.atan, "aten::atan"),
        (math.atan2, "aten::atan2"),
        (math.cos, "aten::cos"),
        (math.sin, "aten::sin"),
        (math.tan, "aten::tan"),
        (math.asinh, "aten::asinh"),
        (math.atanh, "aten::atanh"),
        (math.acosh, "aten::acosh"),
        (math.sinh, "aten::sinh"),
        (math.cosh, "aten::cosh"),
        (math.tanh, "aten::tanh"),
        (math.fmod, "aten::fmod"),
        (math.modf, "aten::modf"),
        (math.factorial, "aten::factorial"),
        (math.frexp, "aten::frexp"),
        (math.isnan, "aten::isnan"),
        (math.isinf, "aten::isinf"),
        (math.degrees, "aten::degrees"),
        (math.radians, "aten::radians"),
        (math.ldexp, "aten::ldexp"),
        (torch._C._infer_size, "aten::_infer_size"),
        (torch.nn.functional._no_grad_embedding_renorm_, "aten::_no_grad_embedding_renorm_"),
        (torch.nn.functional.assert_int_or_pair, "aten::_assert_int_or_pair"),
        (torch.nn.functional.interpolate, "aten::__interpolate"),
        (torch.nn.functional.upsample_bilinear, "aten::__upsample_bilinear"),
        (torch.nn.functional.upsample_nearest, "aten::__upsample_nearest"),
        (torch.nn.functional.upsample, "aten::__upsample"),
        (torch.nn.init._no_grad_fill_, "aten::_no_grad_fill_"),
        (torch.nn.init._no_grad_normal_, "aten::_no_grad_normal_"),
        (torch.nn.init._no_grad_uniform_, "aten::_no_grad_uniform_"),
        (torch.nn.init._no_grad_zero_, "aten::_no_grad_zero_"),
        (torch._C._get_tracing_state, "aten::_get_tracing_state"),
        (warnings.warn, "aten::warn"),
    ]
    return builtin_ops
  if  get_torch_version() < 150 and get_torch_version() > 120:
    return torch.jit._builtin_ops
  else:
    return torch.jit._builtins._builtin_ops


def parse_node_signature(node):
  node_type = node.kind()
  input_args_type = []
  out_args_type = []
  for inp in node.inputs():
    input_args_type.append(str(inp.type()))

  for out in node.outputs():
    out_args_type.append(str(out.type()))

  input_str = ", ".join(input_args_type)
  output_str = ", ".join(out_args_type)
  return node_type + "::" + "(" + input_str + ")" + " -> " + output_str

def get_node_schema(node):
  schema_op = node_type(node)
  schemas = torch._C._jit_get_schemas_for_operator(schema_op)
  for schema in schemas:
    if is_schema_matching(node, schema):
      if NndctOption.nndct_parse_debug.value >= 1:
        NndctDebugLogger.write(f"%{get_node_outputs_name(node)[0]} signature: {parse_node_signature(node)}\n")
        schema_handler = SchemaHelper(schema)
        NndctDebugLogger.write(f"matched schema: {schema_handler.toString()}\n")
      return schema
  if schema_op.split("::")[0] == "aten":
    #assert False
    NndctScreenLogger().warning(f"Can't find schema for {node}.If you can get quantizable model successfully, please ignore it.\n")


def is_schema_matching(node, schema):
  if len(list(node.inputs())) != len(schema.arguments):
    return False

  if len(list(node.outputs())) != len(schema.returns):
    return False

  schema_handler = SchemaHelper(schema)
  for inp, arg in zip(node.inputs(), schema.arguments):
    inp_type = str(inp.type())
    arg_type = schema_handler.arg_type(arg)
    if inp_type in ["None", "NoneType"] and "Optional" in arg_type:
      continue

    inp_type = inp_type.replace("int", "number")
    inp_type = inp_type.replace("float", "number")

    arg_type = arg_type.replace("int", "number")
    arg_type = arg_type.replace("float", "number")
    if convert_type_str(inp_type).replace("?", "") not in convert_type_str(arg_type).replace("?", ""):
      return False

  return True



