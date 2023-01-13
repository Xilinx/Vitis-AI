#
# Copyright 2019 Xilinx Inc.
# # Licensed under the Apache License, Version 2.0 (the "License");
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
from .torch_const import TorchGraphSymbol
from nndct_shared.utils import DeprecatedAPIError, NndctScreenLogger, NndctDebugLogger, NndctOption, GLOBAL_MAP, NNDCT_KEYS
_NODE_NAME_SEPERATOR = TorchGraphSymbol.NODE_NAME_SEPERATOR
_GRAPH_SCOPE_SYM = TorchGraphSymbol.GRAPH_SCOPE_SYM
_TENSOR = "TensorType"
_FLOAT_TYPE = "FloatType"
_INT_TYPE = "IntType"
_BOOL_TYPE = "BoolType"
_FLOAT = "Float"


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
      
def _optimize_graph_19(graph, is_jit_graph=False, module=None):
    from torch.onnx.utils import _split_tensor_list_constants
    # Inline everything
    torch._C._jit_pass_inline(graph)
    
    # Remove fork/wait nodes
    torch._C._jit_pass_inline_fork_wait(graph)
    torch._C._jit_pass_lint(graph)
    if not is_jit_graph:
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

    if is_jit_graph:
      torch._C._jit_pass_peephole(graph, True)
      # torch._C._jit_pass_lower_all_tuples(graph)
      torch._C._jit_pass_lint(graph)
      torch._C._jit_pass_onnx_remove_print(graph)
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
  pattern = re.compile(r'[0-9]+.[0-9]+.[0-9]+')
  version = re.findall(pattern, torch.__version__)
  version = ''.join(version[0].split('.'))
  return int(version)


def post_process_script_graph(graph):
  split_shared_const_tensor(graph)
  split_shared_bias(graph)
  split_scalar_const(graph)
  convert_scalar_to_const(graph)
  # remove_dce_node(graph)
  torch._C._jit_pass_dce(graph)
  torch._C._jit_pass_lint(graph)
  



def remove_ops(graph, kind, op_remove_handler, recurse=True):
  remove_nodes = _find_all_nodes(graph, kind)
  for node in remove_nodes:
    op_remove_handler(node)
    
  for node in remove_nodes:
    _node_destroy(node)
    
    
def _node_destroy(node):
  node.destroy()
  
def _find_all_nodes(graph, node_kind, recurse=True):
  return graph.findAllNodes(node_kind, recurse)


def _collect_unpack_pack_pair(graph, remove_nodes=None):
  if remove_nodes is None:
    remove_nodes = []
  for _, node in get_fw_op_nodes(graph):
    if node_type(node) in ["prim::TupleUnpack", "prim::ListUnpack"]:
      use_by_other = False
      for user in get_users(node):
        if user in remove_nodes:
          continue
        if node_type(user) in ["prim::TupleConstruct", "prim::ListConstruct"]:
          remove_nodes.append(user)
          for index, out in enumerate(node_outputs(user)):
            out.replaceAllUsesWith(node_inputs(node)[index])    
        else:
          use_by_other = True
      if not use_by_other:
        remove_nodes.append(node)
    elif has_block(node):
      for block in node_blocks(node):
        _collect_unpack_pack_pair(block, remove_nodes)
    
  return remove_nodes

  
def _collect_pack_unpack_pair(graph, remove_nodes=None):
  # TupleConstruct + TupleUnpack
  if remove_nodes is None:
    remove_nodes = []
    
  for _, node in get_fw_op_nodes(graph):
    if node_type(node) in ["prim::TupleConstruct", "prim::ListConstruct"]:
      use_by_other = False
      for user in get_users(node):
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
      for user in get_users(node):
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
  
def optimize_graph(graph, is_jit_graph=False, module=None):
  if get_torch_version() > 159 and get_torch_version() < 190:
    _optimize_graph_17(graph)
    return graph
  
  if get_torch_version() >= 190:
    _optimize_graph_19(graph, is_jit_graph, module)
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

def value_type(tensor: torch.Value):
  return tensor.type().kind()

def scalar_type(tensor: torch.Value):
  return  tensor.type().scalarType()

def get_node_output_type(node: torch.Node):
  assert node.outputsSize() == 1
  return value_type(node.output())

def get_node_output_scalar_type(node: torch.Node):
  assert node.outputsSize() == 1 and value_type(node.output()) == _TENSOR
  return scalar_type(node.output())

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
        users = get_users(current)

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


def get_users(node):
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

def get_node_output_at(node, index):
  return node.outputsAt(index)

def get_in_node_at(node, index):
  return node.inputsAt(index).node()
  
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


def find_input_value_by_name(graph, name):
  for inp in graph.inputs():
    if unique_name(inp) == name:
      return inp


def get_fw_graph_input_node(graph):
  return graph.param_node()
  
def get_fw_graph_ret_node(graph):
  if isinstance(graph, torch._C.Block):
    return graph.returnNode()
  else:
    return graph.return_node()

def get_fw_graph_ret_value(graph):
  if isinstance(graph, torch._C.Block):
    for ret in graph.returnNode().inputs():
      yield ret
  else:
    for ret in graph.return_node().inputs():
      yield ret

def should_construct_dynamic_list(list_construct_node):
    # if this list is element-accessed or modified at runtime, generate List ADT
    return node_type(list_construct_node) == "prim::ListConstruct" and len(list(list_construct_node.inputs())) == 0



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

def rename_graph_inputs(graph):
  for i, inp in enumerate(list(graph.inputs())[1:]):
    set_unique_name(inp, 'input_' + str(i))

def find_fw_node_by_name(g, name, recursive=False):
  return _find_fw_node_by_name([g], name, recursive)

def _find_fw_node_by_name(blocks, name, recursive):
  for block in blocks:
    for node in block.nodes():
      if has_block(node) and recursive:
        found_node = _find_fw_node_by_name(node_blocks(node), name, recursive)
        if found_node is not None:
          return found_node
      if name in get_node_outputs_name(node):
        return node

def nndct_name_2_jit_name(nndct_name):
  return nndct_name.split(_GRAPH_SCOPE_SYM)[-1].split(_NODE_NAME_SEPERATOR)[-1]

def create_fix_node(g, input, quant_min, quant_max, scale_inv, zero_point, method, device_id, inplace, name, tensor_type, index=None):
  if scale_inv is None:
    return 
  quant_min = g.insertConstant(int(quant_min))
  quant_max = g.insertConstant(int(quant_max))
  scale_inv = g.insertConstant(float(scale_inv))
  zero_point = g.insertConstant(zero_point)
  method = g.insertConstant(method)
  device_id = g.insertConstant(device_id)
  inplace = g.insertConstant(int(inplace))
  quant_min.node().moveAfter(input.node())
  quant_max.node().moveAfter((quant_min.node()))
  scale_inv.node().moveAfter((quant_max.node()))
  zero_point.node().moveAfter((scale_inv.node()))
  method.node().moveAfter((zero_point.node()))
  device_id.node().moveAfter((method.node()))
  inplace.node().moveAfter((device_id.node()))
  fix_node = g.create("vai::fix_neuron", [input, quant_min, quant_max, scale_inv, zero_point, method, device_id, inplace])
  output = get_node_output_at(fix_node, 0)
  name = name.replace('/', '_')
  if tensor_type == 'param':
    tensor_name = name + NNDCT_KEYS.FIX_OP_SUFFIX
  elif tensor_type == 'output':
    if index is None:
      tensor_name = name + NNDCT_KEYS.FIX_OP_SUFFIX
    else:
      tensor_name = name + NNDCT_KEYS.FIX_OP_SUFFIX + f"_i{index}"
  else:
    if index is None:
      tensor_name = name + NNDCT_KEYS.PRE_FIX_OP_SUFFIX
    else:
      tensor_name = name + NNDCT_KEYS.PRE_FIX_OP_SUFFIX + f"_i{index}"
  output.setDebugName(tensor_name)
  #output.setDebugName("_".join([input.debugName(), "fix"]))
  fix_node.insertAfter(inplace.node())
  return fix_node
                                     
def insert_after_node(g, fw_node, new_node, fw_value_name, idx=0):
  # new_node.insertAfter(fw_node)
  old_output = None
  for i, out_name in enumerate(get_node_outputs_name(fw_node)):
    if out_name == fw_value_name:
      old_output = get_node_output_at(fw_node, i)
  new_output = get_node_output_at(new_node, idx)
  old_output.replaceAllUsesAfterNodeWith(new_node, new_output)

def insert_before_node(g, fw_node, new_node, idx):
  # new_node.insertBefore(fw_node)
  output = get_node_output_at(fw_node, 0)
  new_output = get_node_output_at(new_node, 0)
  fw_node.replaceInput(idx, new_output)


def set_attr_value(fw_node, attr_name, attr_value):
    type_map = {
      torch.Tensor: "t_",
      int: "i_"
    }
    getattr(fw_node, type_map.get(type(attr_value)))(attr_name, attr_value)

def find_fw_nodes_by_type(g, type, filter=None):
  ret = []
  nodes = list(g.findAllNodes(type, recurse=True))
  if filter:
    for node in nodes:
      if filter(node):
        ret.append(node)
  else:
    ret = nodes
  return ret

def remove_fused_bn(g):
  bns = []
  def bn_filter(bn):
    pn_node = get_in_node_at(bn, 0)
    if node_type(pn_node) not in ["aten::_convolution"]:
      return False
    return True
  bns = find_fw_nodes_by_type(g, "aten::batch_norm", bn_filter)
  for node in bns:
    remove_node(node)
    
def remove_fused_ln_sigmoid(g):
  lns = []
  sns = []
  def ln_filter(ln):
    pn_node = get_in_node_at(ln, 0)
    if node_type(pn_node) not in ["aten::embedding"]:
      return False
    nn_node = get_users(ln)[0]
    if node_type(nn_node) not in ["aten::sigmoid"]:
      return False
    return True
  def sigmoid_filter(sn):
    pn_node = get_in_node_at(sn, 0)
    if node_type(pn_node) not in ["aten::layer_norm"]:
      return False
    nn_node = get_in_node_at(pn_node, 0)
    if node_type(nn_node) not in ["aten::embedding"]:
      return False
    return True
  lns = find_fw_nodes_by_type(g, "aten::layer_norm", ln_filter)
  sns = find_fw_nodes_by_type(g, "aten::sigmoid", sigmoid_filter)
  for node in lns:
    remove_node(node)
  for node in sns:
    remove_node(node)

def remove_dropout(g):
  dropouts = []
  dropouts += find_fw_nodes_by_type(g, "aten::dropout") 
  dropouts += find_fw_nodes_by_type(g, "aten::dropout_")
  dropouts += find_fw_nodes_by_type(g, "aten::feature_dropout")
  for n in dropouts:
    remove_node(n)

def remove_node(fw_node):
  assert len(list(fw_node.inputs())) > 0
  assert len(list(fw_node.outputs())) == 1
  out_val = fw_node.output()
  inp_val = fw_node.inputsAt(0)
  out_val.replaceAllUsesWith(inp_val)
  fw_node.destroy()

def remove_dce_node(g):
  torch._C._jit_pass_dce(g)
  # redundant_nodes = []
  # _remove_dce_node([g], redundant_nodes)
  # for node in redundant_nodes:
  #   node.destroy()

def _remove_dce_node(blocks, redundant_nodes):
  for block in blocks:
    for node in block.nodes():
      _remove_dce_node(node.blocks(), redundant_nodes)
      if not get_users(node):
        redundant_nodes.append(node)


def split_scalar_const(g):
  def _split_scalar_const(blocks):
    for block in blocks:
      for node in block.nodes():
        _split_scalar_const(node.blocks())
        if node_type(node) in ["aten::add", "aten::mul", "aten::div", "aten::sub"]:
          if node_type(get_in_node_at(node, 1)) == "prim::Constant" and get_node_output_type(get_in_node_at(node, 1)) in ["FloatType"]:
            const = get_attr_value(get_in_node_at(node, 1), "value")
            if all([node is use.user  for use  in node.inputsAt(1).uses()]):
              continue
            clone_const_value = g.insertConstant(const)
            clone_const_value.node().moveBefore(node)
            # set_attr_value(clone_const_value.node(), "value", const)
            node.replaceInput(1, clone_const_value)

  _split_scalar_const([g])



def split_shared_bias(g):
  def _split_shared_bias(blocks):
    for block in blocks:
      for node in block.nodes():     
        _split_shared_bias(node.blocks())
        if node_type(node) in ["aten::linear", "aten::_convolution"]:
          if node_type(get_in_node_at(node, 1)) != "prim::Constant" or node_type(get_in_node_at(node, 2)) != "prim::Constant":
              continue
          if unique_name(node.inputsAt(1)).split(".")[-1] != "weight":
            weight_prefix = ".".join(unique_name(node.inputsAt(1)).split(".")[:-2])
            bias_suffix = ".".join(["bias", unique_name(node.inputsAt(1)).split(".")[-1]])
          else:
            weight_prefix = ".".join(unique_name(node.inputsAt(1)).split(".")[:-1])
            bias_suffix = "bias"
          if unique_name(node.inputsAt(2)).split(".")[-1] != "bias":
            bias_prefix = ".".join(unique_name(node.inputsAt(2)).split(".")[:-2])
          else:
            bias_prefix = ".".join(unique_name(node.inputsAt(2)).split(".")[:-1])

          if weight_prefix != bias_prefix:
            weight_value = get_attr_value(get_in_node_at(node, 1), "value")
            is_transpose = bool(get_attr_value(get_in_node_at(node, 6), "value"))
            shape = weight_value.size(0) if is_transpose is False else  weight_value.size(1)
            bias_value = torch.zeros(shape, device=weight_value.device, requires_grad=weight_value.requires_grad)
            const_value = g.insertConstant(bias_value)
            const_value.node().moveBefore(node)
            set_attr_value(const_value.node(), "value", bias_value)
            set_unique_name(const_value, ".".join([weight_prefix, bias_suffix]))
            node.replaceInput(2, const_value)
    
  _split_shared_bias([g])  



def split_shared_const_tensor(g):
  def _split_shared_const_tensor(blocks):
    for block in blocks:
      for node in block.nodes():     
        _split_shared_const_tensor(node.blocks())
        if node_type(node) == "prim::Constant" and get_node_output_type(node) == _TENSOR and get_node_output_scalar_type(node) == _FLOAT:
          if len(get_users(node)) == 1:
            continue
          
          for id, use in enumerate(_get_uses(node)[1:], 1):
            value = get_attr_value(node, "value")
            new_value = torch.tensor(value.clone().detach(), device=value.device, requires_grad=value.requires_grad)
            new_const_value = g.insertConstant(new_value)
            new_const_value.node().moveBefore(use.user)
            set_attr_value(new_const_value.node(), "value", new_value)
            set_unique_name(new_const_value, "nndct_" + ".".join([unique_name(node.output()), str(id)]))
            use.user.replaceInput(use.offset, new_const_value)
           
  _split_shared_const_tensor([g])  



def create_mul_node(g, input, scale):
  scale = g.insertConstant(float(scale))
  scale.node().moveAfter(input.node())
  mul_node = g.create("aten::mul", [input, scale])
  mul_node.insertAfter(scale.node())
  return mul_node


const_type_converter = {
  _FLOAT_TYPE: torch.float32,
  _INT_TYPE: torch.int64,
  _BOOL_TYPE: torch.bool
}


def convert_scalar_to_const(g):
  def _convert_scalar_to_const(blocks):
    for block in blocks:
      for node in block.nodes():
        _convert_scalar_to_const(node.blocks())
        if node_type(node) in ["aten::add", "aten::mul", "aten::div", "aten::sub"]:
          if node_type(get_in_node_at(node, 1)) == "prim::Constant" and get_node_output_type(node) == "TensorType":
            const = get_attr_value(get_in_node_at(node, 1), "value")

            clone_const_tensor = g.insertConstant(torch.tensor(const, dtype=const_type_converter.get(get_node_output_type(get_in_node_at(node, 1)), torch.float32)))
            clone_const_tensor.node().moveBefore(node)
            node.replaceInput(1, clone_const_tensor)
    
  _convert_scalar_to_const([g])


def get_jit_value_device(value):
  if str(value.type()) == "Tensor" and hasattr(value.type(), "device"):
    return value.type().device()
  elif value.type().str():
    if "cpu" in value.type().str():
      device = "cpu"
    elif "cuda" in value.type().str():
      device = "cuda"
    else:
      device = "unknown"
    return device
  else:
    return "unknown"

def freeze_graph_wo_opt(module, preserved_attrs=None):
  from torch.jit._script import RecursiveScriptModule

  if not isinstance(module, torch.jit.ScriptModule):
        raise RuntimeError(
            "Freezing expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

  if module.training:
      raise RuntimeError(
          "Freezing is currently only implemented for modules in eval mode. "
          "Please call .eval() on your module before freezing."
      )

  preserved_attrs = preserved_attrs if preserved_attrs is not None else []

  out = RecursiveScriptModule(torch._C._freeze_module(module._c, preserved_attrs))
  RecursiveScriptModule._finalize_scriptmodule(out)
  return out

def get_operation_caller_by_schema_name(schema_name):
  caller_info = torch._C._jit_get_operation(schema_name)
  
  if isinstance(caller_info, list) or isinstance(caller_info, tuple):
    real_caller = []
    [real_caller.append(x) for x in  caller_info if 'builtin_function_or_method' in str(x.__class__)]
    if len(real_caller) == 0:
      return None
    if len(real_caller) > 1:
      NndctScreenLogger().warning(f"schema_name has more than one caller, but our program only use the first caller!!")
    return real_caller[0] 

  if isinstance(caller_info, object):
    return caller_info

  raise("unknow caller type")

def get_node_scope_name(fw_node):
  scope_name = fw_node.scopeName()
  if scope_name.startswith("__"):
    return scope_name.split("/")[-1]
  else:
    return scope_name

def remove_quant_dequant_stub(g):
  def pyop_filter(fw_node):
    if fw_node.pyname() in ["QuantStubF", "DeQuantStubF"]:
      return True
    else:
      return False
  stubs = find_fw_nodes_by_type(g, "prim::PythonOp", pyop_filter)
  for node in stubs:
    remove_node(node)
