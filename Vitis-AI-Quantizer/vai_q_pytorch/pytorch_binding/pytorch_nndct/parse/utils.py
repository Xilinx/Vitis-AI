import contextlib
from torch.jit import _unique_state_dict
import torch.jit

from nndct_shared.utils import DeprecatedAPIError

_GRAPH_SCOPE_SYM = "::"
# IGNORE_STATEDICT_KEYS = ['num_batches_tracked']
scalar_type_to_pytorch_type = [
    'torch.uint8',  # 0
    'torch.int8',  # 1
    'torch.short',  # 2
    'torch.int',  # 3
    'torch.int64',  # 4
    'torch.half',  # 5
    'torch.float',  # 6
    'torch.double',  # 7
    'placehold',  # 8
    'torch.complex64',  # 9
    'torch.complex128',  # 10
    'torch.bool',  # 11
]


def get_full_name(graph_name: str, name: str) -> str:
  """get the full name of node/tensor in graph
 
  Args:
     graph_name (str): graph name
 
  Returns:
     str: full name
  """
 
  return _GRAPH_SCOPE_SYM.join([graph_name, name])


def get_short_name(full_name: str) -> str:
  """get the name of node/tensor in graph without graph name
  
  Args:
      full_name (str): full name of node/tensor
  Returns:
      str: short name
  """
  return full_name.split(_GRAPH_SCOPE_SYM)[-1]


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


def optimize_graph(graph):
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
  torch._C._jit_pass_fixup_onnx_loops(graph)
  torch._C._jit_pass_lint(graph)
  graph = torch._C._jit_pass_canonicalize(graph)
  torch._C._jit_pass_lint(graph)
  return graph


_TRACED_STACK_MODULES = []
_HOOKS = []


def _init_trace_state():
  torch.jit._trace_module_map = {}
  _TRACED_STACK_MODULES.clear()


def _tracing_name(module):
  if not _TRACED_STACK_MODULES:
      return None
  parent = _TRACED_STACK_MODULES[-1]
  for name, child in parent.named_children():
    if child is module:
      return name
  return None
      
      
def _set_trace_module_map(module):
  name = _tracing_name(module)
  if name:
    torch.jit._trace_module_map[module] = f"{module._get_name()}[{name}]"
  else:
    torch.jit._trace_module_map[module] = f"{module._get_name()}"


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
      old_map = torch.jit._trace_module_map 
      _init_trace_state()
      model.apply(_register_trace_fn)
      graph, torch_out = torch.jit._get_trace_graph(model, args)
      _remove_trace_fn()
      torch.jit._trace_module_map = old_map

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






