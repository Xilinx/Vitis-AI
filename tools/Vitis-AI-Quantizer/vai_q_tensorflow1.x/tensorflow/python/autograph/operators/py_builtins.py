# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Operators corresponding to Python builtin functions.

List of built-in functions: https://docs.python.org/3/library/functions.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import six

from tensorflow.python.autograph.utils import py_func
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops


UNSPECIFIED = object()


def overload_of(f):
  if f in SUPPORTED_BUILTINS:
    return BUILTIN_FUINCTIONS_MAP[f.__name__]
  return f


def _find_originating_frame(caller_fn_scope, innermost=True):
  """Locates the frame in which `caller_fn_scope` was defined."""
  ctx_frame = inspect.currentframe()
  result = None
  while ctx_frame is not None:
    # Note it should not be normally possible to get false positives this way
    # because the function scope object is not accessible to user code (barring
    # call stack introspection).
    if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
      result = ctx_frame
      if innermost:
        break
    ctx_frame = ctx_frame.f_back

  assert result is not None, (
      'the conversion process should ensure the caller_fn_scope is always'
      ' found somewhere on the call stack')

  return result


def eval_in_original_context(f, args, caller_fn_scope):
  """Executes the eval function in the context of a specified function."""
  # When control flow is rewritten using functions, eval should use the
  # variables found in the same block where it was called. That is equivalent
  # to the innermost function call.
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)

  args = (
      args[0],
      ctx_frame.f_globals if len(args) < 2 else args[1],
      ctx_frame.f_locals if len(args) < 3 else args[2],
  )
  return f(*args)


def super_in_original_context(f, args, caller_fn_scope):
  """Executes the super function in the context of a specified function.

  See https://docs.python.org/3/library/functions.html#super for the exact
  details

  Args:
    f: Callable, typically the super builtin
    args: List[Any], the original call arguments
    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
      scope of the converted function in which this call was originally made

  Returns:
    The result of calling `f` as if it was called in the frame indicated by
      `caller_fn_scope`.
  """

  # Python 2 doesn't support implicit argument super variants.
  if six.PY2:
    return f(*args)

  # Only the no-arg call is desugared.
  if args:
    return f(*args)

  # Inner functions seem to include their closure in f_locals, so we need
  # to find the outermost frame.
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)

  # When super(..) is called without arguments, it looks for __class__ cell
  # variable and the first argument passed in the enclosing function according
  # to the spec https://www.python.org/dev/peps/pep-3135/ .
  #
  # We couldn't verify if `inspect.currentframe().f_code.co_varnames[0]` is
  # guaranteed to be the first argument from an official doc or PEP, however,
  # it's fairly stable and well established:
  # - An unofficial community doc mentions it.
  #   https://python-reference.readthedocs.io/en/latest/docs/code/varnames.html
  # - CPython has tests checking that order, which was merged in 2008, and
  #   unchanged since then.
  #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py2_test_grammar.py#L157
  #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py3_test_grammar.py#L192
  #
  # Note: the name can be more reliably obtained by inspecting the calling
  # function's argspec.
  #
  # Even though methods can be declared using *args (def method(*args)),
  # that pattern is disallowed by super() -- it raises super() no arguments.
  # Method definitions using **kwargs are not allowed at all.
  # In other words, we can always assume that self is on the first positional
  # argument (for correct code).
  #
  # TODO(mdan): Consider additional checks in case the input code is incorrect.
  # For example, the error might be cryptic compared to what super() regularly
  # raises.

  type_arg = ctx_frame.f_locals['__class__']
  self_arg_name = ctx_frame.f_code.co_varnames[0]
  self_arg = ctx_frame.f_locals[self_arg_name]
  return f(type_arg, self_arg)


def abs_(x):
  if tensor_util.is_tensor(x):
    return _tf_abs(x)
  return _py_abs(x)


def _tf_abs(x):
  return math_ops.abs(x)


def _py_abs(x):
  return abs(x)


def float_(x=0):
  if tensor_util.is_tensor(x):
    return _tf_float(x)
  return _py_float(x)


def _tf_float(x):
  # TODO(mdan): We shouldn't assume float32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.float32)
  return math_ops.cast(x, dtype=dtypes.float32)


def _py_float(x):
  return float(x)


def int_(x=0, base=UNSPECIFIED):
  if tensor_util.is_tensor(x):
    return _tf_int(x, base)
  return _py_int(x, base)


def _tf_int(x, base):
  if base not in (10, UNSPECIFIED):
    raise NotImplementedError('base {} not supported for int'.format(base))

  # TODO(mdan): We shouldn't assume int32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.int32)
  return math_ops.cast(x, dtype=dtypes.int32)


def _py_int(x, base):
  if base is UNSPECIFIED:
    return int(x)
  return int(x, base)


def len_(s):
  if tensors.is_tensor_array(s):
    return _tf_tensor_array_len(s)
  elif tensors.is_tensor_list(s):
    return _tf_tensor_list_len(s)
  elif tensor_util.is_tensor(s):
    return _tf_tensor_len(s)
  return _py_len(s)


def _tf_tensor_array_len(s):
  return s.size()


def _tf_tensor_list_len(s):
  return list_ops.tensor_list_length(s)


def _tf_tensor_len(s):
  """Overload of len_ for Tensor arguments."""
  # Statically shaped tensors: length is known ahead of time.
  if s.shape.ndims and s.shape.dims[0].value is not None:
    return s.shape.dims[0].value

  # Static shape of unknown dimensions: use dynamic shape but statically
  # check that it's a scalar.
  shape = array_ops.shape(s)

  assert shape.shape, 'shape tensor of zero size? {}'.format(shape)

  if shape.shape[0] == 0:
    raise ValueError(
        'len requires a non-scalar tensor, got one of shape {}'.format(shape))

  if shape.shape.dims[0].value is not None:
    return array_ops.shape(s)[0]

  # Fully dynamic shape: use ops.
  rank = array_ops.rank(s)

  def raise_zero_rank_error():
    msg = gen_string_ops.string_join(
        ['len requires non-zero rank, got ',
         gen_string_ops.as_string(rank)])
    with ops.control_dependencies([control_flow_ops.Assert(False, [msg])]):
      return constant_op.constant(0, dtype=dtypes.int32)

  return control_flow_ops.cond(rank > 0, lambda: array_ops.shape(s)[0],
                               raise_zero_rank_error)


def _py_len(s):
  return len(s)


def print_(*objects, **kwargs):
  """Overload of the print builtin."""
  # Note: Python 2.6 doesn't support explicit keywords after starargs.
  unknown_kwargs = tuple(
      set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
  if unknown_kwargs:
    raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))

  # TODO(mdan): Use next.flatten(objects) instead?
  if any(tensor_util.is_tensor(o) for o in objects):
    # TODO(mdan): use tf.print instead.
    return _tf_py_func_print(objects, kwargs)
  else:
    _py_print(*objects, **kwargs)


def _py_print(*objects, **kwargs):
  print(*objects, **kwargs)


def _tf_py_func_print(objects, kwargs):
  """Overload of print_ as a py_func implementation."""
  override_kwargs = {k: v for k, v in kwargs.items() if v is not UNSPECIFIED}
  if 'flush' not in override_kwargs:
    # Defaulting to flushing the console in graph mode, which helps reduce
    # garbled output in IPython.
    override_kwargs['flush'] = True

  def print_wrapper(*vals):
    vals = tuple(v.numpy() if tensor_util.is_tensor(v) else v for v in vals)
    if six.PY3:
      # TensorFlow doesn't seem to generate Unicode when passing strings to
      # py_func. This causes the print to add a "b'" wrapper to the output,
      # which is probably never what you want.
      vals = tuple(
          v.decode('utf-8') if isinstance(v, bytes) else v for v in vals)
    six.print_(*vals, **override_kwargs)

  return py_func.wrap_py_func(
      print_wrapper, None, objects, use_dummy_return=True)


def range_(start_or_stop, stop=UNSPECIFIED, step=UNSPECIFIED):
  if any(tensor_util.is_tensor(s) for s in (start_or_stop, stop, step)):
    return _tf_range(start_or_stop, stop, step)
  return _py_range(start_or_stop, stop, step)


def _tf_range(start_or_stop, stop, step):
  """Overload of range_ that generates a TF range tensor."""
  # Note: for static inputs (e.g. constants), tf.range errors out at graph
  # construction time, instead of returning an empty tensor. Preventing the
  # graph construction error aligns the semantics with Python.

  # TODO(mdan): We should optimize this when a full tensor is not required.
  if step is not UNSPECIFIED:
    # TODO(mdan): Add argument coercion similar to other cases.
    return math_ops.range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    stop = math_ops.maximum(start_or_stop, stop)
    return math_ops.range(start_or_stop, stop)
  start_or_stop = math_ops.maximum(start_or_stop, 0)
  return math_ops.range(start_or_stop)


def _py_range(start_or_stop, stop, step):
  if step is not UNSPECIFIED:
    return range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    return range(start_or_stop, stop)
  return range(start_or_stop)


def enumerate_(s, start=0):
  if isinstance(s, dataset_ops.DatasetV2):
    return _tf_dataset_enumerate(s, start)
  return _py_enumerate(s, start)


def _tf_dataset_enumerate(s, start=0):
  return s.enumerate(start)


def _py_enumerate(s, start=0):
  return enumerate(s, start)


def zip_(*iterables):
  if all(isinstance(x, dataset_ops.DatasetV2) for x in iterables):
    return _tf_dataset_zip(*iterables)
  return _py_zip(*iterables)


def _tf_dataset_zip(*iterables):
  return dataset_ops.DatasetV2.zip(tuple(iterables))


def _py_zip(*iterables):
  return zip(*iterables)


SUPPORTED_BUILTINS = (abs, float, int, len, print, range, enumerate, zip)

if six.PY2:
  SUPPORTED_BUILTINS += (xrange,)

BUILTIN_FUINCTIONS_MAP = {
    'abs': abs_,
    'float': float_,
    'int': int_,
    'len': len_,
    'print': print_,
    'range': range_,
    # TODO(mdan): This might make more sense as tf.data.range.
    'xrange': range_,
    'enumerate': enumerate_,
    'zip': zip_,
}
