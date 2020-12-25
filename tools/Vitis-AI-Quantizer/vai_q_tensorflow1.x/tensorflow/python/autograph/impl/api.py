# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""This module contains the user-facing API for AutoGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import inspect
import os
import pdb
import re
import sys
import textwrap
import traceback

# pylint:disable=g-bad-import-order
import six
# pylint:enable=g-bad-import-order

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export


def is_autograph_strict_conversion_mode():
  return int(os.environ.get('AUTOGRAPH_STRICT_CONVERSION', '0')) > 0


# TODO(mdan): Export this symbol.
class AutoGraphError(Exception):
  """Base class for all AutoGraph exceptions."""
  pass


class ConversionError(AutoGraphError):
  """Raised during the conversion process."""
  pass


class StagingError(AutoGraphError):
  """Raised during the staging (i.e. Python execution) of converted code."""
  pass


class _ErrorMetadata(errors.ErrorMetadataBase):
  """AutoGraph-specific error metadata. See base class."""

  def create_exception(self, source_error):
    preferred_type = type(source_error)
    if issubclass(preferred_type, errors_impl.OpError):
      # Best-effort unpacking of OpError exceptions.
      # TODO(mdan): Use a mechanism that is more future-proof.
      init_argspec = tf_inspect.getfullargspec(preferred_type.__init__)
      message = self.get_message()
      init_args = tuple(init_argspec.args)
      # At the time of this writing, TF errors either take 3 or 4 arguments,
      # with the fourth being error_code.
      if init_args == ('self', 'node_def', 'op', 'message', 'error_code'):
        return preferred_type(
            node_def=source_error.node_def,
            op=source_error.op,
            message=message,
            error_code=self.error_code)
      elif init_args == ('self', 'node_def', 'op', 'message'):
        if 'error_code' in init_argspec.kwonlyargs:
          return preferred_type(
              node_def=source_error.node_def,
              op=source_error.op,
              message=message,
              errro_code=self.error_code)
        else:
          return preferred_type(
              node_def=source_error.node_def,
              op=source_error.op,
              message=message)

    elif preferred_type in (AutoGraphError, ConversionError, StagingError,
                            errors_impl.InaccessibleTensorError,
                            errors_impl.OperatorNotAllowedInGraphError):
      return preferred_type(self.get_message())

    exc = super(_ErrorMetadata, self).create_exception(source_error)
    if exc is not None:
      return exc

    # Note: While changing an error's message property to change the message it
    # displays will probably work a lot of times, there is no standard way in
    # Python to do that. The safest way is therefore to create a new exception.
    # For user defined exceptions, we could define an interface that allowed
    # them to work under this mechanism.
    return StagingError(self.get_message())


class StackTraceMapper(tf_stack.StackTraceMapper):
  """Remaps generated code to code it originated from."""

  def __init__(self, converted_fn):
    self._source_map = converted_fn.ag_source_map

  def get_effective_source_map(self):
    effective_source_map = self._effective_source_map
    if effective_source_map is None:
      if self.parent is not None:
        parent_map = self.parent.get_effective_source_map()
      else:
        parent_map = {}

      effective_source_map = {}
      for loc, origin in self._source_map.items():
        effective_source_map[(loc.filename, loc.lineno)] = (
            origin.loc.filename, origin.loc.lineno, origin.function_name)

      for key, value in parent_map.items():
        filename, lineno, _ = value
        value_loc = origin_info.LineLocation(filename=filename, lineno=lineno)
        if value_loc in self._source_map:
          origin = self._source_map[value_loc]
          effective_source_map[key] = (
              origin.loc.filename, origin.loc.lineno, origin.function_name)
        else:
          effective_source_map[key] = value
      self._effective_source_map = effective_source_map
    return effective_source_map


def tf_convert(f, ctx, convert_by_default=True, user_requested=False):
  """Decorator that applies AutoGraph to a function.

  Use in internal APIs.

  This API is suitable for high order functions internal to the TensorFlow API,
  and more generally any function to which Autograph is not applied.

  Guidance: convert was a decorator meant for use directly by developers, and
  will be soon deprecated in favor of tf.function. tf_convert is to be called
  from high order functions internal to TF.

  Args:
    f: Callable.
    ctx: ag_ctx.ControlStatusCtx, the Autograph context in which `f` is used.
    convert_by_default: bool, whether to use AutoGraph when the context doesn't
      specify.
    user_requested: bool, whether to ignore the conversion whitelist. See
      ConversionOptions.user_requested.

  Returns:
    Either `f or the converted version of `f`.
  """

  if hasattr(f, '__ag_compiled'):
    return f
  f_wrapper = f
  decorators, f = tf_decorator.unwrap(f)

  # TODO(mdan): Grab features from context.
  if ctx.status == ag_ctx.Status.ENABLED:
    wrapper = convert(recursive=True, user_requested=user_requested)(f)
  elif ctx.status == ag_ctx.Status.DISABLED:
    wrapper = do_not_convert(f)
  elif ctx.status == ag_ctx.Status.UNSPECIFIED:
    if convert_by_default:
      wrapper = convert(recursive=True, user_requested=user_requested)(f)
    else:
      wrapper = call_with_unspecified_conversion_status(f)
  else:
    raise ValueError(ctx.status)

  if decorators:
    wrapper = tf_decorator.rewrap(f_wrapper, f, wrapper)

  setattr(wrapper, '__ag_compiled', True)
  return wrapper


# TODO(mdan): Make private.
def convert(recursive=False, optional_features=None, user_requested=True):
  """Decorator that compiles a function to use TensorFlow ops.

  The decorator is dynamic - it recompiles the target whenever the decorated
  function is called. This means the parameter values are known at conversion.
  It also means that repeated calls with different types of parameters will be
  correctly processed.

  Args:
    recursive: bool, whether to recursively convert any functions or classes
      that the converted function may use.
    optional_features: converted.Feature, allows toggling optional or
      experimental features. When set to None, only the core features are
      enabled.
    user_requested: bool, whether to ignore the conversion whitelist. See
      ConversionOptions.user_requested.

  Returns:
    Callable, a decorator that converts the given function into an equivalent
    function that uses TensorFlow ops.
  """

  def decorator(f):
    """Decorator implementation."""

    def wrapper(*args, **kwargs):
      """Wrapper that calls the converted version of f."""
      options = converter.ConversionOptions(
          recursive=recursive,
          user_requested=user_requested,
          optional_features=optional_features)
      try:
        return converted_call(f, options, args, kwargs)
      except Exception as e:  # pylint:disable=broad-except
        if hasattr(e, 'ag_error_metadata'):
          raise e.ag_error_metadata.to_exception(e)
        else:
          raise

    if inspect.isfunction(f) or inspect.ismethod(f):
      wrapper = functools.update_wrapper(wrapper, f)

    decorated_wrapper = tf_decorator.make_decorator(f, wrapper)

    # Sometimes the decorator is just desugared, making it impossible to detect.
    # This attribute makes detection easier.
    setattr(decorated_wrapper, '__ag_compiled', True)
    return decorated_wrapper

  return decorator


def call_with_unspecified_conversion_status(func):
  """Decorator that resets the conversion context to the unspecified status."""
  def wrapper(*args, **kwargs):
    with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.UNSPECIFIED):
      return func(*args, **kwargs)

  if inspect.isfunction(func) or inspect.ismethod(func):
    wrapper = functools.update_wrapper(wrapper, func)

  setattr(wrapper, '__ag_compiled', True)
  return wrapper


def do_not_convert_internal(f):
  """Decorator that marks internal functions which do not need conversion."""
  setattr(f, '__ag_compiled', True)
  return f


@tf_export('autograph.experimental.do_not_convert')
def do_not_convert(func=None):
  """Decorator that suppresses the conversion of a function.

  Args:
    func: function to decorate.

  Returns:
    If `func` is not None, returns a `Callable` which is equivalent to
    `func`, but is not converted by AutoGraph.
    If `func` is None, returns a decorator that, when invoked with a
    single `func` argument, returns a `Callable` equivalent to the
    above case.
  """
  if func is None:
    return do_not_convert

  def wrapper(*args, **kwargs):
    with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED):
      return func(*args, **kwargs)

  if inspect.isfunction(func) or inspect.ismethod(func):
    wrapper = functools.update_wrapper(wrapper, func)

  setattr(wrapper, '__ag_compiled', True)
  return wrapper


def _attach_metadata(e, f, converted):
  """Augments an error with the metadata necessary for rewrite."""
  if hasattr(e, 'ag_pass_through'):
    return

  metadata = getattr(e, 'ag_error_metadata', None)
  source_map = f.ag_source_map if converted else {}

  if metadata is None:
    logging.log(
        1, 'Caught error in %s (converted=%s)', f, converted, exc_info=True)
    message = '{}: {}'.format(e.__class__.__name__, e)
  else:
    message = None

  cause_tb = traceback.extract_tb(sys.exc_info()[2])[1:]
  e.ag_error_metadata = _ErrorMetadata(cause_tb, metadata, message, source_map)


def _call_unconverted(f, args, kwargs, options, update_cache=True):
  """Calls the original function without converting with AutoGraph."""
  if update_cache:
    conversion.cache_unconverted(f, options)

  if inspect_utils.istfmethodtarget(f):
    return f.__self__.call(args, kwargs)

  try:
    if kwargs is not None:
      return f(*args, **kwargs)
    else:
      return f(*args)
  except Exception as e:  # pylint:disable=broad-except
    _attach_metadata(e, f, False)
    raise


def _is_known_loaded_type(f, module_name, entity_name):
  """Tests whether the function or method is an instance of a known type."""
  if (module_name not in sys.modules or
      not hasattr(sys.modules[module_name], entity_name)):
    return False
  type_entity = getattr(sys.modules[module_name], entity_name)
  if isinstance(f, type_entity):
    # The method if of this type. Example:
    #
    # o = ClassType()
    # function(o.method)()
    return True
  # Note: inspect is required here, to avoid unpacking tf.function decorators.
  if inspect.ismethod(f):
    # The the unbound method if of this type. Example:
    #
    # class ClassType:
    #   @function
    #   def method(self):
    #     ...
    # o = ClassType()
    # o.method()
    if isinstance(f.__func__, type_entity):
      return True
  return False


def converted_call(f, options, args, kwargs, caller_fn_scope=None):
  """Compiles a function call inline.

  For internal use only.

  Args:
    f: The function to convert.
    options: converter.ConversionOptions
    args: Tuple, the original positional arguments of f
    kwargs: Dict, the original keyword arguments of f
    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
      scope of the converted function in which this call was originally made.

  Returns:
    Any, the result of executing a possibly-converted `f` with the given
      arguments.
  """
  logging.log(1, 'Converted call: %s\n    args: %s\n    kwargs: %s\n', f, args,
              kwargs)

  if conversion.check_cached_unconverted(f, options):
    return _call_unconverted(f, args, kwargs, options, False)

  if inspect_utils.isbuiltin(f):
    if f is eval:
      return py_builtins.eval_in_original_context(f, args, caller_fn_scope)
    if f is super:
      return py_builtins.super_in_original_context(f, args, caller_fn_scope)
    if kwargs:
      return py_builtins.overload_of(f)(*args, **kwargs)
    else:
      return py_builtins.overload_of(f)(*args)

  # TODO(mdan): Clean up the naming inconsistency.
  if hasattr(f, 'autograph_info__') or hasattr(f, '__ag_compiled'):
    logging.log(2, 'Permanently whitelisted: %s: already converted', f)
    return _call_unconverted(f, args, kwargs, options)

  # TODO(b/122265385): Remove this bypass.
  if (_is_known_loaded_type(f, 'wrapt', 'FunctionWrapper') or
      _is_known_loaded_type(f, 'wrapt', 'BoundFunctionWrapper')):
    logging.warn(
        'Entity {} appears to be decorated by wrapt, which is not yet supported'
        ' by AutoGraph. The function will be called without transformation.'
        ' You may however apply AutoGraph before the decorator.'.format(f))
    logging.log(2, 'Permanently whitelisted: %s: wrapt decorated', f)
    return _call_unconverted(f, args, kwargs, options)

  if _is_known_loaded_type(f, 'functools', '_lru_cache_wrapper'):
    logging.log(2, 'Permanently whitelisted: %s: lru_cache', f)
    return _call_unconverted(f, args, kwargs, options)

  # Constructors are permanently whitelisted.
  # TODO(mdan): Toggle as experimental feature instead.
  # TODO(b/124016764): Remove this limitation.
  if tf_inspect.isclass(f):
    logging.log(2, 'Permanently whitelisted: %s: constructor', f)
    return _call_unconverted(f, args, kwargs, options)

  # Other built-in modules are permanently whitelisted.
  # TODO(mdan): Figure out how to do this consistently for all stdlib modules.
  if any(
      f in m.__dict__.values() for m in (collections, pdb, copy, inspect, re)):
    logging.log(2, 'Permanently whitelisted: %s: part of builtin module', f)
    return _call_unconverted(f, args, kwargs, options)

  # Custom ops and kernels are also permanently whitelisted.
  # See tensorflow.framework.load_library.
  if (hasattr(f, '__module__') and
      hasattr(f.__module__, '_IS_TENSORFLOW_PLUGIN')):
    logging.log(2, 'Permanently whitelisted: %s: TensorFlow plugin', f)
    return _call_unconverted(f, args, kwargs, options)

  if not options.user_requested and conversion.is_whitelisted_for_graph(f):
    return _call_unconverted(f, args, kwargs, options)

  # internal_convert_user_code is for example turned off when issuing a dynamic
  # call conversion from generated code while in nonrecursive mode. In that
  # case we evidently don't want to recurse, but we still have to convert
  # things like builtins.
  if not options.internal_convert_user_code:
    return _call_unconverted(f, args, kwargs, options)

  # TODO(mdan): Move this entire block inside to_graph.
  try:  # Begin of transformation error guards

    # Unwrap functools.partial objects
    # TODO(mdan): Consider sharing unwrapping logic with tf_inspect.
    # TODO(b/120224672): This unwrapping should be done before the checks above.
    while isinstance(f, functools.partial):
      args = f.args + args
      new_kwargs = {}
      if f.keywords is not None:
        new_kwargs.update(f.keywords)
      if kwargs is not None:
        new_kwargs.update(kwargs)
      kwargs = new_kwargs
      f = f.func

    if tf_inspect.isfunction(f) or tf_inspect.ismethod(f):
      # Regular functions
      target_entity = f
      f_self = inspect_utils.getmethodself(f)

      # TODO(b/119246461): This may be more elegantly handled using __get__?
      if f_self is not None:
        effective_args = (f_self,) + args
      else:
        effective_args = args

    elif hasattr(f, '__call__') and hasattr(f, '__class__'):
      # Callable objects
      target_entity = f.__call__
      effective_args = (f,) + args

    elif tf_inspect.isclass(f):
      # Constructors
      # Note: Until we support class constructurs, and enable whole-class
      # conversion with an experimental flag, this branch is dead code.
      # TODO(mdan): Consider removing unless there is a compelling use case.
      target_entity = f
      effective_args = args

    else:
      target_entity = f
      raise NotImplementedError('unknown callable type "%s"' % type(f))

    if not tf_inspect.isclass(target_entity):
      if not hasattr(target_entity, '__code__'):
        logging.log(2, 'Permanently whitelisted: %s: native binding',
                    target_entity)
        return _call_unconverted(f, args, kwargs, options)
      elif (hasattr(target_entity.__code__, 'co_filename') and
            target_entity.__code__.co_filename == '<string>'):
        # TODO(mdan): __globals__['txt'] might work in Py3.
        logging.log(2, 'Permanently whitelisted: %s: dynamic code (exec?)',
                    target_entity)
        return _call_unconverted(f, args, kwargs, options)

    program_ctx = converter.ProgramContext(
        options=options, autograph_module=tf_inspect.getmodule(converted_call))
    converted_f = conversion.convert(target_entity, program_ctx)

    if logging.has_verbosity(2):
      logging.log(2, 'Defaults of %s : %s', converted_f,
                  converted_f.__defaults__)
      if six.PY3:
        logging.log(2, 'KW defaults of %s : %s',
                    converted_f, converted_f.__kwdefaults__)

      if kwargs is not None:
        callargs = tf_inspect.getcallargs(converted_f, *effective_args,
                                          **kwargs)
      else:
        callargs = tf_inspect.getcallargs(converted_f, *effective_args)

      formatted_callargs = '\n'.join(
          '    {}: {}'.format(k, v) for k, v in callargs.items())
      logging.log(2, 'Calling %s with\n%s\n', converted_f, formatted_callargs)

  except Exception as e:  # pylint:disable=broad-except
    logging.log(1, 'Error transforming entity %s', target_entity, exc_info=True)
    if is_autograph_strict_conversion_mode():
      raise
    logging.warn(
        'Entity %s could not be transformed and will be executed as-is.'
        ' Please report this to the AutoGraph team. When filing the bug, set'
        ' the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and'
        ' attach the full output. Cause: %s', target_entity, e)
    return _call_unconverted(f, args, kwargs, options)

  with StackTraceMapper(converted_f), tf_stack.CurrentModuleFilter():
    try:
      if kwargs is not None:
        result = converted_f(*effective_args, **kwargs)
      else:
        result = converted_f(*effective_args)
    except Exception as e:
      _attach_metadata(e, converted_f, True)
      raise

  return result


@tf_export('autograph.to_graph', v1=[])
def to_graph(entity, recursive=True, experimental_optional_features=None):
  """Converts a Python entity into a TensorFlow graph.

  Also see: `tf.autograph.to_code`, `tf.function`.

  Unlike `tf.function`, `to_graph` is a low-level transpiler that converts
  Python code to TensorFlow graph code. It does not implement any caching,
  variable management or create any actual ops, and is best used where greater
  control over the generated TensorFlow graph is desired. Another difference
  from `tf.function` is that `to_graph` will not wrap the graph into a
  TensorFlow function or a Python callable. Internally, `tf.function` uses
  `to_graph`.

  _Example Usage_

  ```python
    def foo(x):
      if x > 0:
        y = x * x
      else:
        y = -x
      return y

    converted_foo = to_graph(foo)

    x = tf.constant(1)
    y = converted_foo(x)  # converted_foo is a TensorFlow Op-like.
    assert is_tensor(y)
  ```

  Supported Python entities include:
    * functions
    * classes
    * object methods

  Functions are converted into new functions with converted code.

  Classes are converted by generating a new class whose methods use converted
  code.

  Methods are converted into unbound function that have an additional first
  argument called `self`.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the converted
      function may call.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    Same as `entity`, the converted Python function or class.

  Raises:
    ValueError: If the entity could not be converted.
  """
  try:
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(
            recursive=recursive,
            user_requested=True,
            optional_features=experimental_optional_features),
        autograph_module=tf_inspect.getmodule(to_graph))
    return conversion.convert(entity, program_ctx)
  except (ValueError, AttributeError, KeyError, NameError, AssertionError) as e:
    logging.error(1, 'Error converting %s', entity, exc_info=True)
    raise ConversionError('converting {}: {}: {}'.format(
        entity, e.__class__.__name__, str(e)))


@tf_export(v1=['autograph.to_graph'])
def to_graph_v1(entity,
                recursive=True,
                arg_values=None,
                arg_types=None,
                experimental_optional_features=None):
  """Converts a Python entity into a TensorFlow graph.

  Also see: `tf.autograph.to_code`, `tf.function`.

  Unlike `tf.function`, `to_graph` is a low-level transpiler that converts
  Python code to TensorFlow graph code. It does not implement any caching,
  variable management or create any actual ops, and is best used where greater
  control over the generated TensorFlow graph is desired. Another difference
  from `tf.function` is that `to_graph` will not wrap the graph into a
  TensorFlow function or a Python callable. Internally, `tf.function` uses
  `to_graph`.

  _Example Usage_

  ```python
    def foo(x):
      if x > 0:
        y = x * x
      else:
        y = -x
      return y

    converted_foo = to_graph(foo)

    x = tf.constant(1)
    y = converted_foo(x)  # converted_foo is a TensorFlow Op-like.
    assert is_tensor(y)
  ```

  Supported Python entities include:
    * functions
    * classes
    * object methods

  Functions are converted into new functions with converted code.

  Classes are converted by generating a new class whose methods use converted
  code.

  Methods are converted into unbound function that have an additional first
  argument called `self`.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the converted
      function may call.
    arg_values: Deprecated.
    arg_types: Deprecated.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    Same as `entity`, the converted Python function or class.

  Raises:
    ValueError: If the entity could not be converted.
  """
  del arg_types
  del arg_values
  return to_graph(
      entity,
      recursive=recursive,
      experimental_optional_features=experimental_optional_features)


@tf_export(v1=['autograph.to_code'])
def to_code_v1(entity,
               recursive=True,
               arg_values=None,
               arg_types=None,
               indentation='  ',
               experimental_optional_features=None):
  """Similar to `to_graph`, but returns Python source code as a string.

  Also see: `tf.autograph.to_graph`.

  `to_graph` returns the Python source code that can be used to generate a
  TensorFlow graph that is functionally identical to the input Python code.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the converted
      function may call.
    arg_values: Deprecated.
    arg_types: Deprecated.
    indentation: Deprecated.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    The converted code as string.
  """
  del arg_values
  del arg_types
  del indentation
  return to_code(
      entity,
      recursive=recursive,
      experimental_optional_features=experimental_optional_features)


@tf_export('autograph.to_code', v1=[])
def to_code(entity, recursive=True, experimental_optional_features=None):
  """Similar to `to_graph`, but returns Python source code as a string.

  Also see: `tf.autograph.to_graph`.

  `to_graph` returns the Python source code that can be used to generate a
  TensorFlow graph that is functionally identical to the input Python code.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the converted
      function may call.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    The converted code as string.
  """
  source = tf_inspect.getsource(
      to_graph(
          entity,
          recursive=recursive,
          experimental_optional_features=experimental_optional_features))
  return textwrap.dedent(source)
