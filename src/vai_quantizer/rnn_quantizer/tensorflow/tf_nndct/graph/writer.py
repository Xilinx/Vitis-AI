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

import numpy as np
import os
import tensorflow as tf

from collections import OrderedDict
from distutils.version import LooseVersion
from tensorflow.keras import activations
from tensorflow.keras import layers as keras_layers
from tensorflow.python.framework import dtypes as tf_dtypes
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

from tf_nndct import layers as nndct_layers
from tf_nndct import ops as nndct_ops
from tf_nndct.graph import OpTypes
from tf_nndct.graph import dtypes
from tf_nndct.graph import ops
from tf_nndct.graph import utils
from tf_nndct.ops.signal import fft_ops
from tf_nndct.quantization import utils as quant_utils
from tf_nndct.utils import generic_utils
from tf_nndct.utils import keras_utils
from tf_nndct.utils import registry
from tf_nndct.utils import tf_utils

if tf_utils.tf_version() >= LooseVersion('2.6'):
  from keras.utils import tf_utils as tf_keras_utils
else:
  from tensorflow.python.keras.utils import tf_utils as tf_keras_utils

_INPUT_ARG_PREFIX = '%input%_'

# call(self, input_0, input_1, ...)
_CALL_ARG_TEMPLATE = 'input_%d'

class CodeFormatter(object):

  def __init__(self, init_indent_level=0, indent_length=2):
    self._indent_level = init_indent_level
    self._init_indent_level = init_indent_level
    self._indent_length = indent_length
    self._start_of_line = False
    self._text = []

  def indent(self):
    self._indent_level += 1

  def outdent(self):
    if self._indent_level == 0 or self._indent_level < self._init_indent_level:
      raise RuntimeError("outdent() without matching indent()")
    self._indent_level -= 1

  def current_indentation(self):
    return self._indent_length * self._indent_level

  def newline(self):
    self.add_text('\n')

  def add_statement(self, statement):
    self.add_text(statement)
    self.newline()

  def add_text(self, text):
    text = str(text)
    if self._indent_level > 0:
      pos = 0
      for i, c in enumerate(text):
        if c == '\n':
          self._append(text[pos:i + 1])
          pos = i + 1
          self._start_of_line = True
      self._append(text[pos:])
    else:
      self._append(text)
      if text[-1] == '\n':
        self._start_of_line = True

  def code(self):
    return ''.join(self._text)

  def _append(self, text):
    if not text:
      return
    if self._start_of_line:
      self._start_of_line = False
      self._append_indent()

    self._text.append(text)

  def _append_indent(self):
    if self._indent_level == 0:
      return

    size = self.current_indentation()
    self._text.append(size * ' ')

class GraphCodeGenerator(object):
  """Generate python code from graph and write it to a given path.

  The code is keras subclass style and can be used to recreate a keras model
  represented by the graph.
  """

  def __init__(self, graph, class_spec):
    self._class_spec = class_spec

    # alias for imported module to be written.
    self._module_alias = {}

    self._graph = graph
    self._input_signature = graph.input_signature
    self._structured_output_tensors = graph.structured_output_tensors

    self._translator = GraphTranslator(graph, self._class_spec.quantized)

  def write(self, filepath):
    self._translator.translate()

    import_statements = self.generate_imports()
    class_def, init_statements, call_statements = self.generate_class_def()

    def format_function_statements(code_formatter, statements):
      code_formatter.indent()
      code_formatter.add_statement(statements[0])
      code_formatter.indent()
      for statement in statements[1:]:
        code_formatter.add_statement(statement)
      code_formatter.outdent()
      code_formatter.outdent()

    code_formatter = CodeFormatter()
    code_formatter.add_statement(
        '# THIS FILE IS GENERATED PROGRAMMATICALLY, DO NOT MODIFY IT MANUALLY.\n'
    )
    for imp in import_statements:
      code_formatter.add_statement(imp)

    code_formatter.newline()
    code_formatter.add_statement(class_def)
    format_function_statements(code_formatter, init_statements)
    code_formatter.newline()
    format_function_statements(code_formatter, call_statements)

    generic_utils.mkdir_if_not_exist(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
      f.write(code_formatter.code())
      f.flush()
      os.fsync(f.fileno())

    layer_to_node = {}
    for node in self._graph.nodes:
      entity = self._translator.get_entity(node.name)
      if entity.init_needed:
        layer_to_node[entity.name] = node

    return layer_to_node

  def generate_imports(self):
    objects = [self._class_spec.base]
    for entity in self._translator.entities:
      if not entity.obj:
        continue
      objects.append(entity.obj)

    imports = set(['import tensorflow as tf'])
    for obj in objects:
      imports.add(self._generate_obj_import(obj))

    return sorted(imports)
    #for impt in imports:
    #  self._code_formatter.add_statement(impt)
    #  self._code_formatter.newline()
    #self._code_formatter.newline()

  def _generate_obj_import(self, obj):
    module, pkg, name = _get_module(obj)
    if module in self._module_alias:
      alias = self._module_alias[module]
    else:
      alias = name
      while alias in self._module_alias.values():
        alias = '_' + alias
      self._module_alias[module] = alias

    if pkg == name:
      return 'import {}'.format(pkg)
    elif alias == name:
      return 'from {} import {}'.format(pkg, name)
    else:
      return 'from {} import {} as {}'.format(pkg, name, alias)

  def generate_class_def(self):
    module, _, _ = _get_module(self._class_spec.base)
    class_def = 'class {cls_name}({base_cls}):\n'.format(
        cls_name=self._class_spec.name,
        base_cls='.'.join(
            [self._module_alias[module], self._class_spec.base.__name__]))
    init_statements = self.generate_init()
    call_statements = self.generate_call_fn(self._class_spec.call_fn_name)
    return class_def, init_statements, call_statements

  def generate_init(self):
    statements = []
    statements.append('def __init__(self):')

    statements.append('super({}, self).__init__(name={})'.format(
        self._class_spec.name, utils.stringfy_to_write(self._class_spec.name)))

    for entity in self._translator.entities:
      if not entity.init_needed:
        continue

      init_str = self._init_string(entity)
      statements.append('self.{} = {}'.format(entity.name, init_str, entity))
    return statements

  def _init_string(self, entity):

    def stringfy_arg_value(value):
      if type(value) == tf_dtypes.DType:
        value = tf_utils.dtype_to_tf_string(value)
      elif isinstance(value, np.ndarray):
        value = value.tolist()
      elif isinstance(value, str):
        # Use a quote to wrap a string value.
        value = ''.join(['\'', value, '\''])
      elif isinstance(value, Entity):
        value = 'self.' + value.name
      else:
        pass
      return value

    args = []
    input_index = 0
    for arg, value in entity.args.items():
      # Use input entity name to fill in placeholder.
      if arg.startswith(_INPUT_ARG_PREFIX):
        args.append(entity.inputs[input_index].name)
        input_index += 1
        continue

      if not isinstance(value, (list, tuple)):
        args.append('{}={}'.format(arg, stringfy_arg_value(value)))
      else:
        value_strs = [stringfy_arg_value(val) for val in value]
        args.append(f'{arg}={str(value_strs)}')  #.format(arg, str(value_strs)))

    module, _, _ = _get_module(entity.obj)
    return '{module}.{obj_name}({args})'.format(
        module=self._module_alias[module],
        obj_name=entity.obj.__name__,
        args=', '.join(args))

  def generate_call_fn(self, fn_name):
    statements = []
    call_args = ['self']
    for i in range(len(self._input_signature)):
      call_args.append(_CALL_ARG_TEMPLATE % i)
    statements.append('def {}({}):'.format(fn_name, ', '.join(call_args)))

    placeholders = []
    for node in self._graph.nodes:
      if node.op.type == OpTypes.INPUT:
        assert len(node.out_tensors) == 1
        placeholders.append(node)
    # [args_0, args_1_1, args_1] -> [args_0, args_1, args_1_1]
    placeholders = sorted(placeholders, key=lambda x: x.name)

    flattened_input_signature = nest.flatten(self._input_signature)
    assert len(placeholders) == len(flattened_input_signature)

    # TODO(yuwang): Maybe use more solid way to do this.
    # Function arguments are Placeholder nodes in tf.Graph.
    # We sort the placeholders by their names and map the flattened input
    # signature in the sorted order. This method works now but it relies on
    # TensorFlow's implementation for generating placeholder names from input
    # argument. Specifically, the first input is named args_0, the second
    # input is named args_1 and so on. If an input is a type of sequence,
    # then the input is flattened to a list and each element in the list
    # is named by args_[input_index]_[element_index_in_the_list].
    # For example, we have a call function and the signature is like:
    #   def call(self, input, state)
    # The first argument 'input' is a tensor and TF will generate a
    # placeholder for it named args_0. The second argument 'state' is a list
    # of tensor, then TF will generate a placeholder for each element in the
    # list and the names of the placeholders are arg_1 and arg_1_1. In the
    # end, we got got three placeholders [args_0, args_1, args_1_1]. In a
    # nndct graph, the order of the placeholders may be [args_0, args_1_1,
    # args_1], that is why we sort the nodes by their names before mapping.
    # Example:
    # placeholders: ['args_0', 'args_1', 'args_1_1']
    # args: (TensorSpec(shape=(1, 10), dtype=tf.float32, name='args_0'),
    #            [TensorSpec(shape=(1, 5), dtype=tf.float32, name='args_1/0'),
    #             TensorSpec(shape=(1, 5), dtype=tf.float32, name='args_1/1')]
    #       )
    # args_to_placeholder = {
    #     'args_0': 'args_0',
    #     'args_1/0': 'args_1',
    #     'args_1/1': 'args_1_1',
    # }
    #
    # Example:
    # placeholders: ['args_0', 'args_0_1', 'args_0_2']
    # args: [{'y': TensorSpec(shape=(1, 5), dtype=tf.float32, name='args_0/0/y'),
    #         'x': TensorSpec(shape=(1, 5), dtype=tf.float32, name='args_0/0/x')},
    #        TensorSpec(shape=(1, 5), dtype=tf.float32, name='args_0/1')]
    # args_to_placeholder = {
    #     'args_0/0/y': 'arg_0',
    #     'args_0/0/x': 'arg_0_1',
    #     'args_0/1': 'arg_0_2',
    # }

    args_to_placeholder = {}
    for i, arg in enumerate(flattened_input_signature):
      args_to_placeholder[arg.name] = placeholders[i].name

    def index_string(getter, key):
      key = utils.stringfy_to_write(key)

      if getter == '[]':
        return '[{}]'.format(key)
      return '{}{}'.format(getter, key)

    def arg_retriving_path(arg, path=()):
      """
      Get retriving path of an argument.
      See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/nest.py::_yield_sorted_items

      Args:
        arg: The input signature of an argument.
        path: Current path.

      Yield:
        Path(gettter, key) used to retrive the given argument.
      """
      if not nest.is_sequence(arg):
        yield path
      elif isinstance(arg, nest._collections_abc.Mapping):
        for key in nest._sorted(arg):
          for res in arg_retriving_path(arg[key], path + (('[]', key),)):
            yield res
      elif nest._is_attrs(arg):
        for item in nest._get_attrs_items(arg):
          for res in arg_retriving_path(item[1], path + (('.', item[0]),)):
            yield res
      elif nest._is_namedtuple(arg):
        for field in arg._fields:
          for res in arg_retriving_path(
              getattr(arg, field), path + (('.', field),)):
            yield res
      # Doesn't support composite_tensor comprared with _yield_sorted_items.
      elif nest._is_type_spec(arg):
        # Note: to allow CompositeTensors and their TypeSpecs to have matching
        # structures, we need to use the same key string here.
        for res in arg_retriving_path(arg._component_specs,
                                      path + (('.', arg.value_type.__name__),)):
          yield res
      else:
        for item in enumerate(arg):
          for res in arg_retriving_path(item[1], path + (('[]', item[0]),)):
            yield res

    for i, arg in enumerate(self._input_signature):
      for path in list(arg_retriving_path(arg)):
        # [('[]' 0), ('[]', 'x')]
        arg_name_path = ['args_%d' % i]
        retriving_path = []
        for getter, key in path:
          arg_name_path.append(key)
          retriving_path.append(index_string(getter, key))
        arg_name = '/'.join(str(s) for s in arg_name_path)
        retriving = ''.join(r for r in retriving_path)
        if arg_name not in args_to_placeholder:
          continue

        placeholder_name = args_to_placeholder[arg_name]
        entity = self._translator.get_entity(placeholder_name)
        call_arg = _CALL_ARG_TEMPLATE % i
        entity.inputs.append(
            Entity('{}{}'.format(call_arg, retriving), EntityTypes.Tensor))

    # TODO(yuwang): Use entities directly, no longer use node.
    for node in self._graph.nodes:
      statements.append(self._call_string(node))

    output_tensors = [
        t.name for t in nest.flatten(self._structured_output_tensors)
    ]
    returns = [self._translator.get_entity(t) for t in output_tensors]
    if len(returns) == 1:
      returns = returns[0]
    else:
      returns = nest.pack_sequence_as(self._structured_output_tensors, returns)

    def return_string(returns):
      return_strs = []
      if isinstance(returns, Entity):
        s = returns.name
      elif isinstance(returns, list):
        for ret in returns:
          return_strs.append(return_string(ret))
        s = ''.join(['[', ', '.join(return_strs), ']'])
      elif isinstance(returns, tuple):
        for ret in returns:
          return_strs.append(return_string(ret))
        s = ''.join(['(', ', '.join(return_strs), ')'])
      elif isinstance(returns, dict):
        for key in returns:
          return_strs.append('{}: {}'.format(
              utils.stringfy_to_write(key),
              ''.join(return_string(returns[key]))))
        s = ''.join(['{', ', '.join(return_strs), '}'])
      else:
        raise NotImplementedError(
            'Can not rewrite return object of type '.format(type(returns)))
      return s

    statements.append('return %s' % return_string(returns))
    return statements

  def _call_string(self, node):
    entity = self._translator.get_entity(node.name)

    input_names = [str(input) for input in entity.inputs]
    output_names = [output.name for output in entity.outputs]

    input_str = ', '.join(input_names)
    output_str = ', '.join(output_names)

    # Layer type
    if entity.init_needed:
      forward_template = '{outputs} = self.{entity}({inputs}) # {node}'
      args_dict = {
          'outputs': output_str,
          'entity': entity.name,
          'inputs': input_str,
          'node': node.name,
      }
      forward_str = forward_template.format(**args_dict)
    else:
      # Function type
      init_str = self._init_string(entity)
      forward_str = '{} = {} # {}'.format(output_str, init_str, node.name)
    return forward_str

class EntityTypes(object):
  Layer = 'layer'
  Function = 'function'
  Tensor = 'tensor'

class Entity(object):

  def __init__(self, name, type, obj=None, args=None):
    self.name = name

    # One of `EntityTypes`.
    self.type = type

    # For EntityTypes.Layer and EntityTypes.Function,
    # obj is the corresponding layer class or function.
    # For EntityTypes.Tensor, obj is None
    self.obj = obj
    self.args = args

    #self.node_name = node_name

    # I/O entities
    self.inputs = []
    self.outputs = []

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name

  def desp(self):
    return '({}, {})'.format(self.name, self.obj)

  @property
  def init_needed(self):
    return self.type == EntityTypes.Layer

class GraphTranslator(object):
  """Translates a computation graph to a call graph composed of entities.

  A `Node` will be translated to an `Entity` object with nested entities.
  Specifically, translator will convert all `Operation` and `Tensor` objects
  in `Node` to entities.
  """
  _op_to_layer = {
      OpTypes.BATCH_NORM:
          keras_layers.BatchNormalization,
      OpTypes.BIDIRECTIONAL_RNN:
          keras_layers.Bidirectional,
      OpTypes.CONV1D:
          keras_layers.Conv1D,
      OpTypes.CONV2D:
          keras_layers.Conv2D,
      OpTypes.DENSE:
          keras_layers.Dense,
      OpTypes.EMBEDDING:
          keras_layers.Embedding,
      OpTypes.GRU:
          keras_layers.GRU,
      OpTypes.INPUT:
          nndct_layers.Identity,
      OpTypes.LSTM:
          nndct_layers.LSTM,
      #OpTypes.LSTM: keras_layers.LSTM,
      OpTypes.LSTM_CELL:
          nndct_layers.LSTMCell,
      OpTypes.MAX_POOL1D:
          keras_layers.MaxPooling1D,
      OpTypes.RNN:
          keras_layers.RNN,
      OpTypes.SIMPLE_RNN:
          keras_layers.SimpleRNN,
      OpTypes.STACKED_RNN_CELLS:
          keras_layers.StackedRNNCells,
  }

  _op_to_func = {
      OpTypes.ADD: tf.math.add,
      OpTypes.CAST: tf.cast,
      OpTypes.BIAS_ADD: tf.nn.bias_add,
      OpTypes.IDENTITY: tf.identity,
      OpTypes.LINEAR: activations.linear,
      OpTypes.RELU: activations.relu,
      OpTypes.SIGMOID: activations.sigmoid,
      OpTypes.SOFTMAX: activations.softmax,
      OpTypes.TANH: activations.tanh,
      OpTypes.MULTIPLY: tf.math.multiply,
      OpTypes.STRIDED_SLICE: tf.strided_slice,
      OpTypes.GATHER: nndct_ops.gather,
      'rfft': fft_ops.rfft,
      'complex_abs': tf.abs,
      'angle': tf.math.angle,
      'cast': tf.cast,
      'exp': tf.math.exp,
      'irfft': fft_ops.irfft,
      'pad': tf.pad,
      'transpose': tf.transpose,
      'sum': tf.math.reduce_sum,
      'reshape': tf.reshape,
      OpTypes.CONST: tf.constant,
  }

  def __init__(self, graph, quantized=False):
    self._graph = graph
    self._quantized = quantized

    # All entities translated from ops.
    self._entities = []

    # Node name -> index in self._entities
    self._name_to_entity = {}

    # Op type -> count
    self._op_count = {}

  def _append_entity(self, entity, name=None):
    """
      Args:
        name: Node or Tensor's name.
        entity: An `Entity` object.
    """
    self._entities.append(entity)
    if name:
      self._name_to_entity[name] = len(self._entities) - 1

  def _get_tf_object(self, op):
    if self._quantized:
      obj = quant_utils.get_quant_module(op.type, None)
      if obj:
        return obj, EntityTypes.Layer

    if op.type in self._op_to_layer:
      obj = self._op_to_layer[op.type]
      entity_type = EntityTypes.Layer
    elif op.type in self._op_to_func:
      obj = self._op_to_func[op.type]
      entity_type = EntityTypes.Function
    elif op.type == OpTypes.RNN_LAYER or op.type == OpTypes.GENERIC:
      obj = op.attr['layer_class']
      entity_type = EntityTypes.Layer
    else:
      raise NotImplementedError("Unable to rewrite operation '{}'".format(
          op.type))
    return obj, entity_type

  def translate(self):
    for node in self._graph.nodes:
      entity = self._translate_node(node)
      self._append_entity(entity, node.name)

      for index, tensor in enumerate(node.out_tensors):
        # Use node entity's name as prefix
        tensor_entity = Entity('%s_%d' % (entity.name, index),
                               EntityTypes.Tensor)
        self._append_entity(tensor_entity, tensor.name)
        entity.outputs.append(tensor_entity)

    # We only want to get the sequence type so we just take the first
    # node data instead of iterating through all the nodes.
    #
    # Example: pack multiply's input entities to [[entity0, entity1]]
    # {'class_name': 'Multiply', 'config': {'name': 'multiply', 'trainable': True, 'dtype': 'float32'},
    # 'name': 'multiply', 'inbound_nodes': [[['lambda', 0, 0, {}], ['activation', 0, 0,{}]]]}
    #
    # See tensorflow/python/keras/engine/functional.py::process_node
    for node in self._graph.nodes:
      input_entities = [
          self.get_entity(tensor.name) for tensor in node.in_tensors
      ]
      entity = self.get_entity(node.name)
      if node.op.type != OpTypes.INPUT and entity.type == EntityTypes.Layer:
        if node.inbound_nodes:
          inbound_nodes_data = tf_keras_utils.convert_inner_node_data(
              node.inbound_nodes, wrap=True)
          node_data = [inbound_nodes_data[0]]
          input_entities = nest.pack_sequence_as(node_data, input_entities)
          input_entities = keras_utils.unnest_if_single_tensor(input_entities)
          if isinstance(input_entities, Entity):
            input_entities = [input_entities]
      entity.inputs = input_entities

  def _translate_node(self, node):
    obj, ent_type = self._get_tf_object(node.op)
    if ent_type == EntityTypes.Layer:
      self._op_to_entity(node.op)
      argspec = tf_inspect.getfullargspec(obj.__init__)
      arg_to_value = self._arguments_for_keras_layer_init(node.op, argspec)
    else:
      argspec = tf_inspect.getfullargspec(obj)
      arg_to_value = self._arguments_for_tf_operation_calling(node, argspec)
    return Entity(
        self._unique_entity_name(node.op), ent_type, obj, arg_to_value)

  def _op_to_entity(self, op):
    """Given an `Operation`, traverse its configs and find all `Operation`
    items, then:
    1. Convert these ops to `Entity` objects
    2. Set these entities back to configs to replace original ops.

    Arguments:
      op: An `Operation` object.

    Returns:
      An entity converted from the given operation.

    Example:
      Given: RNN -> config('cell') -> StackedRNNCells -> config('cells')
          -> [LSTMCell0, LSTMCell1]
      Returns: Entity(RNN) -> config('cell') -> Entity(StackedRNNCells)
          -> config('cells') -> [Entity(LSTMCell0), Entity(LSTMCell1)]
    """

    obj, ent_type = self._get_tf_object(op)
    assert ent_type == EntityTypes.Layer
    argspec = tf_inspect.getfullargspec(obj.__init__)

    for name in op.configs:
      config_value = op.get_config(name)
      config_values = generic_utils.to_list(config_value)

      # Convert Operation to Entity.
      converted_values = []
      for value in config_values:
        if isinstance(value, ops.Operation):
          entity = self._op_to_entity(value)
          self._append_entity(entity)
          converted_values.append(entity)
        else:
          converted_values.append(value)

      # Keep as the original type.
      if isinstance(config_value, tuple):
        config_value = tuple(converted_values)
      elif isinstance(config_value, list):
        config_value = converted_values
      elif len(converted_values) <= 1:
        config_value = converted_values[0]
      else:
        raise RuntimeError('Unexpected sequence type: {}'.format(
            type(config_value)))
      op.set_config(name, config_value)

    arg_to_value = self._arguments_for_keras_layer_init(op, argspec)
    return Entity(self._unique_entity_name(op), ent_type, obj, arg_to_value)

  def _arguments_for_keras_layer_init(self, op, argspec):
    # Keep the order of args same as the original signature.
    default_count = 0 if not argspec.defaults else len(argspec.defaults)
    required_count = len(argspec.args) - default_count

    arg_to_value = OrderedDict()
    for arg in argspec.args[:required_count]:
      if arg == 'self':
        continue
      elif arg in op.configs:
        arg_to_value[arg] = op.get_config(arg)
      else:
        raise RuntimeError(
            'Missing value for argument "{}" of operation {}'.format(
                arg, op.type))
    if default_count:
      for arg, value in zip(argspec.args[-default_count:], argspec.defaults):
        # Skip args that have default values.
        if arg not in op.configs or op.get_config(arg) == value:
          continue
        arg_to_value[arg] = op.get_config(arg)
    return arg_to_value

  def _arguments_for_tf_operation_calling(self, node, argspec):
    # Keep the order of args same as the original signature.
    default_count = 0 if not argspec.defaults else len(argspec.defaults)
    required_count = len(argspec.args) - default_count

    arg_to_value = OrderedDict()
    num_inputs = len(node.in_tensors)
    input_count = 0
    for index, arg in enumerate(argspec.args):
      if arg in node.op.configs:
        arg_to_value[arg] = node.op.get_config(arg)
      elif input_count < num_inputs:
        # If an argument not in configs, we treat it as a tensor.
        arg_to_value[_INPUT_ARG_PREFIX + arg] = None
        input_count += 1
      elif index < required_count:
        raise ValueError('Missing required argument for {}'.format(
            node.op.type))

    if input_count < num_inputs:
      raise ValueError(
          'Unused tensor founded, there may be a mismatch with function signature of {}'
          .format(node.op.type))
    return arg_to_value

  def _unique_entity_name(self, op):
    count = self._op_count.get(op.type, 0)
    self._op_count[op.type] = count + 1
    return '%s%d' % (op.type, count)

  def get_entity(self, name):
    return self._entities[self._name_to_entity[name]]

  @property
  def entities(self):
    return self._entities

def _get_module(object):
  module = tf_inspect.getmodule(object)
  module_full_name = module.__name__
  if '.' in module_full_name:
    pkg, module_name = module.__name__.rsplit('.', 1)
  else:
    pkg = module_full_name
    module_name = module_full_name
  return (module, pkg, module_name)
