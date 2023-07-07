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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import six
import tempfile
import tensorflow as tf

from collections import OrderedDict
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes as tf_dtypes
from tensorflow.python.framework import tensor_util

from tf_nndct.graph import dtypes as nndct_dtypes
from tf_nndct.utils.convert_to_constants import convert_variables_to_constants_v2 as convert_to_constants

from tf_nndct.utils import generic_utils

def node_name_parts_from_input(input_name):
  prefix = ''
  node_name = ''
  suffix = ''
  if input_name.startswith('^'):
    prefix = '^'
    input_name = input_name[1:]

  input_parts = input_name.split(':')
  if len(input_parts) < 2:
    suffix = ''
  else:
    suffix = ':' + input_parts[1]
  node_name = input_parts[0]
  return prefix, node_name, suffix

def node_name_from_input(input_name):
  """Strips off ports and other decorations to get the underlying node name."""
  prefix, node_name, suffix = node_name_parts_from_input(input_name)
  return node_name

def canonical_output_name(input_name):
  prefix, node_name, suffix = node_name_parts_from_input(input_name)
  if not suffix:
    suffix = ':0'
  return ''.join([prefix, node_name, suffix])

def dtype_to_tf_string(dtype):
  if type(dtype) == nndct_dtypes.DType:
    tf_dtype = nndct_dtypes.to_tf(dtype)
  elif type(dtype) == tf_dtypes.DType:
    tf_dtype = dtype
  return ".".join(["tf", tf_dtypes._TYPE_TO_STRING[tf_dtype]])

def parse_tf_tensor(tensor):
  """Parse data from given `tensor`."""
  if not isinstance(tensor, tensor_pb2.TensorProto):
    raise TypeError("TensorProto required, but given {}".format(type(tensor)))
  return tensor_util.MakeNdarray(tensor)

def values_from_tf_const(node_def):
  """Extracts the values from a const NodeDef as a numpy ndarray.

  Args:
    node_def: Const NodeDef that has the values we want to access.

  Returns:
    Numpy ndarray containing the values.

  Raises:
    ValueError: If the node isn't a Const.
  """
  if node_def.op != "Const":
    raise ValueError("Node '%s' should be a Const op." % node_def.name)
  input_tensor = node_def.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  return tensor_value

def parse_attr_proto(attr_proto):
  """Convert a list of AttributeProto to a dict, with names as keys."""
  attrs = {}
  for key, value in attr_proto.items():
    attrs[key] = get_attr_proto_value(value)

  return attrs

def get_attr_proto_value(attr_value):
  """Returns the value of the attr of this buf with the given `name`.

  Args:
    attr_value: attrvalue protobuf.

  Returns:
    The value of the attr, as a Python object.

  Raises:
    ValueError: If this op does not have an attr with the given `name`.
  """
  fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

  x = attr_value
  ret = []
  # Treat an empty oneof value as an empty list.
  if not x.WhichOneof("value"):
    return ret
  if x.HasField("list"):
    for f in fields:
      if getattr(x.list, f):
        if f == "type":
          ret += [tf_dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
        else:
          ret += list(getattr(x.list, f))
  else:
    for f in fields:
      if x.HasField(f):
        if f == "type":
          ret = tf_dtypes.as_dtype(getattr(x, f))
        else:
          ret = getattr(x, f)
  return ret

def tf_shape_to_list(shape):
  """Get shape from tensorflow attr 'shape'."""
  dims = None
  try:
    if not shape.unknown_rank:
      dims = [int(d.size) for d in shape.dim]
  except:  # pylint: disable=bare-except
    pass
  return dims

def tf_tensor_shape(tensor):
  shape = []
  try:
    shape = tensor.get_shape().as_list()
  except Exception:  # pylint: disable=broad-except
    shape = None
  return shape

def write_proto(path, message, as_text=False):
  dir_name = os.path.dirname(path)
  generic_utils.mkdir_if_not_exist(dir_name)
  if dir_name:
    os.makedirs(dir_name, exist_ok=True)
  if as_text:
    with open(path, "w") as f:
      f.write(text_format.MessageToString(message))
  else:
    with open(path, "wb") as f:
      f.write(message.SerializeToString())

def write_text_proto(path, message):
  write_proto(path, message, as_text=True)

def write_binary_proto(path, message):
  write_proto(path, message, as_text=False)

def tf_version():
  return tf.__version__

def is_tf_concat(op):
  return op.type in ("Concat", "ConcatV2", "ConcatV3")

def is_tf_const(op):
  return op.type in ["Const", "ConstV2"]

def is_tf_identity(op):
  return op.type == "Identity" or op.type == "IdentityN"

def is_tf_placeholder(op):
  return op.type == "Placeholder"

def is_tf_biasadd(op):
  return op.type == "BiasAdd"
