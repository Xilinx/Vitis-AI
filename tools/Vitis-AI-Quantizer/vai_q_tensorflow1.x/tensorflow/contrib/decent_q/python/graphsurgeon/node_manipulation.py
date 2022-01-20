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

""" create node_def with parameters provided """
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_shape
import numpy as np
# Compatibility with Python 3.X where basestring no longer exists.
try:
    basestring
except NameError:
    basestring = str

def shape(node):
    '''
    Returns the shape of a TensorFlow node.

    Args:
        node (tensorflow.NodeDef): The node whose shape to determine.

    Returns:
        tuple(int)
    '''
    return tuple(int(dim.size) for dim in node.attr["shape"].shape.dim)

# Updates free standing nodes. trt_plugin controls whether UFF parser suffixes are appended to field names.
def update_node(node, name=None, op=None, trt_plugin=False, **kwargs):
    '''
    Updates an existing TensorFlow NodeDef with the specified properties.

    Args:
        name (str): The name of the node.
        op (str): The node's operation.
        trt_plugin (bool): Whether this node should be treated like a TensorRT plugin node.

    Keyword Args:
        dtype (tensorflow.DType): TensorFlow dtype.
        shape (tuple(int)): Iterable container (usually a tuple) describing the shape of a tensor.
        inputs (list(tensorflow.NodeDef) or str): Iterable container (usually a tuple) of input nodes or input node names. Supports mixed-type lists.
        **kwargs (AttrName=Value): Any additional fields that should be present in the node. Currently supports int, float, bool, list(int), list(float), str and NumPy arrays. NumPy arrays can be inserted into the "value" attribute of a node - this can be useful for creating constant nodes equivalent to those created by tensorflow.constant.

    Returns:
        tensorflow.NodeDef
    '''
    node.name = name or node.name
    node.op = op or node.op or node.name
    for key, val in kwargs.items():
        if isinstance(val, dtypes.DType):
            node.attr[key].type = val.as_datatype_enum
        elif key == "shape":
            for val in val:
                node.attr[key].shape.dim.add(size=val)
        elif key == "inputs":
            # Accept either nodes or strings. This method accepts mixed lists too.
            for input_node in val:
                if isinstance(input_node, node_def_pb2.NodeDef):
                    node.input.append(input_node.name)
                elif isinstance(input_node, basestring):
                    node.input.append(input_node)
                else:
                    raise TypeError("Input type unrecognized. Must be a tensorflow.NodeDef or a string.")
        elif isinstance(val, bool):
            node.attr[(key + "_u_bool") if trt_plugin else key].b = val
        elif isinstance(val, int):
            node.attr[(key + "_u_int") if trt_plugin else key].i = val
        elif isinstance(val, float):
            node.attr[(key + "_u_float") if trt_plugin else key].f = val
        elif isinstance(val, list):
            if any(isinstance(n, float) for n in val):
                # If any of the values in the list are floats, the whole list gets promoted to floats.
                node.attr[(key + "_u_flist") if trt_plugin else key].list.f.extend(val)
            elif all(isinstance(n, int) for n in val):
                # For int lists, all values have to be ints - no downcasting should happen.
                node.attr[(key + "_u_ilist") if trt_plugin else key].list.i.extend(val)
            else:
                raise TypeError("Only lists of floats or ints are currently supported.")
        elif isinstance(val, basestring):
            # Workaround for unicode strings.
            try:
                node.attr[(key + "_u_str") if trt_plugin else key].s = str.encode(val)
            except TypeError:
                node.attr[(key + "_u_str") if trt_plugin else key].s = bytes(val)
        elif isinstance(val, np.ndarray):
            node.attr[key].tensor.tensor_shape.CopyFrom(tensor_shape.TensorShape(val.shape).as_proto())
            node.attr[key].tensor.tensor_content = bytes(val)
        else:
            print(key, val)
            raise TypeError("Type: " + str(type(val)) + " unrecognized.")
    # Return a node will all the correct attributes
    return node

def create_node(name, op=None, trt_plugin=False, **kwargs):
    '''
    Creates a free-standing TensorFlow NodeDef with the specified properties.

    Args:
        name (str): The name of the node.
        op (str): The node's operation.

    Keyword Args:
        dtype (tensorflow.DType): TensorFlow dtype.
        shape (tuple(int)): Iterable container (usually a tuple) describing the shape of a tensor.
        inputs (list(tensorflow.NodeDef) or str): Iterable container (usually a tuple) of input nodes or input node names. Supports mixed-type lists.
        **kwargs (AttrName=Value): Any additional fields that should be present in the node. Currently supports int, float, bool, list(int), list(float), str and NumPy arrays. NumPy arrays will be inserted into the "value" attribute of the node - this can be useful for creating constant nodes equivalent to those created by tensorflow.constant.

    Returns:
        tensorflow.NodeDef
    '''
    node = node_def_pb2.NodeDef()
    return update_node(node, name, op, trt_plugin, **kwargs)

def create_plugin_node(name, op=None, **kwargs):
    '''
    Creates a free-standing TensorFlow NodeDef with the specified properties. This is similar to `create_node`,

    Args:
        name (str): The name of the node.
        op (str): The node's operation.
        dtype (tensorflow.DType): TensorFlow dtype.
        shape (tuple(int)): Iterable container (usually a tuple) describing the shape of a tensor.
        inputs (list(tensorflow.NodeDef) or str): Iterable container (usually a tuple) of input nodes or input node names. Supports mixed-type lists.
        **kwargs (AttrName=Value): Any additional fields that should be present in the node. Currently supports int, float, bool, list(int), list(float) and str.

    Returns:
        tensorflow.NodeDef
    '''
    return create_node(name, op, trt_plugin=True, **kwargs)

def extract_numpy_array(node):
    '''
    Given a TensorFlow constant node, returns a NumPy array containing the contents of the node.

    Args:
        node (tensorflow.NodeDef): A Const TensorFlow node.

    Returns:
        numpy.ndarray
    '''
    dtype = dtypes.as_dtype(node.attr["dtype"].type).as_numpy_dtype
    return np.frombuffer(node.attr["value"].tensor.tensor_content, dtype=dtype)
