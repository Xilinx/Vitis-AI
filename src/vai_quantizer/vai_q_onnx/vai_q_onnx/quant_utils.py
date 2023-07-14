#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import logging
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np
import onnx
import os
import sys
from onnx import external_data_helper
from onnx import onnx_pb as onnx_proto
import onnx.helper as helper
from onnx import shape_inference, TensorProto
import onnxruntime as ort
from onnxruntime.quantization.quant_utils import get_qmin_qmax_for_qType, quantize_nparray
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path as _lib_path
from vai_q_onnx.operators.vai_ops.qdq_ops import vai_dquantize

VAI_DOMAIN = "ai.onnx.contrib"

FIX_OP_NAME = "FixNeuron"


class PowerOfTwoMethod(Enum):
    NonOverflow = 0
    MinMSE = 1


class VitisQuantFormat(Enum):
    QDQ = 2
    FixNeuron = 3

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(format):
        try:
            return VitisQuantFormat[format]
        except KeyError:
            raise ValueError()


def get_nonoverflow(f_tensor):
    x_min = f_tensor.min()
    x_max = f_tensor.max()
    return x_min, x_max


def get_pos_overflow(f_tensor, bit_width=8):
    x_min = f_tensor.min()
    x_max = f_tensor.max()
    #  Use 0.5 as a guard
    lower_bound = -2**(bit_width - 1) - 0.5
    upper_bound = 2**(bit_width - 1) - 0.5
    scale = max(x_min / lower_bound, x_max / upper_bound)
    if scale == 0:
        # Set pos to 127(max of uint8) for all zero
        return 127
    else:
        pos = np.floor(np.log2(1 / scale))
        # Set pos to -128 ~ 127
        pos = 127 if pos > 127 else pos
        pos = -128 if pos < -128 else pos

    return pos


def get_pos_diffs(f_tensor, bit_width=8, pos_range=5):

    pos_overflow = get_pos_overflow(f_tensor, bit_width)
    if pos_overflow == 127:
        # Set pos to 127(max of uint8) for all zero
        return 127

    pos_diffs = pos_overflow
    diff_min = float('inf')
    for i in range(pos_range):
        tmp_pos = pos_overflow + i
        q_tensor = vitis_quantize(f_tensor, tmp_pos, bit_width=bit_width)
        diff = np.sum((q_tensor - f_tensor)**2)
        if diff < diff_min:
            diff_min = diff
            pos_diffs = tmp_pos
    return pos_diffs


def vitis_quantize(f_tensor, pos, bit_width=8):
    scale, lower_bound, upper_bound = get_bound_and_scale(pos)
    q_tensor = np.round(f_tensor / scale) * scale
    q_tensor = lower_bound * (q_tensor < lower_bound) \
                + q_tensor * (q_tensor >= lower_bound) # it should be >=, not > (xieyi)
    q_tensor = upper_bound * (q_tensor > upper_bound) \
                + q_tensor * (q_tensor <= upper_bound) # it should be <=, not < (xieyi)
    return q_tensor


def get_bound_and_scale(pos, bit_width=8):
    scale = np.power(2., -pos)
    lower_bound = -np.power(2, bit_width - 1) * scale
    upper_bound = np.power(2, bit_width - 1) * scale - scale
    return scale, lower_bound, upper_bound


def vitis_quantize_data(data, bit_width=8, method="overflow"):

    if method == "overflow":
        pos = get_pos_overflow(data)
    elif method == "diffs":
        pos = get_pos_diffs(data)

    scale, lower_bound, upper_bound = get_bound_and_scale(pos)
    zero_point = 0
    quantized_data = vitis_quantize(data, pos=pos)
    return lower_bound, upper_bound, zero_point, scale, quantized_data


def infer_shape(model):
    inferred_onnx_model = shape_inference.infer_shapes(model)
    return inferred_onnx_model


def get_datatype_shape(tensor):
    name = tensor.name
    elem_type_num = tensor.type.tensor_type.elem_type
    data_type = TensorProto.DataType.Name(elem_type_num).lower()
    data_type = data_type if data_type != 'float' else 'float32'
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    shape = [dims[i].dim_value if dims[i].dim_value else -1 for i in range(n)]
    return (data_type, shape)


def dump_model(model, input_data, dump_float=True, output_dir='./dump_results'):
    """
    Dump model weights and outputs. e.g:
    model_path = 'resnet50-v1-13.quant.onnx'
    model = onnx.load(model_path)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    dump_model(model, input_data)
    """
    tmp_dump_model = "./tmp_dump_model.onnx"

    # modify_output_nodes
    fn_node_pos = {}
    for n in model.graph.node:
        if n.op_type == "FixNeuron":
            fn_node_pos[n.output[0]] = 2**int(n.attribute[1].s)
    node_output = []

    for o in model.graph.output:
        node_output.append(o.name)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            node_output.append(output)
    onnx.save(model, tmp_dump_model)

    so = ort.SessionOptions()
    so.register_custom_ops_library(_lib_path())

    sess = ort.InferenceSession(tmp_dump_model, so)
    input_name = sess.get_inputs()[0].name

    results_outputs = sess.run(None, {input_name: input_data})

    print("Dumping activations...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for node, res in zip(node_output, results_outputs):
        filename = os.path.join(output_dir, node)
        res = res.flatten()
        if node in fn_node_pos:
            res = np.round(res * fn_node_pos[node])
            res = res.clip(-128, 127)
            res.astype(np.int8).tofile(filename + ".bin")
            np.savetxt(filename + ".txt",
                       res.astype(np.int8),
                       fmt="%s",
                       delimiter=",")
        if dump_float:
            res.tofile(filename + '_float.bin')
            np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")
    os.remove(tmp_dump_model)


def get_annotate_output_name(model):
    """
    Traverse the floating-point model to find the names of all conv outputs.
    """
    annotate_output_name_list = []
    annotate_op_list = [
        "Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool"
    ]

    for node in model.graph.node:
        if node.op_type in annotate_op_list:
            annotate_output_name_list.append(node.output[0])
    return annotate_output_name_list


def get_relu_name(model, input_list):
    """
    Find the corresponding pairs that form conv-relu through the output of conv, and return two dictionaries: conv-relu and relu-conv
    """
    relu_to_conv_output = {}
    for node in model.graph.node:
        if node.op_type == "Relu" and node.input[0] in input_list:
            relu_to_conv_output[node.name] = node.input[0]

    return relu_to_conv_output


def get_qdq_to_remove(model, relu_input):
    """
    Return two lists, one for 'dq' and one for 'q'.
    """
    q_nodes_to_remove = []
    dq_nodes_to_remove = []
    q_nodes_output_to_remove = []
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[
                0] in relu_input.values():
            q_nodes_to_remove.append(node)
            q_nodes_output_to_remove.append(node.output[0])
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and node.input[
                0] in q_nodes_output_to_remove:
            dq_nodes_to_remove.append(node)
    return dq_nodes_to_remove, q_nodes_to_remove


def remove_nodes(model, nodes_list):
    """
    Delete nodes according to the nodes in the list
    """
    for node in nodes_list:
        model.graph.node.remove(node)
    return model


def convert_relu_input_to_annotate_output(model, relu_to_conv_output):
    """
    Modify the input of ReLU to the output of annotate op, and delete QDQ
    """
    for node in model.graph.node:
        if node.name in relu_to_conv_output.keys():
            node.input[0] = relu_to_conv_output[node.name]
    return model


def scale2pos(scale):
    """
    Obtain the fixed-point position corresponding to the scale.
    """
    if scale == 0:
        return 127
    return int(np.rint(-np.log2(scale)))


def pos2scale(pos):
    """
    Obtain the scale corresponding to the fixed-point position.
    """
    return float(np.power(2.0, -pos))


def compute_scale_zp_pof2s(rmin, rmax, qmin, qmax, symmetric=False):
    """Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :return: zero and scale [z, s]

    """

    if qmin > 0 or qmax < 0:
        raise ValueError(
            f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}"
        )

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    if rmin == float("-inf"):
        rmin = -sys.float_info.max / 2
    if rmax == float("inf"):
        rmax = sys.float_info.max / 2

    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = -absmax
        rmax = +absmax

    scale = (rmax - rmin) / float(qmax - qmin)
    pos = scale2pos(scale)
    pof2_scale = pos2scale(pos)

    if pof2_scale < np.finfo(np.float32).tiny:
        pof2_scale = 1.0
        zero_point = 0
    else:
        zero_point = round(qmin - rmin / pof2_scale)
    if symmetric:
        zero_point = 0
    return [zero_point, pof2_scale]


def quantize_zero_point(rmin, qmin, qmax, symmetric, scale):
    if qmin > 0 or qmax < 0:
        raise ValueError(
            f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}"
        )

    rmin = min(rmin, 0)

    if symmetric:
        return 0

    pof2_scale = scale

    if pof2_scale < np.finfo(np.float32).tiny:
        pof2_scale = 1.0
        zero_point = 0
    else:
        zero_point = round(qmin - rmin / pof2_scale)

    return zero_point


def dequantize_data(data, scale, zero_point):
    data = data.astype(np.float32)
    deq_arr = (data - zero_point) * scale
    return deq_arr.astype(np.float32)


def quantize_data_pof2s(data,
                        qType,
                        symmetric,
                        reduce_range=False,
                        method=PowerOfTwoMethod.NonOverflow,
                        pos_range=5):
    """
    :param data: data to quantize
    :param qType: data type to quantize to. Supported types UINT8 and INT8
    :param symmetric: whether symmetric quantization is used or not. This is applied to INT8.
    :return: minimum, maximum, zero point, scale, and quantized weights

    To pack weights, we compute a linear transformation

    - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
    - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
        `m = max(abs(rmin), abs(rmax))`

    and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation

    :math:`r = S(q-z)`, where

    - *r*: real original value
    - *q*: quantized value
    - *S*: scale
    - *z*: zero point
    """

    rmin = 0
    rmax = 0
    zero_point = 0
    scale = 1.0
    if isinstance(data, np.ndarray):
        rmin = data.min()
        rmax = data.max()

    elif isinstance(data, list) and len(data):
        rmin = min(data)
        rmax = max(data)

    qmin, qmax = get_qmin_qmax_for_qType(qType,
                                         reduce_range,
                                         symmetric=symmetric)
    zero_point, scale = compute_scale_zp_pof2s(rmin, rmax, qmin, qmax,
                                               symmetric)

    quantized_data = quantize_nparray(qType, np.asarray(data), scale,
                                      zero_point)
    if method == PowerOfTwoMethod.NonOverflow:
        return rmin, rmax, zero_point, scale, quantized_data
    elif method == PowerOfTwoMethod.MinMSE:
        scale_mse = scale
        zp_mse = zero_point
        quantized_data_mse = quantized_data
        diff_min = float("inf")
        for i in range(pos_range):
            new_scale = pos2scale(scale2pos(scale) + i)
            rmin = (qmin - zero_point) * new_scale
            new_zero_point = quantize_zero_point(rmin, qmin, qmax, symmetric,
                                                 new_scale)

            new_quantized_data = quantize_nparray(qType, np.asarray(data),
                                                  new_scale, new_zero_point)
            diff = np.sum((
                dequantize_data(new_quantized_data, new_scale, new_zero_point) -
                np.asarray(data))**2)
            if diff < diff_min:
                diff_min = diff
                scale_mse = new_scale
                zp_mse = new_zero_point
                quantized_data_mse = new_quantized_data

        rmin_mse = (qmin - zp_mse) * scale_mse
        rmax_mse = (qmax - zp_mse) * scale_mse
        return rmin_mse, rmax_mse, zp_mse, scale_mse, quantized_data_mse


def get_exclude_nodes(model_path, input_nodes, output_nodes):
    """
    Return the nodes to be excluded based on the given input and output nodes.
    """
    model = onnx.load(model_path)

    name_list = []
    exclude_nodes = []

    for node in model.graph.node:
        name_list.append(node.name)

    if input_nodes:
        for name in input_nodes:
            input_node_name = name
            index = name_list.index(input_node_name)
            exclude_nodes_i = name_list[:index]
            exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_i))
            exclude_nodes = list(set(exclude_nodes) - set(input_nodes))

    if output_nodes:
        for name in output_nodes:
            output_node_name = name
            index = name_list.index(output_node_name) + 1
            exclude_nodes_o = name_list[index:]
            exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_o))
            exclude_nodes = list(set(exclude_nodes) - set(output_nodes))

    return exclude_nodes
