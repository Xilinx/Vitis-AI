#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef


@onnx_op(
    op_type="VaiQuantize",  # domain='vai.quantize',
    inputs=[
        PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8
    ],
    outputs=[PyCustomOpDef.dt_int8],
    attrs=["bit_width"])
def vai_quantize(x, scale, zero_point, **kwargs):
    # The custom op implementation.
    bit_width = int(kwargs["bit_width"])
    q_min = -2**(bit_width - 1)
    q_max = 2**(bit_width - 1) - 1
    result = np.round(x / scale) + zero_point
    result = np.clip(result, q_min, q_max)
    return result


@onnx_op(
    op_type="VaiDeQuantize",  # domain='vai.quantize',
    inputs=[
        PyCustomOpDef.dt_int8, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8
    ],
    outputs=[PyCustomOpDef.dt_float],
    attrs=["bit_width"])
def vai_dquantize(x, scale, zero_point, **kwargs):
    bit_width = int(kwargs["bit_width"])
    q_min = -2**(bit_width - 1)
    q_max = 2**(bit_width - 1) - 1
    result = np.clip(x, q_min, q_max)
    result = (result - zero_point) * scale
    return result


@onnx_op(
    op_type="FixNeuron",  # domain='vai.quantize',
    # inputs=[PyCustomOpDef.dt_float], outputs=[PyCustomOpDef.dt_float], attrs=["bit_width"])
    inputs=[
        PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int8
    ],
    outputs=[PyCustomOpDef.dt_float],
    attrs=["bit_width"])
def vai_dquantize(x, scale, zero_point, **kwargs):
    bit_width = int(kwargs["bit_width"])
    q_min = -2**(bit_width - 1)
    q_max = 2**(bit_width - 1) - 1
    result = np.round(x / scale) + zero_point
    result = np.clip(result, q_min, q_max)
    result = (result - zero_point) * scale
    return result
