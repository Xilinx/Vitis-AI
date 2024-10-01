#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
import tempfile
from pathlib import Path

import onnx
import onnx.helper as helper
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, create_calibrator
from .onnx_quantizer import VitisAIONNXQuantizer
from onnxruntime.quantization.quantize import quantize_static as ort_quantize_static
from onnxruntime.quantization.quant_utils import QuantizationMode, QuantType, load_model, QuantFormat
from onnxruntime.quantization.registry import IntegerOpsRegistry, QLinearOpsRegistry

from .calibrate import create_calibrator_power_of_two, PowerOfTwoMethod, RandomDataReader
from .qdq_quantizer import VitisQuantizer, VitisAIQDQQuantizer, VitisDPUQDQQuantizer
from .quant_utils import VAI_DOMAIN, VitisQuantFormat, get_exclude_nodes


def quantize_static(
    model_input,
    model_output,
    calibration_data_reader: CalibrationDataReader,
    quant_format=VitisQuantFormat.FixNeuron,
    input_nodes=[],
    output_nodes=[],
    op_types_to_quantize=[],
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    optimize_model=True,
    use_external_data_format=False,
    calibrate_method=PowerOfTwoMethod.MinMSE,
    execution_providers=['CPUExecutionProvider'],
    use_dpu=False,
    extra_options={},
):

    if calibration_data_reader is None:
        calibration_data_reader = RandomDataReader(model_input)

    if calibrate_method in CalibrationMethod:
        return ort_quantize_static(
            model_input, model_output, calibration_data_reader, quant_format,
            op_types_to_quantize, per_channel, reduce_range, activation_type,
            weight_type, nodes_to_quantize, nodes_to_exclude, optimize_model,
            use_external_data_format, calibrate_method, extra_options)

    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)
    model.opset_import.append(helper.make_operatorsetid(VAI_DOMAIN, 1))

    calib_extra_options_keys = [
        ("ActivationSymmetric", "symmetric"),
        ("CalibMovingAverage", "moving_average"),
        ("CalibMovingAverageConstant", "averaging_constant"),
    ]
    calib_extra_options = {
        key: extra_options.get(name)
        for (name, key) in calib_extra_options_keys
        if name in extra_options
    }

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        calibrator = create_calibrator_power_of_two(
            model,
            op_types_to_quantize,
            augmented_model_path=Path(quant_tmp_dir).joinpath(
                "augmented_model.onnx").as_posix(),
            activation_type=QuantType.QInt8,
            method=calibrate_method,
            use_external_data_format=use_external_data_format,
            execution_providers=execution_providers,
            extra_options=calib_extra_options,
        )
        calibrator.collect_data(calibration_data_reader)
        tensors_range = calibrator.compute_range()
        del calibrator

    if input_nodes or output_nodes:
        if nodes_to_exclude:
            nodes_to_exclude += get_exclude_nodes(model_input, input_nodes, output_nodes)
        else:
            nodes_to_exclude = get_exclude_nodes(model_input, input_nodes, output_nodes)

    if quant_format is QuantFormat.QOperator:
        quantizer = VitisAIONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
    elif quant_format is QuantFormat.QDQ and not use_dpu:
        quantizer = VitisAIQDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
    elif quant_format is QuantFormat.QDQ and use_dpu:
        quantizer = VitisDPUQDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
    elif quant_format is VitisQuantFormat.FixNeuron or quant_format is VitisQuantFormat.QDQ:
        quantizer = VitisQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            quant_format,
            calibrate_method,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

    quantizer.quantize_model()

    quantizer.model.save_model_to_file(model_output, use_external_data_format)
