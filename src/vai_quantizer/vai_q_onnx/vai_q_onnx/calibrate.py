#!/usr/bin/env python
# coding: utf-8
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import abc
import itertools
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper
from onnx import onnx_pb as onnx_proto

import onnxruntime
from onnxruntime.quantization.calibrate import CalibraterBase, CalibrationDataCollector, CalibrationDataReader, MinMaxCalibrater
from onnxruntime.quantization.quant_utils import apply_plot, clone_model_with_shape_infer, load_model, smooth_distribution, QuantType
from .quant_utils import get_pos_overflow, get_bound_and_scale, get_pos_diffs, PowerOfTwoMethod, quantize_data_pof2s


class PowOfTwoCalibrater(CalibraterBase):

    def __init__(
        self,
        model,
        op_types_to_calibrate: Optional[Sequence[str]] = None,
        augmented_model_path="augmented_model.onnx",
        use_external_data_format=False,
        activation_type=QuantType.QInt8,
        method=PowerOfTwoMethod.NonOverflow,
        symmetric=True,
    ):

        super(PowOfTwoCalibrater,
              self).__init__(model, op_types_to_calibrate, augmented_model_path,
                             use_external_data_format)
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(
            output.name for output in self.model.graph.output)
        self.collector = None
        self.method = method
        self.symmetric = symmetric
        self.tensors_to_calibrate = None
        self.activation_type = activation_type
        self.use_external_data_format = use_external_data_format

    def augment_graph(self):
        """
        make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        """
        model = clone_model_with_shape_infer(self.model)

        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(
            model)
        for tensor in self.tensors_to_calibrate:
            if tensor not in self.model_original_outputs:
                model.graph.output.append(value_infos[tensor])
        onnx.save(
            model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        self.augment_model = model

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):

        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(
                self.infer_session.run(None, inputs))

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [
            self.infer_session.get_outputs()[i].name
            for i in range(len(self.intermediate_outputs[0]))
        ]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output))
            for intermediate_output in self.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = dict((i, merged_dict[i])
                                 for i in merged_dict
                                 if i in self.tensors_to_calibrate)

        if not self.collector:
            self.collector = PowOfTwoCollector(
                activation_type=self.activation_type,
                method=self.method,
                symmetric=self.symmetric)
        self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_range(self):
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError(
                "No collector created and can't generate calibration data.")

        return self.collector.compute_collection_result()


class PowOfTwoCollector(CalibrationDataCollector):
    """
    Collecting PowOfTwoCollector quantize for each tensor. NonOverflow and min error method are supported.

    """

    def __init__(self,
                 activation_type=QuantType.QInt8,
                 method=PowerOfTwoMethod.NonOverflow,
                 symmetric=True,
                 bit_width=8):
        self.name_to_arr = {}
        self.method = method
        self.symmetric = symmetric
        self.bit_width = bit_width
        self.activation_qType = (onnx_proto.TensorProto.INT8
                                 if activation_type == QuantType.QInt8 else
                                 onnx_proto.TensorProto.UINT8)

    def collect(self, name_to_arr):

        self.name_to_arr = name_to_arr

        return

    def compute_collection_result(self):
        if not self.name_to_arr or len(self.name_to_arr) == 0:
            raise ValueError(
                "PowerOfTwoMethod has not been collected. Please run collect() first."
            )
        print(
            "Finding optimal threshold for each tensor using {} algorithm ...".
            format(self.method))

        if self.method == PowerOfTwoMethod.MinMSE:
            return self.compute_minmse_range()
        else:
            raise ValueError("Only 'MinMSE' method are supported")

    def compute_minmse_range(self):
        thresholds_dict = {}
        for tensor, data_arr in self.name_to_arr.items():
            d = data_arr[0]
            rmin_mse, rmax_mse, _, _, _ = quantize_data_pof2s(
                d,
                self.activation_qType,
                self.symmetric,
                method=self.method)
            thresholds_dict[tensor] = (rmin_mse, rmax_mse)
        return thresholds_dict


def create_calibrator_power_of_two(
    model,
    op_types_to_calibrate: Optional[Sequence[str]] = None,
    augmented_model_path="augmented_model.onnx",
    activation_type=QuantType.QInt8,
    method=PowerOfTwoMethod.NonOverflow,
    use_external_data_format=False,
    execution_providers=['CPUExecutionProvider'],
    extra_options={},
):

    calibrator = None

    # default settings for min-max algorithm
    method = method
    symmetric = False if "symmetric" not in extra_options else extra_options[
        "symmetric"]
    moving_average = False if "moving_average" not in extra_options else extra_options[
        "moving_average"]
    averaging_constant = 0.01 if "averaging_constant" not in extra_options else extra_options[
        "averaging_constant"]
    if method == PowerOfTwoMethod.NonOverflow:
        calibrator = MinMaxCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
        )
    elif method == PowerOfTwoMethod.MinMSE:
        calibrator = PowOfTwoCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            activation_type=activation_type,
            method=method,
            symmetric=symmetric,
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.execution_providers = execution_providers
        calibrator.create_inference_session()
        return calibrator


class RandomDataReader(CalibrationDataReader):

    def __init__(self, input_model_path):
        self.input_model_path = input_model_path
        self.enum_data_dicts = []
        self.datasize = 1
        self.flag = True

    def _get_input_name(self, input_node):
        input_name = input_node.name
        return input_name

    def _get_input_shape(self, input_node):
        input_shape = []
        if len(input_node.shape):
            for index, shape in enumerate(input_node.shape):
                if index == 0:
                    input_shape.append(self.datasize)
                elif isinstance(shape, int):
                    input_shape.append(shape)
                else:
                    input_shape.append(1) # None shape
        return input_shape

    def _get_input_type(self, input_node):
        input_type = None
        if 'tensor(int8)' in input_node.type:
            input_type = np.int8
        elif 'tensor(uint8)' in input_node.type:
            input_type = np.uint8
        elif 'tensor(int16)' in input_node.type:
            input_type = np.int16
        elif 'tensor(uint16)' in input_node.type:
            input_type = np.uint16
        elif 'tensor(int32)' in input_node.type:
            input_type = np.int32
        elif 'tensor(uint32)' in input_node.type:
            input_type = np.uint32
        elif 'tensor(int64)' in input_node.type:
            input_type = np.int64
        elif 'tensor(uint64)' in input_node.type:
            input_type = np.uint64
        elif 'tensor(float16)' in input_node.type:
            input_type = np.float16
        elif 'tensor(float)' in input_node.type:
            input_type = np.float32
        elif 'tensor(double)' in input_node.type:
            input_type = np.double
        elif 'tensor(bool)' in input_node.type:
            input_type = np.bool
        return input_type

    def get_next(self):
        if self.flag:
            self.flag = False
            session = onnxruntime.InferenceSession(
                    self.input_model_path, providers=['CPUExecutionProvider'])
            enum_data = {}
            for input_node in session.get_inputs():
                input_name = self._get_input_name(input_node)
                input_shape= self._get_input_shape(input_node)
                input_type = self._get_input_type(input_node)

                if 'tensor(string)' in input_node.type:
                    input_data = '0' if input_shape == [] else np.chararray(tuple(input_shape))
                elif input_type is not None:
                    input_data = 0 if input_shape == [] else np.zeros(
                                                     tuple(input_shape), dtype=input_type)
                else:
                    raise ValueError("Unsupported input name-{} shape-{} type-{} ".format(
                                     input_node.name, input_node.shape, input_node.type))
                enum_data[input_name] = input_data
            self.enum_data_dicts = iter([enum_data])
        return next(self.enum_data_dicts, None)
