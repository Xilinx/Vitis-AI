#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np

import logging
from enum import Enum

import onnx
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto

from onnxruntime.quantization.qdq_quantizer import QDQQuantizer, QDQQuantTensorType, QDQTensorQuantInfo
from onnxruntime.quantization.quant_utils import (
    QuantFormat,
    DEQUANT_OP_NAME,
    QUANT_OP_NAME,
    QuantizedValue,
    QuantizedValueType,
    __producer__,
    __version__,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_input_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    find_by_name,
)

from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer, tensor_proto_to_array
from .quant_utils import (
    VitisQuantFormat,
    FIX_OP_NAME,
    VAI_DOMAIN,
    vitis_quantize_data,
    get_annotate_output_name,
    get_relu_name,
    get_qdq_to_remove,
    remove_nodes,
    convert_relu_input_to_annotate_output,
    compute_scale_zp_pof2s,
    quantize_data_pof2s,
    PowerOfTwoMethod,
)
from .registry import CreateQDQQuantizer
from .refine import adjust_quantize_info
from .onnx_quantizer import VitisAIONNXQuantizer


class VitisQuantizer(QDQQuantizer):

    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        quant_format,
        calibrate_method,
        static,
        weight_qType,
        input_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        QDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            input_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}

        self.quant_format = quant_format  # QuantFormat.Value
        self.calibrate_method = calibrate_method  # CalibrationMethod.Value
        self.add_qdq_pair_to_weight = True
        self.is_weight_symmetric = True
        self.is_activation_symmetric = True

    def vitis_quantize_initializer(self,
                                   weight,
                                   bit_width=8,
                                   keep_float_weight=False):

        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name = weight.name + "_quantized"
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = tensor_proto_to_array(weight)
        _, _, zero_point, scale, q_weight_data = vitis_quantize_data(
            weight_data.flatten(), bit_width, method="diffs")
        scale_initializer = onnx.helper.make_tensor(
            scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
        zero_initializer = onnx.helper.make_tensor(zp_name,
                                                   onnx_proto.TensorProto.INT8,
                                                   [], [zero_point])
        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            q_weight_data = np.asarray(
                q_weight_data,
                dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                    onnx_proto.TensorProto.FLOAT]).reshape(weight.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(
                q_weight_data, weight.name)
            w_init = find_by_name(weight.name, self.model.initializer())
            w_init.CopyFrom(q_weight_initializer)

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def quantize_model(self):

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(
                            node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        self._quantize_bias_tensors()
        self._quantize_refine()

        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (output_name in self.quantization_params.keys() and len(
                self.model.input_name_to_nodes()[upstream_output_name]) == 1 and
                not self.model.is_graph_output(upstream_output_name) and
                not self.model.is_graph_input(upstream_output_name)):
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def _create_fn_nodes(self,
                         q_input,
                         dq_output,
                         dequant_node_name,
                         scale_name,
                         zp_name,
                         axis=None):
        """
        create fix_neuron node
        """
        fix_neuron_node = onnx.helper.make_node(
            FIX_OP_NAME,
            [q_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        bit_width = onnx.helper.make_attribute("bit_width", "8")
        fix_neuron_node.attribute.append(bit_width)

        scale = find_by_name(scale_name, self.model.initializer())
        scale = scale.float_data[0]
        pos = int(np.rint(-np.log2(scale)))
        pos_attr = onnx.helper.make_attribute("pos", str(pos))
        fix_neuron_node.attribute.append(pos_attr)

        self.model.add_nodes([fix_neuron_node])

    def _create_qdq_nodes(self,
                          q_input,
                          q_output,
                          quant_node_name,
                          dq_input,
                          dq_output,
                          dequant_node_name,
                          scale_name,
                          zp_name,
                          axis=None):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        bit_width = onnx.helper.make_attribute("bit_width", "8")

        scale = find_by_name(scale_name, self.model.initializer())
        scale = scale.float_data[0]
        pos = int(np.rint(-np.log2(scale)))
        pos_attr = onnx.helper.make_attribute("pos", str(pos))

        qlinear_node.attribute.append(bit_width)
        qlinear_node.attribute.append(pos_attr)
        dequant_node.attribute.append(bit_width)
        dequant_node.attribute.append(pos_attr)
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_fn_pair_for_weight(self, weight_proto, axis=None):
        weight_name = weight_proto.name

        q_weight_name, zp_name, scale_name = self.vitis_quantize_initializer(
            weight_proto, self.weight_qType, keep_float_weight=False)

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name,
                                              weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            if self.quant_format == VitisQuantFormat.FixNeuron:
                self._create_fn_nodes(
                    weight_name,
                    weight_dequant_output,
                    add_dequant_suffix(weight_name),
                    scale_name,
                    zp_name,
                    axis,
                )
            elif self.quant_format == VitisQuantFormat.QDQ:
                self._create_qdq_nodes(
                    weight_name,
                    weight_quant_output,
                    add_quant_suffix(weight_name),
                    weight_quant_output,
                    weight_dequant_output,
                    add_dequant_suffix(weight_name),
                    scale_name,
                    zp_name,
                    axis,
                )

        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
                domain=VAI_DOMAIN,
            )
            bit_width = onnx.helper.make_attribute("bit_width", "8")
            dequant_node.attribute.append(bit_width)
            self.model.add_node(dequant_node)

    def _add_fn_pair_for_activation(self, tensor_name, scale_name, zp_name):
        if (self.dedicated_qdq_pair and
                tensor_name in self.tensor_to_its_receiving_nodes and
                len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1):
            num_dedicated_qdq_pair = len(
                self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(
                    tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(
                    tensor_name) + postfix
                if self.quant_format == VitisQuantFormat.FixNeuron:
                    self._create_fn_nodes(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        add_dequant_suffix(tensor_name),
                        scale_name,
                        zp_name,
                    )
                elif self.quant_format == VitisQuantFormat.FixNeuron:
                    self._create_qdq_nodes(
                        weight_name,
                        weight_quant_output,
                        add_quant_suffix(weight_name),
                        weight_quant_output,
                        weight_dequant_output,
                        add_dequant_suffix(weight_name),
                        scale_name,
                        zp_name,
                        axis,
                    )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(
                    node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            if self.quant_format == VitisQuantFormat.FixNeuron:
                self._create_fn_nodes(
                    q_input,
                    dq_output,
                    add_dequant_suffix(tensor_name),
                    scale_name,
                    zp_name,
                )
            elif self.quant_format == VitisQuantFormat.QDQ:
                self._create_qdq_nodes(
                    q_input,
                    add_quant_output_suffix(tensor_name),
                    add_quant_suffix(tensor_name),
                    add_quant_output_suffix(tensor_name),
                    dq_output,
                    add_dequant_suffix(tensor_name),
                    scale_name,
                    zp_name,
                )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():

            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name,
                                           self.model.initializer())
                if initializer:
                    self._add_fn_pair_for_weight(initializer, tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp)

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    self._add_fn_pair_for_activation(tensor_name, scale_name,
                                                     zp_name)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self):
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy(
            ).items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[
                        tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name,
                                               self.model.initializer())
                    if initializer is not None:
                        raise ValueError(
                            "Quantization parameter shared mode is not supported for weight yet"
                        )
                    self._add_fn_pair_for_activation(tensor_name,
                                                     quantized_value.scale_name,
                                                     quantized_value.zp_name)

    def _quantize_bias_tensors(self):
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(
                find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [
                quant_value.q_name, quant_value.scale_name, quant_value.zp_name
            ]
            node_name = add_dequant_suffix(bias_name)
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                    domain=VAI_DOMAIN,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    inputs,
                    [bias_name],
                    node_name,
                    domain=VAI_DOMAIN,
                )
            bit_width = onnx.helper.make_attribute("bit_width", "8")
            dequant_node.attribute.append(bit_width)
            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self,
                             bias_name,
                             input_name,
                             weight_name,
                             beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize[bias_name] = QDQTensorQuantInfo()
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def _adjust_model_scale(self):
        for node in self.model.model.graph.node:
            if node.op_type == "DequantizeLinear" or node.op_type == "QuantizeLinear":
                for attr in node.attribute:
                    if attr.name == "pos":
                        pos = int(attr.s)
                new_scale = float(np.power(2., -pos))
                for i in self.model.model.graph.initializer:
                    if i.name == node.input[1]:
                        if i.float_data[0] != new_scale:
                            i.float_data = [new_scale]

    def _quantize_refine(self):
        self.model = adjust_quantize_info(self.model,
                                          adjust_vitis_sigmoid=True,
                                          adjust_shift_cut=True,
                                          adjust_shift_bias=True,
                                          adjust_shift_read=True,
                                          adjust_shift_write=True,
                                          align_concat=True,
                                          align_pool=True)
        if self.quant_format == VitisQuantFormat.QDQ:
            self._adjust_model_scale()

class VitisDPUQDQQuantizer(QDQQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        input_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
        need_layer_fusing=False,
        extra_options=None,
    ):
        QDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            input_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.calibrate_method = calibrate_method
        self.need_layer_fusing = need_layer_fusing

        if per_channel:
            logging.error("per_channel is not supported in PowerOfTwoMethod calibrate_method.")

        # In PowerOfTwoMethod calibrate_method, QDQ should always appear as a pair.
        # Therefore, we need to add qdq pair to weight.
        if "AddQDQPairToWeight" in self.extra_options and not self.extra_options["AddQDQPairToWeight"]:
            logging.error("AddQDQPairToWeight should be True in PowerOfTwoMethod calibrate_method.")
        self.add_qdq_pair_to_weight = True

        # In PowerOfTwoMethod calibrate_method, QDQ should always set WeightSymmetric as True.
        if "WeightSymmetric" in self.extra_options and not self.extra_options["WeightSymmetric"]:
            logging.error("WeightSymmetric should be True in PowerOfTwoMethod calibrate_method.")
        self.is_weight_symmetric = True

        # In PowerOfTwoMethod calibrate_method, QDQ should always always set ActivationSymmetric as True.
        if "ActivationSymmetric" in self.extra_options and not self.extra_options["ActivationSymmetric"]:
            logging.error("ActivationSymmetric should be True in PowerOfTwoMethod calibrate_method.")
        self.is_activation_symmetric = True

    def vitis_quantize_initializer(self, weight, bit_width=8, keep_float_weight=False):

        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name = weight.name + "_quantized"
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = tensor_proto_to_array(weight)
        _, _, zero_point, scale, q_weight_data = vitis_quantize_data(
            weight_data.flatten(), bit_width, method=self.calibrate_method
        )
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
        zero_initializer = onnx.helper.make_tensor(zp_name, onnx_proto.TensorProto.INT8, [], [zero_point])
        self.model.initializer().extend([scale_initializer, zero_initializer])

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def quantize_model(self):

        self.tensor_info = {}
        model = self.model.model
        annotate_output_name_list = get_annotate_output_name(model)
        relu_to_conv_output = get_relu_name(model, annotate_output_name_list)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        dq_nodes_to_remove, q_nodes_to_remove = get_qdq_to_remove(
            model, relu_to_conv_output)
        convert_relu_input_to_annotate_output(model, relu_to_conv_output)
        if self.need_layer_fusing:
            model = remove_nodes(model, dq_nodes_to_remove)
            model = remove_nodes(model, q_nodes_to_remove)
        self._quantize_refine()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        model.producer_name = __producer__
        model.producer_version = __version__

        return model

    def _add_qdq_pair_for_initializer(self, weight_proto, tensor_type, axis=None):
        weight_name = weight_proto.name
        q_weight_name, zp_name, scale_name = self.vitis_quantize_initializer(
            weight_proto, self.weight_qType, keep_float_weight=True
        )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )

            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                # Use int8 quantization for bias as well as weights.
                self.tensors_to_quantize[bias_name] = QDQTensorQuantInfo()
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def _quantize_refine(self):
        self.model = adjust_quantize_info(
            self.model,
            adjust_vitis_sigmoid=True,
            adjust_shift_cut=True,
            adjust_shift_bias=True,
            adjust_shift_read=True,
            adjust_shift_write=True,
            align_concat=True,
            align_pool=True,
        )

class VitisAIQDQQuantizer(VitisAIONNXQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        ONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.bias_to_quantize = []

        self.nodes_to_remove = []

        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, Matmul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        self.op_types_to_exclude_output_quantization = (
            []
            if "OpTypesToExcludeOutputQuantization" not in extra_options
            else extra_options["OpTypesToExcludeOutputQuantization"]
        )

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (
            False if "AddQDQPairToWeight" not in extra_options else extra_options["AddQDQPairToWeight"]
        )

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can’t be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = (
            False if "DedicatedQDQPair" not in extra_options else extra_options["DedicatedQDQPair"]
        )
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = (
            {}
            if "QDQOpTypePerChannelSupportToAxis" not in extra_options
            else extra_options["QDQOpTypePerChannelSupportToAxis"]
        )

    def _is_tensor_quantizable(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                return True
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                return True
        else:
            logging.warning(
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".format(
                    tensor_name
                )
            )

        return False

    def __quantize_tensor(self, tensor_name, quant_sharing_param=None, tensor_type=QDQQuantTensorType.ACTIVATION):
        """
        Quantize tensors. If quant_param_tensor is not None, tensor with name tensor_name will be quantized with same
        quantization parameters as tensor quant_param_tensor

        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter
            tensor_type: QDQQuantTensorType default ACTIVATION
        """
        if self._is_tensor_quantizable(tensor_name):
            if quant_sharing_param:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=tensor_type, quant_para_provider=quant_sharing_param
                )
            elif tensor_name not in self.tensors_to_quantize:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=tensor_type)

    def quantize_activation_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Activation Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter

        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.ACTIVATION)

    def quantize_weight_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Weight Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter

        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.WEIGHT)

    def quantize_weight_tensor_per_channel(self, tensor_name, axis):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=QDQQuantTensorType.WEIGHT, axis=axis
                )
        else:
            logging.warning(
                "only support per-channel quantization on weight. Tensor: {} is not quantized.".format(tensor_name)
            )

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        self._quantize_bias_tensors()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (
            output_name in self.quantization_params.keys()
            and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
            and not self.model.is_graph_output(upstream_output_name)
            and not self.model.is_graph_input(upstream_output_name)
        ):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def _create_qdq_nodes(
        self, q_input, q_output, quant_node_name, dq_input, dq_output, dequant_node_name, scale_name, zp_name, axis=None
    ):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_qdq_pair_for_initializer(self, weight_proto, tensor_type, axis=None):
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name, onnx_proto.TensorProto.INT8, axis, keep_float_weight=self.add_qdq_pair_to_weight
            )
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT else self.activation_qType,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )
            self.model.add_node(dequant_node)

    def _add_qdq_pair_for_activation(self, tensor_name, scale_name, zp_name):
        if (
            self.dedicated_qdq_pair
            and tensor_name in self.tensor_to_its_receiving_nodes
            and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1
        ):
            num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(tensor_name) + postfix
                quant_node_name_postfix = add_quant_suffix(tensor_name) + postfix
                dequant_node_name_postfix = add_dequant_suffix(tensor_name) + postfix
                self._create_qdq_nodes(
                    tensor_name,
                    tensor_name_quant_output_postfix,
                    quant_node_name_postfix,
                    tensor_name_quant_output_postfix,
                    tensor_name_dequant_output_postfix,
                    dequant_node_name_postfix,
                    scale_name,
                    zp_name,
                )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            self._create_qdq_nodes(
                q_input,
                add_quant_output_suffix(tensor_name),
                add_quant_suffix(tensor_name),
                add_quant_output_suffix(tensor_name),
                dq_output,
                add_dequant_suffix(tensor_name),
                scale_name,
                zp_name,
            )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer, tensor_info.tensor_type, tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp
                    )

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    self._add_qdq_pair_for_activation(tensor_name, scale_name, zp_name)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self):
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name, self.model.initializer())
                    if initializer is not None:
                        raise ValueError("Quantization parameter shared mode is not supported for weight yet")
                    self._add_qdq_pair_for_activation(tensor_name, quantized_value.scale_name, quantized_value.zp_name)

    def _quantize_bias_tensors(self):
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            node_name = add_dequant_suffix(bias_name)
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                )
            self.model.add_node(dequant_node)

    def is_tensor_quantized(self, tensor_name):
        return tensor_name in self.tensors_to_quantize or tensor_name in self.bias_to_quantize
