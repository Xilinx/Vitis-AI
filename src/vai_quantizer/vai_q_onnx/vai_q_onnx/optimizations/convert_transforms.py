#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Graph transforms for the conversion of onnx models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import onnx
from onnx import helper

from vai_q_onnx.graph_transformations import model_transformer
from vai_q_onnx.graph_transformations import transforms
from vai_q_onnx.utils import model_utils

Transform = transforms.Transform
OpTypePattern = transforms.OpTypePattern
NodeTree = transforms.NodeTree
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"


class ConvQDQToQOPTransform(transforms.Transform):

    def __init__(self):
        super().__init__()
        self.allow_multi_consumers = True

    def pattern(self):
        return OpTypePattern(
            'QuantizeLinear',
            [
                OpTypePattern(
                    'Conv',
                    [
                        OpTypePattern(
                            'DequantizeLinear',
                            [
                                OpTypePattern('QuantizeLinear'),  # x
                                OpTypePattern('.*'),  # x scale
                                OpTypePattern('.*'),  # x zero_point
                            ]),
                        OpTypePattern(
                            'DequantizeLinear',
                            [
                                OpTypePattern('.*'),  # w
                                OpTypePattern('.*'),  # w scale
                                OpTypePattern('.*'),  # w zero_point
                            ]),
                        OpTypePattern(
                            'DequantizeLinear',  # conv bias
                            [
                                OpTypePattern('.*'),  # b
                                OpTypePattern('.*'),  # b scale
                                OpTypePattern('.*'),  # b zero_point
                            ]),
                    ]),
                OpTypePattern('.*'),  # y scale
                OpTypePattern('.*'),  # y zero_point
            ])

    def replacement(self, match_node):
        x_node = match_node.input_nodes[0].input_nodes[0].input_nodes[0]
        x_scale_node = match_node.input_nodes[0].input_nodes[0].input_nodes[1]
        x_zero_point_node = match_node.input_nodes[0].input_nodes[
            0].input_nodes[2]

        w_node = match_node.input_nodes[0].input_nodes[1].input_nodes[0]
        w_scale_node = match_node.input_nodes[0].input_nodes[1].input_nodes[1]
        w_zero_point_node = match_node.input_nodes[0].input_nodes[
            1].input_nodes[2]

        b_node = match_node.input_nodes[0].input_nodes[2].input_nodes[0]
        b_scale_node = match_node.input_nodes[0].input_nodes[2].input_nodes[1]
        b_zero_point_node = match_node.input_nodes[0].input_nodes[
            2].input_nodes[2]
        conv_node = match_node.input_nodes[0]
        print('Convert conv: ', conv_node.node.name)
        y_scale_node = match_node.input_nodes[1]
        y_zero_point_node = match_node.input_nodes[2]

        # Make the bias initializer, rescale bias to align with weights
        #   rescale_b = ( (b * b_scale + b_zero_point) - w_zero_point ) / w_scale
        b_value = model_utils.get_tensor_value(b_node.node)
        b_scale_value = model_utils.get_tensor_value(b_scale_node.node)
        b_zero_point_value = model_utils.get_tensor_value(b_scale_node.node)
        w_scale_value = model_utils.get_tensor_value(w_scale_node.node)
        w_zero_point_value = model_utils.get_tensor_value(w_scale_node.node)
        rescale_b_value = ((b_value * b_scale_value + b_zero_point_value) -
                           w_zero_point_value) / w_scale_value
        rescale_b = model_utils.generate_initializer(rescale_b_value,
                                                     dtype=np.int32,
                                                     name=b_node.node.name)
        rescale_b_node = NodeTree(rescale_b,
                                  weights=None,
                                  input_nodes=[],
                                  metadata=None)

        # Make the QLinearConv node
        qlinear_conv = helper.make_node(
            'QLinearConv',
            inputs=[
                x_node.node.output[0],  # x
                x_scale_node.node.name,  # x_scale
                x_zero_point_node.node.name,  # x_zero_point
                w_node.node.name,  # w
                w_scale_node.node.name,  # w_scale
                w_zero_point_node.node.name,  # w_zero_point
                y_scale_node.node.name,  # y_scale
                y_zero_point_node.node.name,  # y_zero_point
                rescale_b_node.node.name,  # b
            ],
            outputs=[match_node.node.output[0]],
            name=conv_node.node.name)
        qlinear_conv_node = NodeTree(qlinear_conv,
                                     weights=None,
                                     input_nodes=[
                                         x_node,
                                         x_scale_node,
                                         x_zero_point_node,
                                         w_node,
                                         w_scale_node,
                                         w_zero_point_node,
                                         y_scale_node,
                                         y_zero_point_node,
                                         rescale_b_node,
                                     ],
                                     metadata=None)

        return qlinear_conv_node


class MatMulQDQToQOPTransform(transforms.Transform):

    def __init__(self):
        super().__init__()
        self.allow_multi_consumers = True

    def pattern(self):
        return OpTypePattern('QuantizeLinear', [
            OpTypePattern('MatMul', [
                OpTypePattern('DequantizeLinear', [
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                ]),
                OpTypePattern('DequantizeLinear', [
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                ]),
            ]),
            OpTypePattern('.*'),
            OpTypePattern('.*'),
        ])

    def replacement(self, match_node):
        x_node = match_node.input_nodes[0].input_nodes[0].input_nodes[0]
        x_scale_node = match_node.input_nodes[0].input_nodes[0].input_nodes[1]
        x_zero_point_node = match_node.input_nodes[0].input_nodes[
            0].input_nodes[2]

        y_node = match_node.input_nodes[0].input_nodes[1].input_nodes[0]
        y_scale_node = match_node.input_nodes[0].input_nodes[1].input_nodes[1]
        y_zero_point_node = match_node.input_nodes[0].input_nodes[
            1].input_nodes[2]

        matmul_node = match_node.input_nodes[0]
        print('Convert MatMul: ', matmul_node.node.name)
        z_scale_node = match_node.input_nodes[1]
        z_zero_point_node = match_node.input_nodes[2]

        if hasattr(x_node.node, 'output'):
            x_name = x_node.node.output[0]
        else:
            x_name = x_node.node.name

        if hasattr(y_node.node, 'output'):
            y_name = y_node.node.output[0]
        else:
            y_name = y_node.node.name
        # Make the QLinearConv node
        qlinear_matmul = helper.make_node(
            'QLinearMatMul',
            inputs=[
                x_name,  # x
                x_scale_node.node.name,  # x_scale
                x_zero_point_node.node.name,  # x_zero_point
                y_name,  # y
                y_scale_node.node.name,  # y_scale
                y_zero_point_node.node.name,  # y_zero_point
                z_scale_node.node.name,  # y_scale
                z_zero_point_node.node.name,  # y_zero_point
            ],
            outputs=[match_node.node.output[0]],
            name=matmul_node.node.name)
        qlinear_matmul_node = NodeTree(qlinear_matmul,
                                       weights=None,
                                       input_nodes=[
                                           x_node,
                                           x_scale_node,
                                           x_zero_point_node,
                                           y_node,
                                           y_scale_node,
                                           y_zero_point_node,
                                           z_scale_node,
                                           z_zero_point_node,
                                       ],
                                       metadata=None)

        return qlinear_matmul_node


class AddQDQToQOPTransform(transforms.Transform):

    def __init__(self):
        super().__init__()
        self.allow_multi_consumers = True

    def pattern(self):
        return OpTypePattern('QuantizeLinear', [
            OpTypePattern('Add', [
                OpTypePattern('DequantizeLinear', [
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                ]),
                OpTypePattern('DequantizeLinear', [
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                ]),
            ]),
            OpTypePattern('.*'),
            OpTypePattern('.*'),
        ])

    def replacement(self, match_node):
        x_node = match_node.input_nodes[0].input_nodes[0].input_nodes[0]
        x_scale_node = match_node.input_nodes[0].input_nodes[0].input_nodes[1]
        x_zero_point_node = match_node.input_nodes[0].input_nodes[
            0].input_nodes[2]

        y_node = match_node.input_nodes[0].input_nodes[1].input_nodes[0]
        y_scale_node = match_node.input_nodes[0].input_nodes[1].input_nodes[1]
        y_zero_point_node = match_node.input_nodes[0].input_nodes[
            1].input_nodes[2]

        add_node = match_node.input_nodes[0]
        print('Convert Add: ', add_node.node.name)
        z_scale_node = match_node.input_nodes[1]
        z_zero_point_node = match_node.input_nodes[2]

        if hasattr(x_node.node, 'output'):
            x_name = x_node.node.output[0]
        else:
            x_name = x_node.node.name

        if hasattr(y_node.node, 'output'):
            y_name = y_node.node.output[0]
        else:
            y_name = y_node.node.name
        # Make the QLinearConv node
        qlinear_add = helper.make_node(
            'QLinearAdd',
            inputs=[
                x_name,  # x
                x_scale_node.node.name,  # x_scale
                x_zero_point_node.node.name,  # x_zero_point
                y_name,  # y
                y_scale_node.node.name,  # y_scale
                y_zero_point_node.node.name,  # y_zero_point
                z_scale_node.node.name,  # y_scale
                z_zero_point_node.node.name,  # y_zero_point
            ],
            outputs=[match_node.node.output[0]],
            domain=ms_domain,
            name=add_node.node.name)
        qlinear_add_node = NodeTree(qlinear_add,
                                    weights=None,
                                    input_nodes=[
                                        x_node,
                                        x_scale_node,
                                        x_zero_point_node,
                                        y_node,
                                        y_scale_node,
                                        y_zero_point_node,
                                        z_scale_node,
                                        z_zero_point_node,
                                    ],
                                    metadata=None)

        return qlinear_add_node


class MulQDQToQOPTransform(transforms.Transform):

    def __init__(self):
        super().__init__()
        self.allow_multi_consumers = True

    def pattern(self):
        return OpTypePattern('QuantizeLinear', [
            OpTypePattern('Mul', [
                OpTypePattern('DequantizeLinear', [
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                ]),
                OpTypePattern('DequantizeLinear', [
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                ]),
            ]),
            OpTypePattern('.*'),
            OpTypePattern('.*'),
        ])

    def replacement(self, match_node):
        x_node = match_node.input_nodes[0].input_nodes[0].input_nodes[0]
        x_scale_node = match_node.input_nodes[0].input_nodes[0].input_nodes[1]
        x_zero_point_node = match_node.input_nodes[0].input_nodes[
            0].input_nodes[2]

        y_node = match_node.input_nodes[0].input_nodes[1].input_nodes[0]
        y_scale_node = match_node.input_nodes[0].input_nodes[1].input_nodes[1]
        y_zero_point_node = match_node.input_nodes[0].input_nodes[
            1].input_nodes[2]

        mul_node = match_node.input_nodes[0]
        print('Convert Mul: ', mul_node.node.name)
        z_scale_node = match_node.input_nodes[1]
        z_zero_point_node = match_node.input_nodes[2]

        if hasattr(x_node.node, 'output'):
            x_name = x_node.node.output[0]
        else:
            x_name = x_node.node.name

        if hasattr(y_node.node, 'output'):
            y_name = y_node.node.output[0]
        else:
            y_name = y_node.node.name
        # Make the QLinearConv node
        qlinear_mul = helper.make_node(
            'QLinearMul',
            inputs=[
                x_name,  # x
                x_scale_node.node.name,  # x_scale
                x_zero_point_node.node.name,  # x_zero_point
                y_name,  # y
                y_scale_node.node.name,  # y_scale
                y_zero_point_node.node.name,  # y_zero_point
                z_scale_node.node.name,  # y_scale
                z_zero_point_node.node.name,  # y_zero_point
            ],
            outputs=[match_node.node.output[0]],
            domain=ms_domain,
            name=mul_node.node.name)
        qlinear_mul_node = NodeTree(qlinear_mul,
                                    weights=None,
                                    input_nodes=[
                                        x_node,
                                        x_scale_node,
                                        x_zero_point_node,
                                        y_node,
                                        y_scale_node,
                                        y_zero_point_node,
                                        z_scale_node,
                                        z_zero_point_node,
                                    ],
                                    metadata=None)

        return qlinear_mul_node


class SigmoidQDQToQOPTransform(transforms.Transform):

    def __init__(self):
        super().__init__()
        self.allow_multi_consumers = False

    def pattern(self):
        return OpTypePattern('QuantizeLinear', [
            OpTypePattern('Sigmoid', [
                OpTypePattern('DequantizeLinear', [
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                    OpTypePattern('.*'),
                ]),
            ]),
            OpTypePattern('.*'),
            OpTypePattern('.*'),
        ])

    def replacement(self, match_node):
        x_node = match_node.input_nodes[0].input_nodes[0].input_nodes[0]
        x_scale_node = match_node.input_nodes[0].input_nodes[0].input_nodes[1]
        x_zero_point_node = match_node.input_nodes[0].input_nodes[
            0].input_nodes[2]

        sigmoid_node = match_node.input_nodes[0]
        print('Convert Sigmoid: ', sigmoid_node.node.name)
        z_scale_node = match_node.input_nodes[1]
        z_zero_point_node = match_node.input_nodes[2]

        if hasattr(x_node.node, 'output'):
            x_name = x_node.node.output[0]
        else:
            x_name = x_node.node.name

        # Make the QLinearConv node
        qlinear_sigmoid = helper.make_node(
            'QLinearSigmoid',
            inputs=[
                x_name,  # x
                x_scale_node.node.name,  # x_scale
                x_zero_point_node.node.name,  # x_zero_point
                z_scale_node.node.name,  # y_scale
                z_zero_point_node.node.name,  # y_zero_point
            ],
            outputs=[match_node.node.output[0]],
            domain=ms_domain,
            name=sigmoid_node.node.name)
        qlinear_sigmoid_node = NodeTree(qlinear_sigmoid,
                                        weights=None,
                                        input_nodes=[
                                            x_node,
                                            x_scale_node,
                                            x_zero_point_node,
                                            z_scale_node,
                                            z_zero_point_node,
                                        ],
                                        metadata=None)

        return qlinear_sigmoid_node


class RemoveQDQTransform(transforms.Transform):

    def __init__(self):
        super().__init__()
        self.allow_multi_consumers = False

    def pattern(self):
        return OpTypePattern(
            'DequantizeLinear',
            [
                OpTypePattern(
                    'QuantizeLinear',
                    [
                        OpTypePattern('.*'),  # input
                        OpTypePattern('.*'),
                        OpTypePattern('.*')
                    ]),
                OpTypePattern('.*'),  # scale
                OpTypePattern('.*'),  # zero_point
            ])

    def replacement(self, match_node):
        print('Remove: ', match_node.node.name)
        return match_node.input_nodes[0].input_nodes[0]
