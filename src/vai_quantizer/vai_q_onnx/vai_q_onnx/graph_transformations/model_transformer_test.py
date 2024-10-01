#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for Model Transformation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

import onnx
from onnx import helper, TensorProto, numpy_helper

from vai_q_onnx.graph_transformations import model_transformer
from vai_q_onnx.graph_transformations import transforms

ModelTransformer = model_transformer.ModelTransformer
Transform = transforms.Transform
OpTypePattern = transforms.OpTypePattern
NodeTree = transforms.NodeTree


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


class ModelTransformerTest(unittest.TestCase):

    def _build_model(self):
        #    (input)
        #       |
        #      GRU
        #      /  \
        #  Conv(1) \
        #     |     \
        #    Relu  Conv(2)
        #     |     |
        #     \     /
        #       Add
        #        |
        #       (output)
        initializers = []
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                              [4, 8, 12])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                               [4, 2, 8, 8])

        # make GRU
        initializers.append(
            generate_input_initializer([2, 24, 12], np.float32, "W_GRU"))
        initializers.append(
            generate_input_initializer([2, 24, 8], np.float32, "R_GRU"))
        initializers.append(
            generate_input_initializer([2, 8, 8], np.float32, "H_GRU"))
        initializers.append(generate_input_initializer([8], np.float32,
                                                       "B_GRU"))
        initializers.append(generate_input_initializer([1], np.float32,
                                                       "S_LEN"))
        gru_node = helper.make_node(
            "GRU", ["input", "W_GRU", "R_GRU", "B_GRU", "S_LEN", "H_GRU"],
            ["GRU_O"],
            hidden_size=8,
            direction="bidirectional",
            name='GRU1')

        initializers.append(
            generate_input_initializer([2, 2, 1, 1], np.float32, "W1"))
        initializers.append(
            generate_input_initializer([2, 2, 1, 1], np.float32, "W2"))
        initializers.append(generate_input_initializer([2], np.float32, "B1"))
        initializers.append(generate_input_initializer([2], np.float32, "B2"))
        conv_node_1 = helper.make_node("Conv", ["GRU_O", "W1", "B1"],
                                       ["Conv1_O"],
                                       name="Conv1")
        conv_node_2 = helper.make_node("Conv", ["GRU_O", "W2", "B2"],
                                       ["Conv2_O"],
                                       name="Conv2")
        relu_node = helper.make_node("Relu", ["Conv1_O"], ["Relu_O"],
                                     name="Relu1")
        add_node = helper.make_node("Add", ["Relu_O", "Conv2_O"], ["output"],
                                    name="Add1")
        graph = helper.make_graph(
            [conv_node_1, relu_node, conv_node_2, gru_node, add_node],
            "onnx_model_test",
            [input],
            [output],
            initializer=initializers,
        )
        model = helper.make_model(graph,
                                  opset_imports=[helper.make_opsetid("", 13)])
        return model

    class ReplaceWholeModel(transforms.Transform):

        def __init__(self):
            super().__init__()
            self.allow_multi_consumers = True

        def pattern(self):
            return OpTypePattern(
                'Add',
                [
                    OpTypePattern(
                        'Relu',
                        [
                            OpTypePattern(
                                'Conv',
                                [
                                    OpTypePattern(
                                        'GRU',
                                        [
                                            OpTypePattern('.*'),  # input
                                            OpTypePattern('.*'),  # initializer
                                            OpTypePattern('.*'),  # initializer
                                            OpTypePattern('.*'),  # initializer
                                            OpTypePattern('.*'),  # initializer
                                            OpTypePattern('.*'),  # initializer
                                        ]),
                                    OpTypePattern('.*'),  # W initializer
                                    OpTypePattern('.*'),  # B initializer
                                ])
                        ]),
                    OpTypePattern(
                        'Conv',
                        [
                            OpTypePattern('GRU'),
                            OpTypePattern('.*'),  # W initializer
                            OpTypePattern('.*'),  # B initializer
                        ])
                ])

        def replacement(self, match_node):
            return match_node

    def testReplaceWholeModel(self):
        model = self._build_model()
        onnx.save(model, 'tmp.onnx')

        transformed_model, _ = ModelTransformer(
            model, [self.ReplaceWholeModel()]).transform()

        onnx.save(transformed_model, 'tmp_transformed.onnx')

    class RemoveRelu(transforms.Transform):

        def __init__(self):
            super().__init__()

        def pattern(self):
            return OpTypePattern('Relu', [
                OpTypePattern('Conv'),
            ])

        def replacement(self, match_node):
            return match_node.input_nodes[0]

    def testRemoveRelu(self):
        model = self._build_model()

        transformed_model, _ = ModelTransformer(
            model, [self.RemoveRelu()]).transform()

        onnx.save(transformed_model, 'tmp_transformed_1.onnx')


if __name__ == '__main__':
    unittest.main()
