#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# pylint: disable=g-explicit-length-test
"""Utility functions."""

import onnx
import enum
import copy

from google.protobuf import text_format


def get_tensor_value(initializer):
    """Convert TensorProto to numpy array."""
    return onnx.numpy_helper.to_array(initializer)


def generate_initializer(tensor_array, dtype, name):
    """Generate initializers from numpy array."""
    tensor = tensor_array.astype(dtype)
    init = onnx.numpy_helper.from_array(tensor, name)
    return init


def save_model(model, path, as_text=False):
    """Save onnx model to disk."""
    if as_text:
        with open(path, 'w') as f:
            f.write(text_format.MessageToString(model))
    else:
        onnx.save(model, path)


class SharedNodesHelper(object):

    class NodeType(enum.Enum):
        NODE = 1
        INITIALIZER = 2
        INPUT = 3

    @staticmethod
    def _node_type(node):
        """Returns whether the node is a node or initializer."""
        if isinstance(node, onnx.NodeProto):
            return SharedNodesHelper.NodeType.NODE
        elif isinstance(node, onnx.TensorProto):
            return SharedNodesHelper.NodeType.INITIALIZER
        elif isinstance(node, onnx.ValueInfoProto):
            return SharedNodesHelper.NodeType.INPUT
        else:
            raise ValueError('Unknown node type for node: {}'.format(node))

    @staticmethod
    def _add_node_init(model, node_init_to_add):
        """Add the node or initializer/input to the model."""
        if SharedNodesHelper._node_type(
                node_init_to_add) == SharedNodesHelper.NodeType.NODE:
            for node in model.graph.node:
                if node.name == node_init_to_add.name:
                    print(
                        'INFO: Node `{}` is already in model, skip adding it.'.
                        format(node.name))
                    return
            new_node = model.graph.node.add()
            new_node.CopyFrom(node_init_to_add)
        elif SharedNodesHelper._node_type(
                node_init_to_add) == SharedNodesHelper.NodeType.INITIALIZER:
            for init in model.graph.initializer:
                if init.name == node_init_to_add.name:
                    print(
                        'INFO: Initializer `{}` is already in model, skip adding it.'
                        .format(init.name))
                    return
            new_init = model.graph.initializer.add()
            new_init.CopyFrom(node_init_to_add)
        elif SharedNodesHelper._node_type(
                node_init_to_add) == SharedNodesHelper.NodeType.INPUT:
            for inp in model.graph.input:
                if inp.name == node_init_to_add.name:
                    print(
                        'INFO: Input `{}` is already in model, skip adding it.'.
                        format(inp.name))
                    return
            new_input = model.graph.input.add()
            new_input.CopyFrom(node_init_to_add)

    @staticmethod
    def _map_name_to_node(model):
        """Returns a dict of name to node.

        Returns:
            {node.name: node}
        """
        name_to_node_map = {}
        for node in model.graph.node:
            name_to_node_map[SharedNodesHelper._get_node_name(node)] = node
        return name_to_node_map

    @staticmethod
    def _map_name_to_init(model):
        """Returns a dict of name to initializer.

        Returns:
            {initializer.name: initializer}
        """
        name_to_init_map = {}
        for init in model.graph.initializer:
            name_to_init_map[init.name] = init
        return name_to_init_map

    @staticmethod
    def _map_name_to_input(model):
        """Returns a dict of name to input.

        Returns:
            {initializer.name: initializer}
        """
        name_to_input_map = {}
        for inp in model.graph.input:
            name_to_input_map[inp.name] = inp
        return name_to_input_map

    @staticmethod
    def _map_tensor_to_producer(model):
        """Returns a dict of tensor to its producer node.

        Returns:
            {tensor.name: producer_node}
        """
        tensor_to_producer_map = {}
        for node in model.graph.node:
            for output_tensor in node.output:
                tensor_to_producer_map[output_tensor] = node

        for init in model.graph.initializer:
            tensor_to_producer_map[init.name] = init

        for inp in model.graph.input:
            tensor_to_producer_map[inp.name] = inp
        return tensor_to_producer_map

    @staticmethod
    def _map_tensor_to_consumer(model):
        """Returns a dict of tensor to its consumer nodes.

        Returns:
            {tensor.name: [consumer_nodes]}
        """
        tensor_to_consumer_map = {}
        for node in model.graph.node:
            for input_tensor in node.input:
                if input_tensor not in tensor_to_consumer_map:
                    tensor_to_consumer_map[input_tensor] = [node]
                else:
                    tensor_to_consumer_map[input_tensor].append(node)
        return tensor_to_consumer_map

    @staticmethod
    def _get_node_name(node):
        return node.name


def copy_shared_nodes(model):
    helper = SharedNodesHelper()

    # Rename all nodes
    type_idx = {}
    for node in model.graph.node:
        if node.op_type not in type_idx:
            type_idx[node.op_type] = 1
        node.name = node.op_type + '_' + str(type_idx[node.op_type])
        print('Add node name: ', node.name)
        type_idx[node.op_type] += 1

    onnx.save(model, 'node_name_added.onnx')

    modified_flag = True
    while modified_flag:
        modified_flag = False
        name_to_node_map = helper._map_name_to_node(model)
        name_to_init_map = helper._map_name_to_init(model)
        name_to_input_map = helper._map_name_to_input(model)
        tensor_to_producer_map = helper._map_tensor_to_producer(model)

        tensor_to_consumer_map = {}
        new_nodes = {}
        node_inputs_to_rename = []
        for node in model.graph.node:
            for inp_id, input_tensor in enumerate(node.input):
                if input_tensor not in tensor_to_consumer_map:
                    tensor_to_consumer_map[input_tensor] = [node]
                else:
                    producer = tensor_to_producer_map[input_tensor]
                    new_node = copy.deepcopy(producer)
                    idx = len(tensor_to_consumer_map[input_tensor])
                    new_node.name = new_node.name + '_' + str(idx)

                    if isinstance(producer, onnx.TensorProto):
                        new_nodes[new_node.name] = new_node
                        print('Need to update init: ', node.name, inp_id,
                              new_node.name)
                        node_inputs_to_rename.append(
                            (node.name, inp_id, new_node.name))
                        tensor_to_consumer_map[input_tensor].append(node)
                    elif isinstance(producer, onnx.NodeProto):
                        if producer.op_type in ["DequantizeLinear"]:
                            new_node.output[0] = new_node.name + '_out'
                            new_nodes[new_node.name] = new_node
                            print('Need to update: ', node.name, inp_id,
                                  new_node.output[0])
                            node_inputs_to_rename.append(
                                (node.name, inp_id, new_node.output[0]))
                            tensor_to_consumer_map[input_tensor].append(node)
                    else:
                        pass

        for name, new_node in new_nodes.items():
            helper._add_node_init(model, new_node)

        name_to_node_map = helper._map_name_to_node(model)

        for (node_name, inp_id, new_node_name) in node_inputs_to_rename:
            modified_flag = True
            node = name_to_node_map[node_name]
            print('Update node input', node_name, inp_id, new_node_name)
            node.input[inp_id] = new_node_name

    return model
