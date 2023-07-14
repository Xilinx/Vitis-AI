#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Apply graph transformations to a onnx model."""

import collections
import copy
import re

import onnx
import numpy as np
import enum

from vai_q_onnx.graph_transformations import transforms as transforms_mod

NodeTree = transforms_mod.NodeTree


class ModelTransformer(object):
    """Matches patterns to apply transforms in a tf.keras model graph."""

    def __init__(self,
                 model,
                 transforms,
                 candidate_nodes=None,
                 node_metadata=None):
        """Construct ModelTransformer.

    Args:
      model: Onnx model to be transformed.
      transforms: List of transforms to be applied to the model.
      candidate_nodes: Names of nodes which may be transformed. Only nodes
        whose names are in candidate_nodes are matched against patterns. The
        default is that all nodes may be transformed.
      node_metadata: Dictionary of metadata associated with each node in the
        model. The keys are node names.
    """
        if not isinstance(model, onnx.ModelProto):
            raise ValueError('Only onnx models can be transformed.')

        if node_metadata is None:
            node_metadata = {}

        self.model = model
        self.transformed_model = copy.deepcopy(self.model)
        self.transforms = transforms
        self.candidate_nodes = {}
        self.node_metadata = node_metadata

        self._update_status()

    class NodeType(enum.Enum):
        NODE = 1
        INITIALIZER = 2
        INPUT = 3

    def _update_status(self):
        """Update internal status."""

        self.name_to_node_map = self._map_name_to_node(self.transformed_model)
        self.name_to_init_map = self._map_name_to_init(self.transformed_model)
        self.name_to_input_map = self._map_name_to_input(self.transformed_model)
        self.tensor_to_producer_map = self._map_tensor_to_producer(
            self.transformed_model)
        self.tensor_to_consumer_map = self._map_tensor_to_consumer(
            self.transformed_model)

    @staticmethod
    def _name(obj):
        return obj.__class__.__name__

    @staticmethod
    def _map_name_to_node(model):
        """Returns a dict of name to node.

        Returns:
            {node.name: node}
        """
        name_to_node_map = {}
        for node in model.graph.node:
            name_to_node_map[ModelTransformer._get_node_name(node)] = node
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

    def _get_node_metadata(self, node_name):
        return self._node_metadata_map.get(node_name, {})

    def _get_consuming_nodes(self, check_node):
        """Returns all the nodes which are out nodes from the node.

        Returns: 
          {output_index: [consumer_node_name]}
          {0: [nodes]} for initializers
        """
        consuming_nodes = {}
        if self._node_type(check_node) == ModelTransformer.NodeType.NODE:
            for index, output_tensor in enumerate(check_node.output):
                if output_tensor in self.tensor_to_consumer_map:
                    consuming_nodes[index] = [
                        self._get_node_name(node)
                        for node in self.tensor_to_consumer_map[output_tensor]
                    ]
        elif self._node_type(
                check_node) == ModelTransformer.NodeType.INITIALIZER:
            consuming_nodes[0] = [
                self._get_node_name(node) for node in
                self.tensor_to_consumer_map[self._get_node_name(check_node)]
            ]
        elif self._node_type(check_node) == ModelTransformer.NodeType.INPUT:
            consuming_nodes[0] = [
                self._get_node_name(node) for node in
                self.tensor_to_consumer_map[self._get_node_name(check_node)]
            ]
        else:
            raise ValueError(
                'Invalid node type for node: {}'.format(check_node))

        return consuming_nodes

    def _get_output_consumers(self, check_node):
        """Returns if any tensors from the node are outputs of the model.

        Returns:
          {output_index: output_name} for nodes
        """
        output_consumers = {}
        if self._node_type(check_node) == ModelTransformer.NodeType.NODE:
            output_tensors = check_node.output
            for output in self.transformed_model.graph.output:
                for index, out in enumerate(output_tensors):
                    if output.name == out:
                        output_consumers[index] = output

        # Typcially, intializers and inputs are not outputs of the model

        return output_consumers

    @staticmethod
    def _get_node_name(node):
        return node.name

    @staticmethod
    def _node_type(node):
        """Returns whether the node is a node or initializer."""
        if isinstance(node, onnx.NodeProto):
            return ModelTransformer.NodeType.NODE
        elif isinstance(node, onnx.TensorProto):
            return ModelTransformer.NodeType.INITIALIZER
        elif isinstance(node, onnx.ValueInfoProto):
            return ModelTransformer.NodeType.INPUT
        else:
            raise ValueError('Unknown node type for node: {}'.format(node))

    def _get_matched_nodes(self, transform):
        return self._transform_matched_nodes_map.get(self._name(transform), [])

    def _match_pattern(self, target, pattern):
        for p in pattern.split('|'):
            if re.match('^' + p + '$', target) is not None:
                return True
        return False

    def _match_node(self, node, pattern):
        """Check if any specific node or initializer matches the pattern."""

        if self.candidate_nodes and self._get_node_name(
                node) not in self.candidate_nodes:
            return False

        node_type = self._node_type(node)

        node_matched = False
        if node_type == ModelTransformer.NodeType.NODE and self._match_pattern(
                node.op_type, pattern.op_type):
            node_matched = True

        init_matched = False
        match_init = pattern.op_type in ['initializer', '.*']
        if match_init and node_type == ModelTransformer.NodeType.INITIALIZER and self._get_node_name(
                node) in self.name_to_init_map:
            init_matched = True

        input_matched = False
        match_input = pattern.op_type in ['input', '.*']
        if match_input and node_type == ModelTransformer.NodeType.INPUT and self._get_node_name(
                node) in self.name_to_input_map:
            input_matched = True

        #TODO: match config

        return node_matched or init_matched or input_matched

    def _is_match_supported(self, node, is_head_node, allow_multi_consumers):
        """Check if ModelTransformer supports transformations given number of inputs and outputs at a node.

        Args:
          node: node for pattern matching.
          is_head_node: whether this is the head node (e.g. in A -> B , B is the
            head node).
          allow_multi_consumers: whether to allow matching for intermediate nodes
            with multiple consumers. Since replacing intermedieate nodes with multiple
            consumers may lead to dangling nodes, this is disabled by default. It is
            useful to match some complicated patterns.

        Returns:
          whether match is supported.
        """
        consuming_nodes = self._get_consuming_nodes(node)
        consuming_nodes_list = []
        for L in consuming_nodes.values():
            consuming_nodes_list.extend(L)
        if len(consuming_nodes_list) > 1:
            if not is_head_node and not allow_multi_consumers:
                return False

        output_nodes = self._get_output_consumers(node)
        if len(output_nodes) >= 1:
            if not is_head_node:
                return False

        return True

    def _get_input_node_init_names(self, node):
        """Get the names of a node's input nodes or initializers."""
        input_node_init_names = []
        # Keep the order during matching
        if self._node_type(node) == ModelTransformer.NodeType.NODE:
            for input_tensor in node.input:
                if input_tensor in self.tensor_to_producer_map:
                    input_node_init_names.append(
                        self.tensor_to_producer_map[input_tensor].name)
                elif input_tensor in self.name_to_init_map:
                    input_node_init_names.append(
                        self.name_to_init_map[input_tensor].name)
                elif input_tensor in self.name_to_input_map:
                    input_node_init_names.append(
                        self.name_to_input_map[input_tensor].name)
                else:
                    raise ValueError(
                        'Cannot find producer of tensor: {}'.format(
                            input_tensor))
            return input_node_init_names

        # Initializers and inputs have no inputs
        return None

    def _get_nodes_inits(self, node_names):
        """Returns nodes or initializers with given names, keep the order."""
        nodes_inits = []
        for name in node_names:
            if name in self.name_to_node_map:
                nodes_inits.append(self.name_to_node_map[name])
            elif name in self.name_to_init_map:
                nodes_inits.append(self.name_to_init_map[name])
            elif name in self.name_to_input_map:
                nodes_inits.append(self.name_to_input_map[name])
            else:
                raise ValueError(
                    'Cannot find node or initializer `{}` in the model.'.format(
                        name))
        return nodes_inits

    def _match_node_with_inputs(self, node, pattern, is_head_node,
                                allow_multi_consumers):
        """Match pattern at this node, and continue to match at its inputs."""

        if not self._match_node(node, pattern):
            return None

        if not self._is_match_supported(node, is_head_node,
                                        allow_multi_consumers):
            return None

        if len(pattern.inputs) == 0:
            # Leaf node in pattern.
            return NodeTree(node, [], [],
                            self._get_node_metadata(self._get_node_name(node)))

        input_node_init_names = self._get_input_node_init_names(node)
        input_nodes_inits = self._get_nodes_inits(input_node_init_names)

        if len(input_nodes_inits) != len(pattern.inputs):
            return None

        input_match_node_matches = []
        for input_node_init, pattern_ in zip(input_nodes_inits, pattern.inputs):
            match_node = self._match_node_with_inputs(
                input_node_init,
                pattern_,
                is_head_node=False,
                allow_multi_consumers=allow_multi_consumers)
            if not match_node:
                return None
            input_match_node_matches.append(match_node)

        return NodeTree(node, [], input_match_node_matches,
                        self._get_node_metadata(self._get_node_name(node)))

    def _find_pattern(self,
                      pattern,
                      matched_nodes=None,
                      allow_multi_consumers=False):
        for node in self.transformed_model.graph.node:
            if matched_nodes and node.name in matched_nodes:
                continue

            match_node = self._match_node_with_inputs(
                node,
                pattern,
                is_head_node=True,
                allow_multi_consumers=allow_multi_consumers)

            if match_node:
                return match_node

        return None

    def _store_successful_match(self, transform, node_tree):
        """Store matched results to avoid duplicated matching."""
        if self._name(transform) not in self._transform_matched_nodes_map:
            self._transform_matched_nodes_map[self._name(transform)] = []

        self._transform_matched_nodes_map[self._name(transform)].append(
            self._get_node_name(node_tree.node))

    @staticmethod
    def _get_node_names(node_tree):
        """Returns the list of node names in the node tree."""
        result = [ModelTransformer._get_node_name(node_tree.node)]
        for input_node in node_tree.input_nodes:
            result.extend(ModelTransformer._get_node_names(input_node))
        return result

    @staticmethod
    def _remove_nodes_inits(model, node_names):
        """Remove the nodes and initializers/inputs from model."""
        left = set(node_names)

        nodes_to_remove = []
        for node in model.graph.node:
            if node.name in left:
                nodes_to_remove.append(node)
                left.remove(node.name)
        for node in nodes_to_remove:
            model.graph.node.remove(node)

        inits_to_remove = []
        for init in model.graph.initializer:
            if init.name in left:
                inits_to_remove.append(init)
                left.remove(init.name)
        for init in inits_to_remove:
            model.graph.initializer.remove(init)

        inputs_to_remove = []
        for inp in model.graph.input:
            if inp.name in left:
                inputs_to_remove.append(inp)
                left.remove(inp.name)
        for inp in inputs_to_remove:
            model.graph.input.remove(inp)

        if left:
            print(
                'Warning: Cannot find nodes, initializers or inputs in model: {}'
                .format(left))

    @staticmethod
    def _add_node_init(model, node_init_to_add):
        """Add the node or initializer/input to the model."""
        if ModelTransformer._node_type(
                node_init_to_add) == ModelTransformer.NodeType.NODE:
            for node in model.graph.node:
                if node.name == node_init_to_add.name:
                    print(
                        'INFO: Node `{}` is already in model, skip adding it.'.
                        format(node.name))
                    return
            new_node = model.graph.node.add()
            new_node.CopyFrom(node_init_to_add)
        elif ModelTransformer._node_type(
                node_init_to_add) == ModelTransformer.NodeType.INITIALIZER:
            for init in model.graph.initializer:
                if init.name == node_init_to_add.name:
                    print(
                        'INFO: Initializer `{}` is already in model, skip adding it.'
                        .format(init.name))
                    return
            new_init = model.graph.initializer.add()
            new_init.CopyFrom(node_init_to_add)
        elif ModelTransformer._node_type(
                node_init_to_add) == ModelTransformer.NodeType.INPUT:
            for inp in model.graph.input:
                if inp.name == node_init_to_add.name:
                    print(
                        'INFO: Input `{}` is already in model, skip adding it.'.
                        format(inp.name))
                    return
            new_input = model.graph.input.add()
            new_input.CopyFrom(node_init_to_add)

    def _get_leaf_nodes(self, node_tree):
        """Return leaf nodes from the node tree."""
        # Initializers will not be treated as leaf nodes.
        if not node_tree.input_nodes and self._node_type(
                node_tree.node) == ModelTransformer.NodeType.NODE:
            return [node_tree.node]

        leaf_nodes = []
        for inp in node_tree.input_nodes:
            leaf_nodes.extend(self._get_leaf_nodes(inp))

        # Remove duplicate leaf nodes in case of:
        # 1) Two different nodes point to the same leaf node
        # 2) One node uses the same leaf node multiple times

        uniq_leaf_nodes = []
        for node in leaf_nodes:
            if node not in uniq_leaf_nodes:
                uniq_leaf_nodes.append(node)

        return uniq_leaf_nodes

    @staticmethod
    def _get_input_index(node, input_tensor):
        """Get the index of input tensor."""
        return list(node.input).index(input_tensor)

    def _replace(self, matched_node_tree, replacement_node_tree):
        """Replace the matched node tree with replacement node tree."""

        # 1. Point all consumers of the head of the matching sub-tree to the head
        # replacement node.

        matched_head_node = matched_node_tree.node
        replacement_head_node = replacement_node_tree.node

        consuming_nodes = self._get_consuming_nodes(matched_node_tree.node)
        for output_index, consumers in consuming_nodes.items():
            for consumer in consumers:
                consumer_node = self.name_to_node_map[consumer]
                input_index = self._get_input_index(consumer_node,
                                                    matched_head_node.output[0])
                consumer_node.input[input_index] = replacement_head_node.output[
                    0]

        # 2. Update the graph outputs

        output_consumers = self._get_output_consumers(matched_node_tree.node)
        for index, consumer in output_consumers.items():
            consumer.name = replacement_node_tree.node.output[index]

        # 3. Create input tensors for the replacement leaf nodes, connect the nodes
        # to the original graph.

        original_leaf_nodes = self._get_leaf_nodes(matched_node_tree)
        replacement_leaf_nodes = self._get_leaf_nodes(replacement_node_tree)

        if len(original_leaf_nodes) != len(replacement_leaf_nodes):
            raise RuntimeError(
                'Difference size of leaf layers not supported yet({} vs {})'.
                format(len(original_leaf_node), len(replacement_leaf_node)))

        for original_leaf_node, replacement_leaf_node in zip(
                original_leaf_nodes, replacement_leaf_nodes):
            replacement_leaf_node.ClearField('input')
            for input_tensor in original_leaf_node.input:
                replacement_leaf_node.input.append(input_tensor)

        # 4. Remove the original matched nodes

        nodes_inits_to_remove = self._get_node_names(matched_node_tree)
        self._remove_nodes_inits(self.transformed_model, nodes_inits_to_remove)

        # 5. Add the new nodes to the original graph

        def _add_replacement_node(node_tree):
            """Recursively add new nodes"""
            for input_node in node_tree.input_nodes:
                _add_replacement_node(input_node)

            self._add_node_init(self.transformed_model, node_tree.node)

        _add_replacement_node(replacement_node_tree)

        # 6. Update status
        self._update_status()

        # TODO
        # remove unused nodes and initializers
        # remove duplicated nodes and intializers
        # topological sort
        # validate the transformed model
        return

    def transform(self):
        """Transforms the Onnx model by applying all the specified transforms.

        This is the main entry point function used to apply the transformations to
        the Onnx model.

        Not suitable for multi-threaded use. Creates and manipulates internal state.

        Returns:
          (Onnx model after transformation, Updated node metadata map)
        """

        # Stores map of Transform -> List of nodes names matched by transform.
        # Same transform should not match+replace the same node more than once
        # to prevent infinite loops.
        self._transform_matched_nodes_map = {}

        # Maintains a current mutable copy of the metadata through transformation.
        self._node_metadata_map = copy.deepcopy(self.node_metadata)

        # We run an infinite loop and keep applying transformations as long as
        # patterns are found. This allows recursive pattern matching where a
        # modification by one transform may lead to another match.
        while True:
            match_found = False
            for transform in self.transforms:
                # A transform may find multiple instances of a pattern in the model.
                # Keep finding and replacing till done.
                while True:
                    matched_node_tree = self._find_pattern(
                        transform.pattern(), self._get_matched_nodes(transform),
                        transform.allow_multi_consumers)

                    if not matched_node_tree:
                        break

                    self._store_successful_match(transform, matched_node_tree)

                    # Copying the match_node ensures the replacement code can
                    # freely modify the match.
                    replacement_node_tree = transform.replacement(
                        copy.deepcopy(matched_node_tree))

                    match_found = True
                    self._replace(matched_node_tree, replacement_node_tree)

            if not match_found:
                break

        return self.transformed_model, self._node_metadata_map
