#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Defines core classes for expressing onnx model transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import six


class OpTypePattern(object):
    """Defines a tree sub-graph pattern of onnx nodes to match in a model.

    `OpTypePattern` can be used to describe various common patterns in model
    graphs that we need to find.

    Examples:
        Matches a Conv+BN+ReLU6 and DepthwiseConv+BN+ReLU6 pattern.
        pattern = OpTypePattern('ReLU', {'max_value': 6.0}, [
            OpTypePattern('BatchNormalization', {}, [
                OpTypePattern('Conv2D|DepthwiseConv2D', {} [])
            ])
        ])

        Matches multiple Conv2Ds feeding into a Concat.
        pattern = OpTypePattern('Concat', {}, [
            OpTypePattern('Conv2D', {}, []),
            OpTypePattern('Conv2D', {}, [])
        ])
    """

    def __init__(self, op_type="", inputs=None, config=None):
        """Construct pattern to match.

        Args:
            op_type: Type of onnx node.
            inputs: input nodes to the node.
            config: Map of arguments of the node to match. For eg., for ReLU(6.0)
                it would be {'max_value': 6.0}.
        """
        if inputs is None:
            inputs = []

        if not isinstance(op_type, str):
            raise ValueError('Invalid op_type: {}'.format(op_type))
        self.op_type = op_type

        self.inputs = inputs
        self.config = config

    def __str__(self):
        return '{} <- [{}]'.format(self.op_type,
                                   ', '.join([str(inp) for inp in self.inputs]))


class NodeTree(object):
    """Represents a pattern matching results in a node containing a tree.

    `NodeTree` is used to represent a tree of nodes in a model. It contains
    the NodeDef which describes the node, and other input nodes feeding into it.

    It is used as a generic class to represent both sets of nodes which have
    been found in a model, and nodes which should be replaced inside the model.
    """

    def __init__(self,
                 node=None,
                 weights=None,
                 input_nodes=None,
                 metadata=None):
        """Construct a NodeTree representing a tree of nodes.

        Args:
          node: NodeDef of this node.
          weights: An OrderedDict of weight name => value for the node.
          input_nodes: List of `NodeTree`s that feed into this node.
          metadata: Dictionary of metadata for a given node.
        """
        if input_nodes is None:
            input_nodes = []

        self.node = node
        self.weights = weights
        self.input_nodes = input_nodes

    def __str__(self):
        return '{} <- [{}]'.format(
            self.node.name,
            ', '.join([str(input_node) for input_node in self.input_nodes]))


@six.add_metaclass(abc.ABCMeta)
class Transform(object):
    """Defines a transform to be applied to a onnx model graph.

    A transform is a combination of 'Find + Replace' which describes how to find
    a pattern of nodes in a model, and what to replace those nodes with.

    A pattern is described using `OpTypePattern`. The replacement function receives
    a `NodeTree` which contains the matched nodes and should return a
    `NodeTree` which contains the set of nodes which replaced the matched
    nodes.
    """

    def __init__(self):
        # Disallow multi_consumers by default.
        self._allow_multi_consumers = False

    @abc.abstractmethod
    def pattern(self):
        """Return the `OpTypePattern` to find in the model graph."""
        raise NotImplementedError()

    @abc.abstractmethod
    def replacement(self, match_node):
        """Generate a replacement sub-graph for the matched sub-graph.

        The fundamental constraint of the replacement is that the replacement
        sub-graph should consume the same input tensors as the original sub-graph
        and also produce a final list of tensors which are same in number and shape
        as the original sub-graph. Not following this could crash model creation,
        or introduce bugs in the new model graph.

        Args:
          match_nodes: Matched NodeTree based on `self.pattern()`.
        """
        raise NotImplementedError()

    @property
    def allow_multi_consumers(self):
        """
        Whether to allow the internal node have multiple consuming nodes.
        
        E.g. 
             B                B
            /                /
        A --        to   E --
            \                \
             C --> D          F

        Should set allow_mulit_consumers if you want to match pattern "A --> C --> D".
        Please be careful to handle the transformation to not break the input connection
        of consumers outside the pattern, otherwise will lead to unknown input tensors.
        """
        return self._allow_multi_consumers

    @allow_multi_consumers.setter
    def allow_multi_consumers(self, value):
        self._allow_multi_consumers = bool(value)
