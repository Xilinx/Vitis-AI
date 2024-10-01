"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import logging
from typing import Dict, List, NoReturn

import numpy as np
from tqdm import tqdm
from xnnc.ir.enums import Layout
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *
from xnnc.optimizer.algorithm.subgraph_isomorph_vf2 import DiGraphMatcher
from xnnc.tensor.xtensor import XTensor

import sys
sys.setrecursionlimit(3000)

# create logger
logger = logging.getLogger(__name__)


class TFOptimizer(object):
    @classmethod
    def run(cls, xmodel: XModel, pattern_dict: Dict[str, Any]) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."
        assert pattern_dict is not None, "'pattern_dict' should not be None."

        level = pattern_dict.get("level")
        patterns = pattern_dict.get("patterns")

        # * remove all subgraphs for shape compute
        if level == "xnnc":
            TFOptimizer.remove_subgraphs_shape_compute(xmodel)

        # * run optimizations
        if patterns is not None:
            need_infer_shape = False
            opt_level = "level-0" if level == "xnnc" else "level-1"
            pbar = tqdm(
                patterns,
                desc=f"[INFO] perform {opt_level} opt",
                bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
            )
            for subgraph in pbar:
                pname = subgraph.name
                if hasattr(TFOptimizer, pname):
                    # find subgraph matching
                    matcher = DiGraphMatcher(xmodel, subgraph)
                    matches = matcher.search_matching()
                    if matches is not None:
                        # call optimization method
                        func = getattr(TFOptimizer, pname)
                        func(xmodel, matches)
                        if not need_infer_shape:
                            need_infer_shape = True
                else:
                    print(f"[WARNING] Not found optimization method: {pname}")

            if need_infer_shape:
                # perform topsort
                xmodel.topsort()

                # infer shape
                xmodel.infer_shape(layout=Layout[xmodel.layout], disable_pbar=True)

    @classmethod
    def remove_subgraphs_shape_compute(cls, xmodel: XModel) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."

        # remove the subgraphs starting with a Shape op
        inputs_xnode = []
        root_removed = []
        for xnode in xmodel.xnodes:
            if xnode.op_type == "shape":
                # update parent nodes
                if xnode.bottom is not None and len(xnode.bottom) > 0:
                    for pname in xnode.bottom:
                        pnode = xnode.host.get_xnode_by_name(pname)
                        assert (
                            pnode is not None
                        ), f"[ERROR] not found parent node: {pname}"
                        idx = pnode.top.index(xnode.op_name)
                        del pnode.top[idx]
                    xnode.bottom = []
                    root_removed.append(xnode)
            elif xnode.op_type in [
                "reshape",
                "resize",
            ]:
                if xnode.op_type == "reshape" and len(xnode.bottom) == 1:
                    # create an XModelNodeConst object to save shape info
                    const_xnode = XModelNodeConst(xnode.op_name + "_shape")
                    const_xnode.tensor = XTensor(np.array(xnode.shape))
                    const_xnode.top = [xnode.op_name]
                    xnode.bottom.append(const_xnode.op_name)
                    xmodel.add_xnode(const_xnode)
                    xnode.shape = []
                else:
                    # update parent nodes
                    assert len(xnode.bottom) == 2
                    pname = xnode.bottom[1]
                    pnode = xnode.host.get_xnode_by_name(pname)
                    assert pnode is not None, f"[ERROR] not found parent node: {pname}"
                    if not isinstance(pnode, XModelNodeConst):
                        pnode.top.remove(xnode.op_name)

                        # create an XModelNodeConst object to save shape info
                        const_xnode = XModelNodeConst(pname + "_shape")
                        const_xnode.tensor = pnode.outputs_tensor[0]
                        const_xnode.top = [xnode.op_name]
                        idx = xnode.bottom.index(pname)
                        assert idx is not None
                        xnode.bottom[idx] = const_xnode.op_name
                        xmodel.add_xnode(const_xnode)

            elif xnode.op_type == "input":
                inputs_xnode.append(xnode)
            elif xnode.op_type == "conv2d_transpose":
                # update parent nodes
                assert 1 <= len(xnode.bottom) <= 2 and xnode.output_shape is not None

                if len(xnode.bottom) == 2:
                    pname = xnode.bottom[1]
                    pnode = xnode.host.get_xnode_by_name(pname)
                    assert pnode is not None, f"[ERROR] not found parent node: {pname}"
                    # disconnet pnode
                    pnode.top = []
                    xnode.bottom.pop()

                # create an XModelNodeConst object to save shape info
                const_xnode = XModelNodeConst(xnode.op_name + "/output_shape/const")
                const_xnode.tensor = XTensor(
                    np.array(xnode.output_shape, dtype=np.int32)
                )
                const_xnode.top = [xnode.op_name]
                xnode.bottom.append(const_xnode.op_name)
                xmodel.add_xnode(const_xnode)

        def dfs(xnode, visited):
            if xnode.op_name not in visited:
                visited[xnode.op_name] = xnode
                if xnode.top is not None and len(xnode.top) > 0:
                    for cname in xnode.top:
                        cnode = xnode.host.get_xnode_by_name(cname)
                        assert (
                            cnode is not None
                        ), f"[ERROR] not found child node: {cname}"
                        dfs(cnode, visited)
                if xnode.bottom is not None and len(xnode.bottom) > 0:
                    for pname in xnode.bottom:
                        pnode = xnode.host.get_xnode_by_name(pname)
                        assert (
                            pnode is not None
                        ), f"[ERROR] not found parent node: {pname}"
                        dfs(pnode, visited)

        # * remove the disconnected subgraphs
        visited = {}
        for xnode in inputs_xnode:
            dfs(xnode, visited)

        xmodel.clear()
        xmodel.add_xnodes(visited.values())
        xmodel.topsort()

    @classmethod
    def merge_const_fix_elemadd_in_conv2d(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        1. replace ElemAdd with BiasAdd;
        2. merge Const-Fix-BiasAdd into Conv2d (or Depthwise_Conv2d, conv2d_transpose)

        # ******************************************************************************************************************
        # const(1)            A                                                                  A
        #     \              /                                                                   |
        #     fix(2)     conv2d, depthwise_conv2d, conv2d_transpose(3)   =>   conv2d, depthwise_conv2d, conv2d_transpose(3)
        #          \     /                                                                       |
        #         elemadd(4)                                                                     |
        #             |                                                                          |
        #             B                                                                          B
        # ******************************************************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_const = nodes_dict["node1"]
            assert node1_const.op_type == "const"
            node2_fix = nodes_dict["node2"]
            assert node2_fix.op_type == "fixneuron"
            node3_conv2d = nodes_dict["node3"]
            assert node3_conv2d.op_type in ["conv2d", "depthwise_conv2d", "conv2d_transpose"]
            node4_elemadd = nodes_dict["node4"]
            assert node4_elemadd.op_type == "elemadd"

            # perform optimization
            node3_conv2d.bias = node1_const.tensor
            for key in node3_conv2d.quant_bias.keys():
                node3_conv2d.quant_bias[key] = node2_fix.quant_in[key]
            # todo: tmp solution
            node3_conv2d.quant_bias[
                "round_mode"
            ] = 0  # "STD_ROUND" for a parent of Const type
            node3_conv2d.bias_term = True

            # remove node1 and node2
            for n in [node1_const, node2_fix]:
                n.bottom = []
                n.top = []
                xmodel.remove_xnode(n)

            # remove node4
            node4_elemadd.bottom.remove(node2_fix.op_name)
            xmodel.remove_xnode(node4_elemadd)

    @classmethod
    def replace_conv2dbackpropinput_with_conv2dtranspose_pattern1(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ****************************************************************************************************************************************************
        #                                                   A
        #                                                   |
        #                                                   |
        #                             __________________ shape(1) _________________
        #                            /                      |                      \
        #                           /                      / \                      \
        #                          /                      /   \                      \
        #                         /                      /     \                      \
        #     const(3)    stridedslice(2)               /       \                stridedslice(8)    const(9)                    A
        #            \      |                          /         \                           |         /                        |
        #             \     |                         |           |                          |        /                         |
        #            elemmul(4)    const(5)           |           |           const(11)     elemmul(10)             =>          |
        #                   \       |                 |           |                |        /                                   |
        #                    \      |                 |           |                |       /                                    |
        #                    elemadd(6)        stridedslice(7)   const(13)       elemadd(12)                             conv2d_transpose(15)
        #                        |                  |                   |             |
        #                        |                  |                   |             |
        #                        |                  |                   |             |
        #                        |__________________|_____ stack(14) ___|_____________|
        #                                                      |
        #                                                      |
        #                                                      |
        #                                               conv2d_transpose(15)
        #
        # ****************************************************************************************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_shape = nodes_dict["node1"]
            assert node1_shape.op_type == "shape"
            node2_stridedslice = nodes_dict["node2"]
            assert node2_stridedslice.op_type == "stridedslice"
            node3_const = nodes_dict["node3"]
            assert node3_const.op_type == "const"
            node4_elemmul = nodes_dict["node4"]
            assert node4_elemmul.op_type == "elemmul"
            node5_const = nodes_dict["node5"]
            assert node5_const.op_type == "const"
            node6_elemadd = nodes_dict["node6"]
            assert node6_elemadd.op_type == "elemadd"
            node7_stridedslice = nodes_dict["node7"]
            assert node7_stridedslice.op_type == "stridedslice"
            node8_stridedslice = nodes_dict["node8"]
            assert node8_stridedslice.op_type == "stridedslice"
            node9_const = nodes_dict["node9"]
            assert node9_const.op_type == "const"
            node10_elemmul = nodes_dict["node10"]
            assert node10_elemmul.op_type == "elemmul"
            node11_const = nodes_dict["node11"]
            assert node11_const.op_type == "const"
            node12_elemadd = nodes_dict["node12"]
            assert node12_elemadd.op_type == "elemadd"
            node13_const = nodes_dict["node13"]
            assert node13_const.op_type == "const"
            node14_stack = nodes_dict["node14"]
            assert node14_stack.op_type == "stack"
            node15_conv2dtranspose = nodes_dict["node15"]
            assert node15_conv2dtranspose.op_type == "conv2d_transpose"

            # * perform optimization

            assert len(node1_shape.bottom) == 1
            pnode_node1_shape = xmodel.get_xnode_by_name(node1_shape.bottom[0])
            assert pnode_node1_shape is not None
            pnode_node1_shape.top.remove(node1_shape.op_name)

            del nodes_dict["node15"]
            for _, node in nodes_dict.items():
                node.bottom, node.top = [], []
                xmodel.remove_xnode(node)

            node15_conv2dtranspose.bottom.remove(node14_stack.op_name)

    @classmethod
    def replace_conv2dbackpropinput_with_conv2dtranspose_pattern2(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ****************************************************************************************************************************************************
        #                               A
        #                               |
        #                               |
        #              _____________ shape(1) ____________
        #             |                  |                |
        #             |                  |                |
        #             |                  |                |
        #             |                  |                |
        #        stridedslice(2)   stridedslice(5)    stridedslice(8)                    A
        #             |                  |                |                              |
        #   const(3)  |        const(6)  |                |                              |
        #        \    |             \    |                |                 =>           |
        #         elemmul(4)         elemmul(7)           |                              |
        #             |                  |                |                     conv2d_transpose(10)
        #             |        const(9)  |                |
        #             |             \    |                |
        #             ----------------stack(10)-------------
        #                                |
        #                                |
        #                                |
        #                        conv2d_transpose(11)
        #
        #
        #
        # ****************************************************************************************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_shape = nodes_dict["node1"]
            assert node1_shape.op_type == "shape"
            node2_stridedslice = nodes_dict["node2"]
            assert node2_stridedslice.op_type == "stridedslice"
            node3_const = nodes_dict["node3"]
            assert node3_const.op_type == "const"
            node4_elemmul = nodes_dict["node4"]
            assert node4_elemmul.op_type == "elemmul"
            node5_stridedslice = nodes_dict["node5"]
            assert node5_stridedslice.op_type == "stridedslice"
            node6_const = nodes_dict["node6"]
            assert node6_const.op_type == "const"
            node7_elemmul = nodes_dict["node7"]
            assert node7_elemmul.op_type == "elemmul"
            node8_stridedslice = nodes_dict["node8"]
            assert node8_stridedslice.op_type == "stridedslice"
            node9_const = nodes_dict["node9"]
            assert node9_const.op_type == "const"
            node10_stack = nodes_dict["node10"]
            assert node10_stack.op_type == "stack"
            node11_conv2dtranspose = nodes_dict["node11"]
            assert node11_conv2dtranspose.op_type == "conv2d_transpose"

            # * perform optimization

            assert len(node1_shape.bottom) == 1
            pnode_node1_shape = xmodel.get_xnode_by_name(node1_shape.bottom[0])
            assert pnode_node1_shape is not None
            pnode_node1_shape.top.remove(node1_shape.op_name)

            del nodes_dict["node11"]
            for _, node in nodes_dict.items():
                node.bottom, node.top = [], []
                xmodel.remove_xnode(node)

            node11_conv2dtranspose.bottom.remove(node10_stack.op_name)

    @classmethod
    def reduce_shape_stridedslice_stack_to_const(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #             A
        #            / \________
        #           /          |
        #       shape(1)       |                 A
        #         /            |                 |
        #        |             |                 |
        #  stridedslice(2)     |          const  |
        #        |             |     =>      \   |
        #        |  const(3)   |              \  |
        #        |   /         |              reshape
        #     stack(4)         /                 |
        #         \           /                  |
        #          \         /                   |
        #           \       /                    |
        #          reshape(5)                    B
        #               |
        #               |
        #               B
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """

        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_shape = nodes_dict["node1"]
            assert node1_shape.op_type == "shape"
            node2_stridedslice = nodes_dict["node2"]
            assert node2_stridedslice.op_type == "stridedslice"
            node3_const = nodes_dict["node3"]
            assert node3_const.op_type == "const"
            node4_stack = nodes_dict["node4"]
            assert node4_stack.op_type == "stack"
            node5_reshape = nodes_dict["node5"]
            assert node5_reshape.op_type == "reshape"

            # * perform optimization

            # create XModelNodeConst object
            const_xnode = XModelNodeConst(node4_stack.op_name + "_const")
            const_xnode.tensor = node4_stack.outputs_tensor[0].to_numpy()
            const_xnode.infer()
            const_xnode.top = [node5_reshape.op_name]
            idx = node5_reshape.bottom.index(node4_stack.op_name)
            assert idx is not None
            node5_reshape.bottom[idx] = const_xnode.op_name
            xmodel.add_xnode(const_xnode)

            # remove node1, node2, node3 and node4
            pnode = xmodel.get_xnode_by_name(node1_shape.bottom[0])
            assert (
                pnode is not None
            ), f"[ERROR] Not found parent of node ({node1_shape.op_name}): name: {node1_shape.bottom[0]}"
            pnode.top.remove(node1_shape.op_name)
            node1_shape.bottom = node1_shape.top = []
            node2_stridedslice.bottom = node2_stridedslice.top = []
            node3_const.bottom = node3_const.top = []
            node4_stack.bottom = node4_stack.top = []
            xmodel.remove_xnodes(
                [node1_shape, node2_stridedslice, node3_const, node4_stack]
            )

    @classmethod
    def recover_l2_normalize(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #             A
        #             | \________
        #             |          |
        #             |        square(1)                        A
        #             |          |                              |
        #             |          |                              |
        #             |        sum(2)                           |
        #             |          |                  =>          |
        #             |          |  const(3)                    |
        #             |          |   /                     l2_normalize
        #             |       maximum(4)
        #             |          |
        #             |          |
        #             |          |
        #             |      rsqrt(5)
        #             |         /
        #             |________/
        #             |
        #            mul(6)
        #
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """

        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_square = nodes_dict["node1"]
            assert node1_square.op_type == "elemsquare"
            node2_sum = nodes_dict["node2"]
            assert node2_sum.op_type == "reducesum"
            node3_const = nodes_dict["node3"]
            assert node3_const.op_type == "const"
            node4_maximum = nodes_dict["node4"]
            assert node4_maximum.op_type == "elemmax"
            node5_rsqrt = nodes_dict["node5"]
            assert node5_rsqrt.op_type == "elemrsqrt"
            node6_mul = nodes_dict["node6"]
            assert node6_mul.op_type == "elemmul"

            # * perform optimization

            # create XModelNodeL2Normalize object
            node_l2_norm = XModelNodeL2Normalize(node6_mul.op_name + "_l2_norm")

            # bottom
            node_l2_norm.bottom = [node6_mul.bottom[0]]
            # top
            node_l2_norm.top = [x for x in node6_mul.top]

            # axis
            node_l2_norm.axis = node2_sum.axis
            # keep dims
            node_l2_norm.keep_dims = node2_sum.keep_dims
            assert node_l2_norm.keep_dims == True
            # epsilon
            node_l2_norm.epsilon = node3_const.tensor.tolist()[0]

            # add to xmodel
            xmodel.add_xnode(node_l2_norm)

            # update parent's top
            parent_node = xmodel.get_xnode_by_name(node_l2_norm.bottom[0])
            assert parent_node is not None
            assert (node1_square.op_name and node6_mul.op_name) in parent_node.top
            parent_node.top.remove(node1_square.op_name)
            parent_node.top = [node_l2_norm.op_name if i == node6_mul.op_name else i for i in parent_node.top]

            # udpate child's bottom
            if node6_mul.top is not None and len(node6_mul.top) > 0:
                for cname in node6_mul.top:
                    cnode = xmodel.get_xnode_by_name(cname)
                    assert cnode is not None
                    idx = cnode.bottom.index(node6_mul.op_name)
                    cnode.bottom[idx] = node_l2_norm.op_name

            # remove used nodes
            node1_square.bottom = node1_square.top = []
            node2_sum.bottom = node2_sum.top = []
            node3_const.bottom = node3_const.top = []
            node4_maximum.bottom = node4_maximum.top = []
            node5_rsqrt.bottom = node5_rsqrt.top = []
            node6_mul.bottom = node6_mul.top = []
            xmodel.remove_xnodes(
                [
                    node1_square,
                    node2_sum,
                    node3_const,
                    node4_maximum,
                    node5_rsqrt,
                    node6_mul,
                ]
            )

    @classmethod
    def recover_leaky_relu_1(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #                any(0)
        #      ____________|
        #      |           |
        #      |           |
        #      |     elemnegative(2)    const(4)
        #      |           |             /
        #      |           |            /
        #   relu(1)     relu(3)  fixneuron(5)
        #      |           \        /                        any(0)
        #      |            \      /                           |
        #      |           elemmul(6)                          |
        #      |               |                      =>   leaky_relu
        #      |               |                               |
        #      |_______________|                               |
        #             |                                        B
        #             |
        #         elemsub(7)
        #             |
        #             |
        #             B
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node0_any = nodes_dict["node0"]
            node1_relu = nodes_dict["node1"]
            assert isinstance(node1_relu, XModelNodeRelu)
            node2_elemneg = nodes_dict["node2"]
            assert isinstance(node2_elemneg, XModelNodeElemNegative)
            node3_relu = nodes_dict["node3"]
            assert isinstance(node3_relu, XModelNodeRelu)
            node4_const = nodes_dict["node4"]
            assert isinstance(node4_const, XModelNodeConst)
            node5_fix = nodes_dict["node5"]
            assert isinstance(node5_fix, XModelNodeFixNeuron)
            node6_elemmul = nodes_dict["node6"]
            assert isinstance(node6_elemmul, XModelNodeElemMul)
            node7_elemsub = nodes_dict["node7"]
            assert isinstance(node7_elemsub, XModelNodeElemSub)

            # * perform optimization

            op_name = (
                node1_relu.op_name.split("/")[0]
                if "/" in node1_relu.op_name
                else node1_relu.op_name
            )
            # create XModelNodeRelu object
            xnode = XModelNodeRelu(op_name)

            # alpha
            xnode.alpha = node4_const.tensor.tolist()[0]

            # update top and bottom
            xnode.bottom = [node0_any.op_name]
            xnode.top = node7_elemsub.top
            xmodel.add_xnode(xnode)

            # update parent
            idx = node0_any.top.index(node1_relu.op_name)
            node0_any.top[idx] = xnode.op_name
            node0_any.top.remove(node2_elemneg.op_name)

            # update child
            for cname in node7_elemsub.top:
                cnode = xmodel.get_xnode_by_name(cname)
                assert (
                    cnode is not None
                ), f"[ERROR] Not found child node: {cname}. Current node name: {node7_elemsub.op_name}"
                idx = cnode.bottom.index(node7_elemsub.op_name)
                cnode.bottom[idx] = xnode.op_name

            # clear unused nodes
            nodes = []
            node1_relu.bottom = node1_relu.top = []
            nodes.append(node1_relu)
            node2_elemneg.bottom = node2_elemneg.top = []
            nodes.append(node2_elemneg)
            node3_relu.bottom = node3_relu.top = []
            nodes.append(node3_relu)
            node4_const.bottom = node4_const.top = []
            nodes.append(node4_const)
            node5_fix.bottom = node5_fix.top = []
            nodes.append(node5_fix)
            node6_elemmul.bottom = node6_elemmul.top = []
            nodes.append(node6_elemmul)
            node7_elemsub.bottom = node7_elemsub.top = []
            nodes.append(node7_elemsub)
            xmodel.remove_xnodes(nodes)

    @classmethod
    def recover_leaky_relu_2(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #                any(0)
        #      ____________|
        #      |           |
        #      |           |
        #      |           |          const(1)
        #      |           |             /
        #      |           |            /
        #      |           |      fixneuron(2)
        #      |           \        /                        any(0)
        #      |            \      /                           |
        #      |           elemmul(3)                          |
        #      |               |                      =>   leaky_relu
        #      |               |                               |
        #      |_______________|                               |
        #             |                                        B
        #             |
        #         elemmax(4)
        #             |
        #             |
        #             B
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node0_any = nodes_dict["node0"]
            assert isinstance(node0_any, XModelNode)
            node1_const = nodes_dict["node1"]
            assert isinstance(node1_const, XModelNodeConst)
            node2_fix = nodes_dict["node2"]
            assert isinstance(node2_fix, XModelNodeFixNeuron)
            node3_elemmul = nodes_dict["node3"]
            assert isinstance(node3_elemmul, XModelNodeElemMul)
            node4_elemmax = nodes_dict["node4"]
            assert isinstance(node4_elemmax, XModelNodeElemMax)

            # * perform optimization

            op_name = node4_elemmax.op_name
            # create XModelNodeRelu object
            xnode = XModelNodeRelu(op_name)

            # alpha
            xnode.alpha = node1_const.tensor.tolist()[0]

            # update top and bottom
            xnode.bottom = [node0_any.op_name]
            xnode.top = [x for x in node4_elemmax.top]

            # update parent
            node0_any.top.remove(node3_elemmul.op_name)

            # clear unused nodes
            nodes = []
            node1_const.bottom = node1_const.top = []
            nodes.append(node1_const)
            node2_fix.bottom = node2_fix.top = []
            nodes.append(node2_fix)
            node3_elemmul.bottom = node3_elemmul.top = []
            nodes.append(node3_elemmul)
            node4_elemmax.bottom = node4_elemmax.top = []
            nodes.append(node4_elemmax)
            xmodel.remove_xnodes(nodes)

            xmodel.add_xnode(xnode)

    @classmethod
    def merge_round_typecast(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #             A
        #             |
        #             |                  A
        #        elemround(1)            |
        #             |                  |
        #             |          => round_typecast
        #        type_cast(2)            |
        #             |                  |
        #             |                  B
        #             B
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_elemround = nodes_dict["node1"]
            assert isinstance(node1_elemround, XModelNodeElemRound)
            node2_type_cast = nodes_dict["node2"]
            assert isinstance(node2_type_cast, XModelNodeTypeCast)

            # * perform optimization

            # create an XModelNode object as round_typecast
            xnode = XModelNode(
                op_name=node1_elemround.op_name + "_round_typecast",
                op_type="round_typecast",
            )
            xnode.tmp_params["round_mode"] = node1_elemround.round_mode
            xnode.tmp_params["src_dtype"] = node2_type_cast.src_dtype
            xnode.tmp_params["dst_dtype"] = node2_type_cast.dst_dtype

            # update bottom
            if node1_elemround.bottom is not None and len(node1_elemround.bottom) > 0:
                assert len(node1_elemround.bottom) == 1
                xnode.bottom = [node1_elemround.bottom[0]]
                # update parent node
                pnode = xmodel.get_xnode_by_name(xnode.bottom[0])
                assert pnode is not None
                idx = pnode.top.index(node1_elemround.op_name)
                pnode.top[idx] = xnode.op_name

            # update top
            if node2_type_cast.top is not None and len(node2_type_cast.top) > 0:
                xnode.top = [x for x in node2_type_cast.top]
                # update child nodes
                for cname in xnode.top:
                    cnode = xmodel.get_xnode_by_name(cname)
                    assert cnode is not None
                    idx = cnode.bottom.index(node2_type_cast.op_name)
                    cnode.bottom[idx] = xnode.op_name

            # inputs_tensor and outputs_tensor
            if (
                node1_elemround.inputs_tensor is not None
                and len(node1_elemround.inputs_tensor) > 0
            ):
                assert len(node1_elemround.inputs_tensor) == 1
                xnode.inputs_tensor = [node1_elemround.inputs_tensor[0]]
            if (
                node2_type_cast.outputs_tensor is not None
                and len(node2_type_cast.outputs_tensor) > 0
            ):
                assert len(node2_type_cast.outputs_tensor) == 1
                xnode.outputs_tensor = [node2_type_cast.outputs_tensor[0]]

            xmodel.add_xnode(xnode)

            node1_elemround.bottom = node1_elemround.top = []
            node2_type_cast.bottom = node1_elemround.top = []
            xmodel.remove_xnodes([node1_elemround, node2_type_cast])

            xmodel.topsort()

    @classmethod
    def remove_shape_from_random_standard_normal(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #
        #
        #
        #           shape(1)
        #             |
        #             |                => random_standard_normal(2)
        #  random_standard_normal(2)               |
        #             |                            |
        #             |                            B
        #             B
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_shape = nodes_dict["node1"]
            assert isinstance(node1_shape, XModelNodeShape)
            node2_random_standard_normal = nodes_dict["node2"]
            assert isinstance(
                node2_random_standard_normal, XModelNodeRandomStandardNormal
            )

            # * perform optimization

            assert node1_shape.bottom is None or len(node1_shape.bottom) == 0

            assert node1_shape.outputs_tensor[0].ndims == 1
            node2_random_standard_normal.shape = node1_shape.outputs_tensor[0].tolist()
            node2_random_standard_normal.bottom.remove(node1_shape.op_name)

            node1_shape.bottom = []
            node1_shape.top = []
            xmodel.remove_xnode(node1_shape)

    @classmethod
    def recover_hard_sigmoid(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #
        #             A
        #             | const(2)                      A
        #             | /                             |
        #           add(1)                            |
        #             |                          hard_sigmoid
        #             |                               |
        #          relu6(3)           =>              | const(7)
        #             | const(5)                      |  /
        #             | /                           mul(6)
        #         mul,div(4)
        #             | const(7)
        #             | /
        #           mul(6)
        #
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """

        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_add = nodes_dict["node1"]
            assert node1_add.op_type == "elemadd"
            node2_const = nodes_dict["node2"]
            assert node2_const.op_type == "const"
            node3_relu6 = nodes_dict["node3"]
            assert node3_relu6.op_type == "relu6"
            node4_mul_or_div = nodes_dict["node4"]
            assert node4_mul_or_div.op_type in ["elemmul", "elemrealdiv"]
            node5_const = nodes_dict["node5"]
            assert node5_const.op_type == "const"
            node6_mul = nodes_dict["node6"]
            assert node6_mul.op_type == "elemmul"
            node7_const = nodes_dict["node7"]
            assert node7_const.op_type == "const"

            # * perform optimization
            # create XModelNodeSigmoid object
            node_hard_sigmoid = XModelNodeSigmoid(node6_mul.op_name + "_hard_sigmoid")
            node_hard_sigmoid.tmp_params["hard_sigmoid"] = True

            # bottom
            node_hard_sigmoid.bottom = [node1_add.bottom[0]]
            # top
            node_hard_sigmoid.top = [node6_mul.op_name]

            # add to xmodel
            xmodel.add_xnode(node_hard_sigmoid)

            # update parent's top
            parent_node = xmodel.get_xnode_by_name(node1_add.bottom[0])
            assert parent_node is not None
            idx = parent_node.top.index(node1_add.op_name)
            parent_node.top[idx] = node_hard_sigmoid.op_name

            # update child
            idx = node6_mul.bottom.index(node4_mul_or_div.op_name)
            node6_mul.bottom[idx] = node_hard_sigmoid.op_name

            # remove used nodes
            node1_add.bottom = node1_add.top = []
            node2_const.bottom = node2_const.top = []
            node3_relu6.bottom = node3_relu6.top = []
            node4_mul_or_div.bottom = node4_mul_or_div.top = []
            node5_const.bottom = node5_const.top = []
            xmodel.remove_xnodes(
                [
                    node1_add,
                    node2_const,
                    node3_relu6,
                    node4_mul_or_div,
                    node5_const
                ]
            )

    @classmethod
    def recover_hard_softmax(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #           type_cast(1)       type_cast(1)
        #               |                   |
        #           65 nodes      =>   hard_softmax
        #
        #           type_cast:  'node1','node10','node12','node15','node17','node18',
        #                       'node21','node23','node24','node27','node29','node31',
        #                       'node37','node38','node40','node41','node44','node47',
        #                       'node48','node51','node53','node55','node57','node60',
        #                       'node61','node64'
        #           const:      'node2','node5', 'node7','node11','node14','node20',
        #                       'node26', 'node33','node34','node42','node45','node50',
        #                       'node56'
        #           elemmul:            'node3','node8','node13','node19','node25',
        #                               'node30','node46','node49','node54','node58',
        #                               'node62','node65'
        #           elemadd:            'node16','node22','node28','node63'
        #           elemsub:            'node9','node43','node52','node59'
        #           permute:            'node32','node36'
        #           elemrealpow:        'node6'
        #           reducesum:          'node39'
        #           elemfloor:          'node4'
        #           unsortedsegmentsum: 'node35'
        #
        # *******************************************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance to be optimized
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """

        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1 = nodes_dict["node1"]
            node65 = nodes_dict["node65"]

            type_cast_node_list = ['node1','node10','node12','node15','node17','node18',
                                    'node21','node23','node24','node27','node29','node31','node37',
                                    'node38','node40','node41','node44','node47','node48',
                                    'node51','node53','node55','node57','node60','node61','node64']
            const_node_list = ['node2','node5', 'node7','node11','node14','node20','node26',
                                'node33','node34','node42','node45','node50','node56']
            elemmul_node_list = ['node3','node8','node13','node19','node25','node30',
                                'node46','node49','node54','node58','node62','node65']
            elemadd_node_list = ['node16','node22','node28','node63']
            elemsub_node_list = ['node9','node43','node52','node59']
            permute_node_list = ['node32','node36']

            for node_name, node in nodes_dict.items():
                if node_name in type_cast_node_list:
                    assert node.op_type == "type_cast"
                elif node_name in const_node_list:
                    assert node.op_type == "const"
                elif node_name in elemmul_node_list:
                    assert node.op_type == "elemmul"
                elif node_name in elemadd_node_list:
                    assert node.op_type == "elemadd"
                elif node_name in elemsub_node_list:
                    assert node.op_type == "elemsub"
                elif node_name in permute_node_list:
                    assert node.op_type == "permute"
                elif node_name == "node6":
                    assert node.op_type == "elemrealpow"
                elif node_name == "node39":
                    assert node.op_type == "reducesum"
                elif node_name == "node4":
                    assert node.op_type == "elemfloor"
                elif node_name == "node35":
                    assert node.op_type == "unsortedsegmentsum"

            # * perform optimization
            # create XModelNodeSigmoid object
            node_hard_softmax = XModelNodeSoftmax(node65.op_name + "_hard_softmax")
            node_hard_softmax.tmp_params["hard_softmax"] = True
            node_hard_softmax.tmp_params["type"] = "poly"
            node_hard_softmax.axis = -1

            # update nodes top and bottom
            node_hard_softmax.bottom = [node1.op_name]
            node1.top = [node_hard_softmax.op_name]
            node_hard_softmax.top = [x for x in node65.top]
            for cname in node65.top:
                cnode = xmodel.get_xnode_by_name(cname)
                assert cnode is not None
                idx = cnode.bottom.index(node65.op_name)
                cnode.bottom[idx] = node_hard_softmax.op_name

            # add to xmodel
            xmodel.add_xnode(node_hard_softmax)

            # remove used nodes
            for _, node in nodes_dict.items():
                if _ == "node1":
                    continue
                node.bottom, node.top = [], []
                xmodel.remove_xnode(node)
