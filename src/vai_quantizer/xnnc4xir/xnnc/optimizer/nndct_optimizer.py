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
from typing import Any, Dict, List, NoReturn

import numpy as np
from tqdm import tqdm
from xnnc.ir.enums import Layout
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *
from xnnc.optimizer.algorithm.subgraph_isomorph_vf2 import DiGraphMatcher
from xnnc.tensor.xtensor import XTensor

# create logger
logger = logging.getLogger(__name__)


class NNDCTOptimizer(object):
    @classmethod
    def run(cls, xmodel: XModel, pattern_dict: Dict[str, Any]) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."
        assert pattern_dict is not None, "'pattern_dict' should not be None."

        level = pattern_dict.get("level")
        patterns = pattern_dict.get("patterns")

        if level == "xnnc":
            NNDCTOptimizer.remove_subgraphs_with_shape_op(xmodel)

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
                if hasattr(NNDCTOptimizer, pname):
                    # find subgraph matching
                    matcher = DiGraphMatcher(xmodel, subgraph)
                    matches = matcher.search_matching()
                    if matches is not None:
                        # call optimization method
                        func = getattr(NNDCTOptimizer, pname)
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
    def remove_subgraphs_with_shape_op(cls, xmodel: XModel) -> NoReturn:
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
                    # const_xnode.infer()
                    # xnode.inputs_tensor.append(const_xnode.outputs_tensor[0])
                    xnode.shape = []
                    # shape_before = xnode.outputs_tensor[0].shape
                    # xnode.infer()
                    # assert shape_before == xnode.outputs_tensor[0].shape
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
                        # const_xnode.infer()
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
                # const_xnode.infer()
                # if len(xnode.bottom) != len(xnode.inputs_tensor):
                #     xnode.inputs_tensor.append(const_xnode.outputs_tensor[0])

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
    def reduce_shape_before_reshape(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #             A      B
        #             |      |
        #             |   shape(1)               A      B
        #             |    /                     |
        #             |   /                      |
        #          reshape(2)          =>     reshape(2)
        #             |                          |
        #             |                          |
        #             |                          C
        #             C
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
            node1_shape = nodes_dict["node1"]
            assert isinstance(node1_shape, XModelNodeShape)
            node2_reshape = nodes_dict["node2"]
            assert isinstance(node2_reshape, XModelNodeReshape)

            # * perform optimization

            # remove the edge between shape node and its parent
            node0_any.top.remove(node1_shape.op_name)
            node1_shape.bottom = []

            # convert shape node into const node with new shape info
            const_xnode = XModelNodeConst(node1_shape.op_name)
            const_xnode.layout = node1_shape.layout
            const_xnode.tensor = node1_shape.outputs_tensor[0]
            const_xnode.outputs_tensor = []
            const_xnode.top = [x for x in node1_shape.top]
            # const_xnode.infer()

            # remove shape node from graph
            node1_shape.top = []
            xmodel.remove_xnode(node1_shape)

            # add const node into graph
            xmodel.add_xnode(const_xnode)

        xmodel.topsort()

    @classmethod
    def merge_const_fix_elemadd_in_conv2d_family(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        1. replace ElemAdd with BiasAdd;
        2. merge Const-Fix-BiasAdd into conv2d_family ops

        # ******************************************************************************
        # const(1)            A                           A
        #     \              /                            |
        #     fix(2)     conv2d_family (3)                |
        #          \     /                   =>     conv2d_family (3)
        #         elemadd(4)                              |
        #             |                                   |
        #             B                                   B
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
            node1_const = nodes_dict["node1"]
            assert isinstance(node1_const, XModelNodeConst)
            node2_fix = nodes_dict["node2"]
            assert isinstance(node2_fix, XModelNodeFixNeuron)
            node3_conv2d = nodes_dict["node3"]
            assert node3_conv2d.op_type in [
                "conv2d",
                "depthwise_conv2d",
                "conv2d_transpose",
            ]
            node4_elemadd = nodes_dict["node4"]
            assert isinstance(node4_elemadd, XModelNodeElemAdd)

            # * perform optimization

            node3_conv2d.bias = node1_const.tensor
            for key in node3_conv2d.quant_bias.keys():
                node3_conv2d.quant_bias[key] = node2_fix.quant_in[key]
            node3_conv2d.bias_term = True

            # remove node1 and node2
            for n in [node1_const, node2_fix, node4_elemadd]:
                xmodel.remove_xnode(n)

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
            const_xnode.tensor = node4_stack.outputs_tensor[0]
            const_xnode.init_layout = node4_stack.init_layout
            const_xnode.layout = node4_stack.layout
            # const_xnode.infer()
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
    def merge_into_dilated_conv2d_family(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #
        #               A
        #               |
        #               |
        #               |
        #       spacetobatch (node1)                     A
        #               |                                |
        #               |                                |
        #               |                                |
        #          conv2d (node2)        =>       dilated conv2d
        #               |                                |
        #               |                                |
        #               |                                |
        #       batchtospace (node3)                     B
        #               |
        #               |
        #               |
        #               B
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
            node1_spacetobatch = nodes_dict["node1"]
            assert isinstance(node1_spacetobatch, XModelNodeSpaceToBatchND)
            node2_conv2d = nodes_dict["node2"]
            assert isinstance(node2_conv2d, XModelNodeConv2d) or isinstance(
                node2_conv2d, XModelNodeConv2dDepthwise
            )
            node3_batchtospace = nodes_dict["node3"]
            assert isinstance(node3_batchtospace, XModelNodeBatchToSpaceND)

            # * perform optimization

            # update dilation
            assert node1_spacetobatch.block_shape == node3_batchtospace.block_shape
            node2_conv2d.dilation[-2:] = node1_spacetobatch.block_shape
            node2_conv2d.pad_mode = PadMode.SAME

            # remove both space_to_batch and batch_to_space from xmodel
            xmodel.remove_xnodes([node1_spacetobatch, node3_batchtospace])
