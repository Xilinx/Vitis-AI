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
from xnnc.ir.xnode import (
    XModelNode,
    XModelNodeAvgPool,
    XModelNodeConst,
    XModelNodeElemMul,
    XModelNodeFixNeuron,
    XModelNodeSlice,
    XModelNodeStridedSlice,
)
from xnnc.optimizer.algorithm.subgraph_isomorph_vf2 import DiGraphMatcher
from xnnc.tensor.xtensor import XTensor

# create logger
logger = logging.getLogger(__name__)


class CaffeOptimizer(object):
    @classmethod
    def run(cls, xmodel: XModel, pattern_dict: Dict[str, Any]) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."
        assert pattern_dict is not None, "'pattern_dict' should not be None."

        level = pattern_dict.get("level")
        patterns = pattern_dict.get("patterns")

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
                if hasattr(CaffeOptimizer, pname):
                    # find subgraph matching
                    matcher = DiGraphMatcher(xmodel, subgraph)
                    matches = matcher.search_matching()
                    if matches is not None:
                        # call optimization method
                        func = getattr(CaffeOptimizer, pname)
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
    def split_slice_into_multiple_stridedslice(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:


        # ******************************************************************************
        #             A                            ___A___
        #             |                           /       \
        #             |                          /         \
        #           slice(1)     =>     stridedslice      stridedslice
        #           /   \                    |                  |
        #          /     \                   |                  |
        #         B       C                  B                  C
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
            node1_slice = nodes_dict["node1"]
            assert isinstance(node1_slice, XModelNodeSlice)

            # * perform optimization

            # compute ends
            in_shape = node1_slice.inputs_tensor[0].shape
            ends = [node1_slice.slice_points[0]]
            cuts = len(node1_slice.slice_points)
            if cuts > 1:
                for i in range(1, cuts):
                    ends.append(ends[-1] + node1_slice.slice_points[i])
            ends.append(in_shape[node1_slice.axis])

            # compute starts
            starts = [0]
            for i in range(0, len(ends) - 1):
                starts.append(ends[i])

            # create XModelNodeStridedSlice nodes
            assert len(starts) == len(ends) == len(node1_slice.top)
            stridedslice_nodes = []
            begin_idx = [0] * len(in_shape)
            end_idx = [x for x in in_shape]
            new_node_names = []
            for i in range(len(node1_slice.top)):
                begin_idx[node1_slice.axis] = starts[i]
                end_idx[node1_slice.axis] = ends[i]
                ss_node = XModelNodeStridedSlice(
                    node1_slice.op_name + f"_{i}",
                    begin=[x for x in begin_idx],
                    end=[x for x in end_idx],
                    strides=[1] * len(in_shape),
                )
                # update bottom and top
                ss_node.bottom = node1_slice.bottom
                ss_node.top = [node1_slice.top[i]]
                out_shape = [x - y for x, y in zip(end_idx, begin_idx)]
                ss_node.outputs_tensor = [
                    XTensor(np.zeros(out_shape).astype(np.float32))
                ]
                xmodel.add_xnode(ss_node)
                stridedslice_nodes.append(ss_node)
                new_node_names.append(ss_node.op_name)

            # update topo
            pname = node1_slice.bottom[0]
            pnode = xmodel.get_xnode_by_name(pname)
            assert pnode is not None
            idx = pnode.top.index(node1_slice.op_name)
            pnode.top = pnode.top[:idx] + new_node_names + pnode.top[idx + 1 :]

            for i in range(len(node1_slice.top)):
                cname = node1_slice.top[i]
                cnode = xmodel.get_xnode_by_name(cname)
                assert cnode is not None
                idx = cnode.bottom.index(node1_slice.op_name)
                cnode.bottom[idx] = stridedslice_nodes[i].op_name

            node1_slice.bottom = []
            node1_slice.top = []
            xmodel.remove_xnode(node1_slice)

        xmodel.topsort()

    @classmethod
    def patch_avgpool_followed_by_fix(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:


        # ******************************************************************************
        #
        #         avgpool(1)          avgpool(1)     const
        #            |                       \       /
        #            |                        \     /
        #            |           =>           elem_mul
        #            |                           |
        #            |                           |
        #        fixneuron(2)               fixneuron(2)
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
            node1_avgpool = nodes_dict["node1"]
            assert isinstance(node1_avgpool, XModelNodeAvgPool)
            node2_fix = nodes_dict["node2"]
            assert isinstance(node2_fix, XModelNodeFixNeuron)

            assert len(node1_avgpool.top) == 1

            # * perform optimization

            if node1_avgpool.kernel_size[0] == node1_avgpool.kernel_size[
                1
            ] and node1_avgpool.kernel_size[0] in [3, 5, 6, 7, 14]:
                # data
                if node1_avgpool.kernel_size[0] == 3:
                    data = 9.0 * 7 / 64
                elif node1_avgpool.kernel_size[0] == 5:
                    data = 25.0 * 10 / 256
                elif node1_avgpool.kernel_size[0] == 6:
                    data = 36.0 * 7 / 256
                elif node1_avgpool.kernel_size[0] == 7:
                    data = 49.0 * 21 / 1024
                else:
                    data = 196.0 * 21 / 4096

                # create XModelNodeConst object
                const_xnode = XModelNodeConst(node1_avgpool.op_name + "_const")
                const_xnode.tensor = XTensor(
                    np.array([data], dtype=np.float32),
                    format=node1_avgpool.inputs_tensor[0].data_format,
                )

                # add XModelNodeElemMul object
                mul_xnode = XModelNodeElemMul(node1_avgpool.op_name + "_elemmul")
                mul_xnode.bottom = [node1_avgpool.op_name, const_xnode.op_name]
                mul_xnode.top = [node2_fix.op_name]
                node1_avgpool.top = [mul_xnode.op_name]
                const_xnode.top = [mul_xnode.op_name]
                node2_fix.bottom = [mul_xnode.op_name]

                # update xmodel
                xmodel.add_xnodes([const_xnode, mul_xnode])

    @classmethod
    def remove_silence_branch(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:


        # ******************************************************************************
        #
        #            A----C               A
        #            |                    |
        #            |                    |
        #            B           =>       |
        #            |                    |
        #            |                    |
        #         silence                 C
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

        def find(node, adict):
            # case1: the target branch is one of current node's branches
            if node.top and len(node.top) > 1:
                node.top = list(filter(lambda cname: cname not in adict, node.top))
                return

            if node.op_name not in adict:
                adict[node.op_name] = node

            # case2: current node has one or more parent nodes
            if node.bottom and len(node.bottom) > 0:
                for pname in node.bottom:
                    pnode = xmodel.get_xnode_by_name(pname)
                    assert pnode is not None
                    find(pnode, adict)

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_silence = nodes_dict["node1"]
            assert isinstance(node1_silence, XModelNode) and node1_silence.op_type == "silence"
            assert node1_silence.top is None or len(node1_silence.top) == 0

            # * perform optimization

            print(f"[WARNING] Found Caffe silence layer: {node1_silence.op_name}. Removing the silence branch ...")
            adict = {}
            find(node1_silence, adict)

            # remove silence branch
            for _, xnode in adict.items():
                xnode.bottom = []
                xnode.top = []
            xmodel.remove_xnodes(adict.values())
