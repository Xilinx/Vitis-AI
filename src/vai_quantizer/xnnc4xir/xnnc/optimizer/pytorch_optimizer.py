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

from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import XModelNode, XModelNodeConst
from xnnc.optimizer.algorithm.subgraph_isomorph_vf2 import DiGraphMatcher

# create logger
logger = logging.getLogger(__name__)


class PyTorchOptimizer(object):
    @classmethod
    def run(cls, xmodel: XModel, pattern_dict: Dict[str, Any]) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."
        assert pattern_dict is not None, "'pattern_dict' should not be None."

        patterns = pattern_dict.get("patterns")

        # * run optimizations based on graph-subgraph matcher
        if patterns is not None:
            need_infer_shape = False
            for subgraph in patterns:
                pname = subgraph.name
                if hasattr(PyTorchOptimizer, pname):
                    # find subgraph matching
                    matcher = DiGraphMatcher(xmodel, subgraph)
                    matches = matcher.search_matching()
                    if matches is not None:
                        # call optimization method
                        func = getattr(PyTorchOptimizer, pname)
                        func(xmodel, matches)
                        if not need_infer_shape:
                            need_infer_shape = True
                else:
                    print(f"[WARNING] Not found optimization method: {pname}")

            if need_infer_shape:
                # perform topsort
                xmodel.topsort()

                # infer shape
                xmodel.infer(layout=xmodel.layout, disable_pbar=True)

    @classmethod
    def merge_size_listconstruct_in_reshape(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # *******************************************
        #  size(1)                A           A
        #        \               /            |
        #   listconstruct(2)    /    =>       |  const
        #              \       /              |  /
        #              reshape(3)          reshape
        #                   |                 |
        #                   B                 B
        # *******************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        print(f"[OPT] merge Size and ListConstruct into Reshape")

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_size = nodes_dict.get("node1")
            assert node1_size.op_type == "size"
            node2_lc = nodes_dict.get("node2")
            assert node2_lc.op_type == "listconstruct"
            node3_reshape = nodes_dict.get("node3")
            assert node3_reshape.op_type == "reshape"

            # * perform optimization

            # new shape
            new_shape = []
            for i in range(len(node1_size.dims)):
                dim = node1_size.dims[i]
                new_shape.append(node1_size.inputs_tensor_shape[0][dim])
            # append the info in listconstruct
            new_shape = new_shape + node2_lc.tmp_params.get("constants")

            # create XModelNodeConst node
            const_node = XModelNodeConst(node2_lc.op_name + "_shape")
            const_node.top = node2_lc.top
            const_node.tensor = np.array(new_shape, dtype=np.int32)
            const_node.outputs_tensor_shape = [list(const_node.tensor.shape)]
            idx = node3_reshape.bottom.index(node2_lc.op_name)
            node3_reshape.bottom[idx] = const_node.op_name
            xmodel.add_xnode(const_node)

            # remove Shape and ListConstruct nodes
            node1_size.top = []
            node2_lc.bottom = node2_lc.top = []
            xmodel.remove_xnode(node1_size)
            xmodel.remove_xnode(node2_lc)

    @classmethod
    def merge_listconstruct_concat_in_concat(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # *******************************************
        #          A                     A
        #          |                     |
        #   listconstruct(1)             |
        #          |            =>    concat
        #       concat(2)                |
        #          |                     |
        #          B                     B
        # *******************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        print(f"[OPT] merge ListConstruct and Concat into Concat")

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_lc = nodes_dict.get("node1")
            assert node1_lc.op_type == "listconstruct"
            node2_concat = nodes_dict.get("node2")
            assert node2_concat.op_type == "concat"

            # * perform optimization
            xmodel.remove_xnode(node1_lc)

    @classmethod
    def compute_scale_of_upsample_nearest(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ***************************************************
        #                A ___________
        #             /    \         |
        #            /      \        |
        #       size(1)   size(6)    |
        #          |         |       |           A
        #    to_dtype(2)  to_dtype(7)|           |
        #          |         |       |           |
        #        mul(3)    mul(8)    |     =>    |
        #          |         |       |           |
        #    to_dtype(4)  to_dtype(9)|           |
        #          |         |       |           |
        #       floor(5)  floor(10)  |           |
        #           \       /        |        upsample
        #        listconstruct(11)   |
        #               |            |
        #          upsample(12) _____|
        # ***************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        print(f"[OPT] compute scale of Upsample nearest from the preceeding nodes")

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node12_upsample = nodes_dict.get("node12")
            assert node12_upsample.op_type == "upsample"

            # * perform optimization
            assert len(node12_upsample.bottom) == 2
            pnode = xmodel.get_xnode_by_name(node12_upsample.bottom[0])
            assert (
                pnode is not None
            ), f"'pnode' should not be None: {node12_upsample.bottom[0]}"
            assert len(pnode.top) > 1

            # update scale
            assert (
                node12_upsample.scale is None
            ), f"The scale property of XModelNodeUpsample node is not None: {node10_upsample.scale}."
            scale: List[float] = []
            input_tensor_shape = node12_upsample.inputs_tensor_shape[0]
            output_tensor_shape = node12_upsample.outputs_tensor_shape[0]
            assert len(input_tensor_shape) == len(output_tensor_shape)
            for i in range(len(input_tensor_shape)):
                scale.append(input_tensor_shape[i] * 1.0 / output_tensor_shape[i])
            node12_upsample.scale = scale

            pnode.top = [node12_upsample.op_name]
            node12_upsample.bottom = [pnode.op_name]

            # remove unused nodes
            del nodes_dict["node12"]
            for _, xnode in nodes_dict.items():
                xnode.bottom = xnode.top = []
                xmodel.remove_xnode(xnode)

    @classmethod
    def remove_contiguous(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ***************************************************
        #      A                        A
        #      |                        |
        #      |                        |
        #    node1(contiguous)     =>   |
        #      |                        |
        #      |                        |
        #      B                        B
        # ***************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        print(f"[OPT] remove contiguous node")

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_contiguous = nodes_dict.get("node1")
            assert node1_contiguous.op_type == "contiguous"

            xmodel.remove_xnode(node1_contiguous)

    @classmethod
    def remove_contiguous(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ***************************************************
        #      A                        A
        #      |                        |
        #      |                        |
        #    node1(permute)     =>      |
        #      |                        |
        #      |                        |
        #      B                        B
        # ***************************************************

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        matches : List[Dict[XModelNode, XModelNode]]
            matching subgraphs on XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."
        assert matches is not None, "'matches' should not be None."

        print(f"[OPT] remove contiguous node")

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_permute = nodes_dict.get("node1")
            assert node1_permute.op_type == "permute"

            xmodel.remove_xnode(node1_permute)
