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
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *
from xnnc.ir.enums import Layout
from xnnc.tensor.xtensor import XTensor
from xnnc.optimizer.algorithm.subgraph_isomorph_vf2 import DiGraphMatcher

# create logger
logger = logging.getLogger(__name__)


class TF2Optimizer(object):
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
                if hasattr(TF2Optimizer, pname):
                    # find subgraph matching
                    matcher = DiGraphMatcher(xmodel, subgraph)
                    matches = matcher.search_matching()
                    if matches is not None:
                        # call optimization method
                        func = getattr(TF2Optimizer, pname)
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
    def compute_shape_const_for_custom_upsamplelike(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #
        #     node1(any)      node2(any)              node1(any)
        #       |                 |                      |
        #       |                 |                      |
        #       |                 |                      |       const
        #        \               /                       |        /
        #         \             /                        |       /
        #      node3(resize_bilinear)       =>    node3(resize_bilinear)
        #                |                                  |
        #                |                                  |
        #                |                                  |
        #                |                                  |
        #              node4                              node4
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
            node1_any = nodes_dict["node1"]
            assert isinstance(node1_any, XModelNode)
            node2_any = nodes_dict["node2"]
            assert isinstance(node2_any, XModelNode)
            node3_resize_bilinear = nodes_dict["node3"]
            assert (
                isinstance(node3_resize_bilinear, XModelNodeResize)
                and node3_resize_bilinear.mode == "bilinear"
            )

            # * perform optimization

            # create XModelNodeConst object
            assert len(node3_resize_bilinear.inputs_tensor) == 2
            out_size = node3_resize_bilinear.inputs_tensor[1].shape
            if len(out_size) == 4:
                out_size = (
                    out_size[1:3]
                    if node3_resize_bilinear.layout == "NHWC"
                    else out_size[-2]
                )
                tensor = XTensor(np.array(out_size, dtype=np.int32))
            elif len(out_size) == 2:
                tensor = XTensor(np.array(out_size, dtype=np.int32))
            else:
                raise ValueError(f"[ERROR] Unsupported out_size: {out_size}")

            const_xnode = XModelNodeConst(node3_resize_bilinear.bottom[1] + "_const")
            const_xnode.tensor = tensor
            const_xnode.top = [node3_resize_bilinear.op_name]

            # update bottom of node3_resize_bilinear
            pname = node3_resize_bilinear.bottom.pop()
            pnode = node1_any if pname == node1_any.op_name else node2_any
            pnode.top.remove(node3_resize_bilinear.op_name)
            assert len(pnode.top) > 0

            node3_resize_bilinear.bottom.append(const_xnode.op_name)
            xmodel.add_xnode(const_xnode)

    @classmethod
    def xir_opt_decompose_hard_sigmoid(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #
        #                A                            A
        #                |                            |
        #                |                            |
        #                |                   node1(hard_sigmoid)
        #                |                            |
        #                |                            |  node2(const)
        #      node1(hard_sigmoid)    =>              |   /
        #                |                            |  /
        #                |                       node3(elemmul)
        #                |                            |
        #                |                            |
        #                B                            B
        #
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
            node1_hard_sigmoid = nodes_dict["node1"]
            assert isinstance(node1_hard_sigmoid, XModelNodeSigmoid)
            if "hard_sigmoid" not in node1_hard_sigmoid.tmp_params:
                continue

            # * perform optimization

            # create an XModelNodeConst object
            node2_const = XModelNodeConst(node1_hard_sigmoid.op_name + "_const")
            tensor = np.array([6 * 2731 / 2 ** 14], dtype=np.float32)
            node2_const.tensor = XTensor(
                tensor, format=node1_hard_sigmoid.outputs_tensor[0].data_format
            )

            # create an XModelNodeElemMul object
            node3_elemmul = XModelNodeElemMul(node1_hard_sigmoid.op_name + "_elemmul")

            # update bottom and top
            node3_elemmul.bottom = [node1_hard_sigmoid.op_name, node2_const.op_name]
            node3_elemmul.top = [x for x in node1_hard_sigmoid.top]
            node2_const.top = [node3_elemmul.op_name]
            node1_hard_sigmoid.top = [node3_elemmul.op_name]

            for cname in node3_elemmul.top:
                cnode = xmodel.get_xnode_by_name(cname)
                assert cnode is not None
                idx = cnode.bottom.index(node1_hard_sigmoid.op_name)
                cnode.bottom[idx] = node3_elemmul.op_name

            # update xmodel with new nodes
            xmodel.add_xnodes([node2_const, node3_elemmul])

    @classmethod
    def xir_opt_decompose_global_avgpool2d(
        cls, xmodel: XModel, matches: List[Dict[XModelNode, XModelNode]]
    ) -> NoReturn:
        """Perform the following optimization:

        # ******************************************************************************
        #
        #                A                            A
        #                |                            |
        #                |                            |
        #                |                     node1(avgpool)
        #                |                            |
        #                |                            |  node2(const)
        #         node1(avgpool)    =>                |   /
        #                |                            |  /
        #                |                       node3(elemmul)
        #                |                            |
        #                |                            |
        #                B                            B
        #
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

        # the formular below from @yuxing
        def get_rec_map():
            rec_map = {}
            for kw in range(1, 257):
                for kh in range(kw, 257):
                    rec = kw * kh
                    max_val = math.ceil(math.log(rec * 128, 2))
                    # 1 / rec almost equal k / 2**n
                    k = 0
                    n = 0
                    diff = 1
                    for n_ in range(0, max_val):
                        k_ = round(pow(2, n_) / rec)
                        k_ = k_ if k_ > 0 else 1
                        diff_ = abs(k_ / pow(2, n_) - 1 / rec)
                        if diff_ < diff:
                            k = k_
                            diff = diff_
                            n = n_
                    rec_map[rec] = k / pow(2, n)

        def _gcd_f(a, b):
            """Get the greatest common dividisor."""
            return a if b == 0 else _gcd_f(b, a % b)

        def _lcm_f(a, b):
            """Get the least common multiple."""
            return int((a * b) / _gcd_f(a, b))

        def _get_dpu_kernel_size(kh, kw):
            """For global_average_pooling, DPU will do padding to replace rectangle
            kernel to squre kernel."""
            new_k = _lcm_f(kh, kw)
            return new_k, new_k

        def _get_avgpool_scale(kw, kh):
            kernel = [kw, kh]
            # the formula from @yuxing
            if kernel == [1, 1]:
                value = 1.0
            elif kernel == [2, 2]:
                value = 1.0 / pow(2, 2)
            elif kernel == [3, 3]:
                value = 7.0 / pow(2, 6)
            elif kernel == [4, 4]:
                value = 1.0 / pow(2, 4)
            elif kernel == [5, 5]:
                value = 10.0 / pow(2, 8)
            elif kernel == [6, 6]:
                value = 7.0 / pow(2, 8)
            elif kernel == [7, 7]:
                value = 21.0 / pow(2, 10)
            elif kernel == [14, 14]:
                value = 21.0 / pow(2, 12)
            else:
                rec = kernel[0] * kernel[1]
                max_val = math.ceil(math.log(rec * 128, 2))
                # 1 / rec almost equal k / 2**n
                k = 0
                n = 0
                diff = 1
                for n_ in range(0, max_val):
                    k_ = round(pow(2, n_) / rec)
                    k_ = k_ if k_ > 0 else 1
                    diff_ = abs(k_ / pow(2, n_) - 1 / rec)
                    if diff_ < diff:
                        k = k_
                        diff = diff_
                        n = n_
                value = k / pow(2, n)
            value *= kernel[0] * kernel[1]
            return value

        def _is_global_pooling(node):
            return node.outputs_tensor[0].shape[1:3] == [1, 1]

        for match in matches:
            nodes_dict = {}
            for G1_node, G2_node in match.items():
                nodes_dict[G2_node.op_name] = G1_node

            # type checking
            node1_avgpool = nodes_dict["node1"]
            assert isinstance(node1_avgpool, XModelNodeAvgPool)

            # * perform optimization

            # compute tensor based on kernel of avgpool
            [kw, kh] = node1_avgpool.kernel_size

            if kw == kh:
                value = _get_avgpool_scale(kw, kh)
            else:
                if _is_global_pooling(node1_avgpool):
                    new_kh, new_kw = _get_dpu_kernel_size(kh, kw)
                    if new_kh <= 8:
                        value = _get_avgpool_scale(new_kw, new_kh)
                    else:
                        value = _get_avgpool_scale(kw, kh)
                else:
                    value = _get_avgpool_scale(kw, kh)

            # create an XModelNodeConst object
            node2_const = XModelNodeConst(node1_avgpool.op_name + "_const")
            tensor = np.array([value], dtype=np.float32)
            node2_const.tensor = XTensor(
                tensor, format=node1_avgpool.outputs_tensor[0].data_format
            )

            # create an XModelNodeElemMul object
            node3_elemmul = XModelNodeElemMul(node1_avgpool.op_name + "_elemmul")

            # update bottom and top
            node3_elemmul.bottom = [node1_avgpool.op_name, node2_const.op_name]
            node3_elemmul.top = [x for x in node1_avgpool.top]
            node2_const.top = [node3_elemmul.op_name]
            node1_avgpool.top = [node3_elemmul.op_name]

            for cname in node3_elemmul.top:
                cnode = xmodel.get_xnode_by_name(cname)
                assert cnode is not None
                idx = cnode.bottom.index(node1_avgpool.op_name)
                cnode.bottom[idx] = node3_elemmul.op_name

            # update xmodel with new nodes
            xmodel.add_xnodes([node2_const, node3_elemmul])
