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

from pathlib import Path
import sys

curr_path = Path(__file__).resolve()
PRJ_DIR = curr_path.parents[1]
sys.path.append(str(PRJ_DIR.resolve()))

from xnnc.entity.xmodel import XModel
from xnnc.entity.xnode import *
from xnnc.optimizer.manager import OptManager
from xnnc.utils import helper

import unittest


class GraphIsomorphismTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_subgraph_isomorph_vf2(self):
        # define G1 graph
        G1 = XModel(name="G1", model_type="caffe")
        # input
        node1 = XModelNode(op_name="node1", op_type="input")
        G1.update_xnode(node1)
        # conv2d
        node2 = XModelNode(op_name="node2", op_type="conv2d")
        G1.update_xnode(node2)
        # size
        node3 = XModelNode(op_name="node3", op_type="size")
        G1.update_xnode(node3)
        # listconstruct
        node4 = XModelNode(op_name="node4", op_type="listconstruct")
        G1.update_xnode(node4)
        # reshape
        node5 = XModelNode(op_name="node5", op_type="reshape")
        G1.update_xnode(node5)
        # softmax
        node6 = XModelNode(op_name="node6", op_type="softmax")
        G1.update_xnode(node6)
        # size
        node7 = XModelNode(op_name="node7", op_type="size")
        G1.update_xnode(node7)
        # listconstruct
        node8 = XModelNode(op_name="node8", op_type="listconstruct")
        G1.update_xnode(node8)
        # reshape
        node9 = XModelNode(op_name="node9", op_type="reshape")
        G1.update_xnode(node9)

        node1.top = ["node2", "node3"]
        node2.bottom = ["node1"]
        node2.top = ["node5", "node7"]
        node3.bottom = ["node1"]
        node3.top = ["node4"]
        node4.bottom = ["node3"]
        node4.top = ["node5"]
        node5.bottom = ["node2", "node4"]
        node5.top = ["node6"]
        node6.bottom = ["node5", "node9"]
        node7.bottom = ["node2"]
        node7.top = ["node8"]
        node8.bottom = ["node7"]
        node8.top = ["node9"]
        node9.bottom = ["node8"]
        node9.top = ["node6"]

        helper.render_xmodel(G1, filename="G1")

        # define G2 graph
        G2 = XModel(name="G2", model_type="caffe")
        # size
        node_21 = XModelNode(op_name="node21", op_type="size")
        G2.update_xnode(node_21)
        # listconstruct
        node_22 = XModelNode(op_name="node22", op_type="listconstruct")
        G2.update_xnode(node_22)
        # reshape
        node_23 = XModelNode(op_name="node23", op_type="reshape")
        G2.update_xnode(node_23)

        node_21.top = ["node22"]
        node_22.bottom = ["node21"]
        node_22.top = ["node23"]
        node_23.bottom = ["node22"]

        # helper.render_xmodel(G2, filename="G2")

        # find subgraph matching
        DiGM = DiGraphMatcher(G1, G2)
        matches = DiGM.search_matching()

        if matches:
            print(f"Found {len(matches)} match(es)")

        print("Done!!!")

    # @unittest.skipIf(stop_all, "skip over this routine")
    def test_optimization(self):
        # define G1 graph
        G1 = XModel(name="G1", model_type="tensorflow")
        # input
        node1 = XModelNode(op_name="node1_act", op_type="input")
        G1.update_xnode(node1)
        # conv2d
        node2 = XModelNodeConv2d(op_name="node2_act", ksize=[3, 3])
        G1.update_xnode(node2)
        # const
        node3 = XModelNodeConst("node3_act")
        node3.tensor = np.arange(24).reshape(2, 3, 4)
        G1.update_xnode(node3)
        # fixneuron
        node4 = XModelNodeFixNeuron(op_name="node4_act")
        node4.quant_in["bit_width"] = 8
        node4.quant_in["quantize_pos"] = -7
        G1.update_xnode(node4)
        # reshape
        node5 = XModelNodeElemAdd(op_name="node5_act")
        G1.update_xnode(node5)
        # relu
        node6 = XModelNode(op_name="node6_act", op_type="relu")
        G1.update_xnode(node6)
        # size
        node7 = XModelNode(op_name="node7_act", op_type="size")
        G1.update_xnode(node7)
        # listconstruct
        node8 = XModelNode(op_name="node8_act", op_type="listconstruct")
        G1.update_xnode(node8)
        # reshape
        node9 = XModelNode(op_name="node9_act", op_type="reshape")
        G1.update_xnode(node9)
        # softmax
        node10 = XModelNode(op_name="node10_act", op_type="softmax")
        G1.update_xnode(node10)

        node1.top = [node2.op_name]
        node2.bottom = [node1.op_name]
        node2.top = [node5.op_name]
        node3.top = [node4.op_name]
        node4.bottom = [node3.op_name]
        node4.top = [node5.op_name]
        node5.bottom = [node2.op_name, node4.op_name]
        node5.top = [node6.op_name]
        node6.bottom = [node5.op_name]
        node6.top = [node9.op_name]
        node7.top = [node8.op_name]
        node8.bottom = [node7.op_name]
        node8.top = [node9.op_name]
        node9.bottom = [node6.op_name, node8.op_name]
        node9.top = [node10.op_name]
        node10.bottom = [node9.op_name]

        # G1.render()

        OptManager.dispatch(G1)

        # # find subgraph matching
        # DiGM = DiGraphMatcher(G1, G2)
        # matches = DiGM.search_matching()

        # if matches:
        #     print(f"Found {len(matches)} match(es)")

        print("Done!!!")


if __name__ == "__main__":
    unittest.main()
