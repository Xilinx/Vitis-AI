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

from xnnc.integration.xir_generator import XIRGenerator
from xnnc.proto.tf_pb2 import graph_pb2
from xnnc.utils.helper import Layout

import unittest


class IFITestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf_integration_interface(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ssd_resnet50_v1_fpn_coco_tf1.15/quantize_eval_model.pb"
        )

        # load model file
        graph_def = graph_pb2.GraphDef()
        with open(tfmodel, "rb") as pf:
            graph_def.ParseFromString(pf.read())

        # convert tensorflow frozen model into xir model, and serialize and dump the result
        in_shapes = [[1, 640, 640, 3]]
        fname = Path("quantize_eval_model.xmodel")
        success = XIRGenerator.from_tensorflow(graph_def, fname, Layout.NHWC, in_shapes)
        self.assertTrue(
            success, f"Failed to convert tensorflow frozen model into xir graph."
        )


if __name__ == "__main__":
    unittest.main()
