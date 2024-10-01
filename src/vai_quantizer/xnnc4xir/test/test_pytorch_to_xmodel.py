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

from xnnc.xconverter import XConverter
from xir.graph import Graph

# import logging
import unittest

# import pytorch modules
import torchvision.models as models
import torch


class Torch2XModelTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_torch_fpn_q4_to_xmodel(self):
        model_arch_file = Path(
            "/root/workspace/arch/xmodel_zoo/hot_request/Quantize_FPN_semantic_segmentation/encoding/models/get_model.py"
        )
        in_shapes = [
            [1, 3, 256, 512],
        ]

        # convert to xmodel
        graph = XConverter.run(
            [model_arch_file], model_type="pytorch", layout="NCHW", in_shapes=in_shapes,
        )

        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, Graph)

        fname = "resnet18-5c106cde"
        # serialize graph
        fxmodel = Path(f"./{fname}.xmodel")
        graph.serialize(fxmodel)

        # dump graph
        fimage = Path(f"./{fname}.svg")
        graph.dump(fimage)
        self.assertTrue(fimage.exists())

        # clean up
        print("clean up generated files.")
        fxmodel.unlink()
        fimage.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_torch_adas_2d_detection_q4_to_xmodel(self):
        model_arch_file = Path(
            "/root/workspace/samshin/xmodel_zoo/pytorch_xmodel/adas_4bit_2d_det_interface/scripts/get_model.py"
        )
        in_shapes = [
            [1, 3, 320, 512],
        ]

        # convert to xmodel
        graph = XConverter.run(
            [model_arch_file], model_type="pytorch", layout="NCHW", in_shapes=in_shapes,
        )

        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, Graph)

        fname = "torch_adas_2d_detection_q4"
        # serialize graph
        fxmodel = Path(f"./{fname}.xmodel")
        graph.serialize(fxmodel)

        # dump graph
        fimage = Path(f"./{fname}.svg")
        graph.dump(fimage)
        self.assertTrue(fimage.exists())

        # clean up
        print("clean up generated files.")
        fxmodel.unlink()
        fimage.unlink()


if __name__ == "__main__":
    unittest.main()
