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
from xir import Graph

# import logging
import unittest


class NNDCTTorch2XModelTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_nndct_resnet18_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/pytorch_xmodel/resnet18/ResNet_int.pb"
        )

        dump_to = Path("resnet_v1_50_tf1.15.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_resnet50_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_resnet50.pb"
        )

        dump_to = Path("tf2_resnet50.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_semantic_seg_cities_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_semantic_seg_cities.pb"
        )

        dump_to = Path("tf2_semantic_seg_cities.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_inception_v3_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_inception_v3.pb"
        )

        dump_to = Path("tf2_inception_v3.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_mobilenet_v1_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_mobilenet_v1.pb"
        )

        dump_to = Path("tf2_mobilenet_v1.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_medical_seg_cell_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_medical_seg_cell.pb"
        )

        dump_to = Path("tf2_medical_seg_cell.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_resnet50_v1_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet50_v1.pb"
        )

        dump_to = Path("tf1_resnet50_v1.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_inception_v1_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_v1.pb"
        )

        dump_to = Path("tf1_inception_v1.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_inception_v3_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_v3.pb"
        )

        dump_to = Path("tf1_inception_v3.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_inception_v4_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_v4.pb"
        )

        dump_to = Path("tf1_inception_v4.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_mobilenet_v1_025_128_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_mobilenet_v1_0.25_128.pb"
        )

        dump_to = Path("tf1_mobilenet_v1_0.25_128.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_mobilenet_v2_14_224_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssd_mobilenet_v1_coco.pb"
        )

        dump_to = Path("tf1_ssd_mobilenet_v1_coco.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_yolov3_voc_model_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_yolov3_voc_model.pb"
        )

        dump_to = Path("tf1_yolov3_voc_model.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_resnet34_ssd_code_model_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet34_ssd_code_model.pb"
        )

        dump_to = Path("tf1_resnet34_ssd_code_model.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_MobileNetV2_city_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_MobileNetV2_city.pb"
        )

        dump_to = Path("tf1_MobileNetV2_city.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_RefineDet_Medical_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_RefineDet_Medical.pb"
        )

        dump_to = Path("tf1_RefineDet_Medical.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_RefineDet_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_Refinedet.pb"
        )

        dump_to = Path("tf1_Refinedet.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_ssd_resnet50_v1_fpn_coco_fix_to_xmodel(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssd_resnet50_v1_fpn_coco.pb"
        )

        dump_to = Path("tf1_ssd_resnet50_v1_fpn_coco.xmodel")

        # convert to xmodel
        XConverter.run(
            [nndct_model], model_type="nndct", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()


if __name__ == "__main__":
    unittest.main()
