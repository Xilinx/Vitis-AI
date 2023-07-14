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
from xnnc.ir.enums import TargetType

# import logging
import unittest


class SerdeTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf2_resnet50(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_resnet50.pb"
        )

        fname = Path("serde_tf2_resnet50.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf2_semantic_seg_cities(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_semantic_seg_cities.pb"
        )

        fname = Path("serde_tf2_semantic_seg_cities.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf2_inception_v3(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_inception_v3.pb"
        )

        fname = Path("serde_tf2_inception_v3.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf2_mobilenet_v1(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_mobilenet_v1.pb"
        )

        fname = Path("serde_tf2_mobilenet_v1.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf2_medical_seg_cell(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf2_xmodel/tf2_medical_seg_cell.pb"
        )

        fname = Path("serde_tf2_medical_seg_cell.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_resnet50_v1(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet50_v1.pb"
        )

        fname = Path("serde_tf1_resnet50_v1.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_resnet101_v1(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet101_v1.pb"
        )

        fname = Path("serde_tf1_resnet101_v1.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_resnet152_v1(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet152_v1.pb"
        )

        fname = Path("serde_tf1_resnet152_v1.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_inception_v1(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_v1.pb"
        )

        fname = Path("serde_tf1_inception_v1.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_inception_v3(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_v3.pb"
        )

        fname = Path("serde_tf1_inception_v3.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_inception_v4(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_v4.pb"
        )

        fname = Path("serde_tf1_inception_v4.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_inception_resnet_v2(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_resnet_v2.pb"
        )

        fname = Path("serde_tf1_inception_resnet_v2.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_mobilenet_v1_025_128(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_mobilenet_v1_0.25_128.pb"
        )

        fname = Path("serde_tf1_mobilenet_v1_025_128.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_mobilenet_v1_05_160(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_mobilenet_v1_0.5_160.pb"
        )

        fname = Path("serde_tf1_mobilenet_v1_05_160.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_mobilenet_v1_10_224(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_mobilenet_v1_1.0_224.pb"
        )

        fname = Path("serde_tf1_mobilenet_v1_10_224.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_mobilenet_v2_10_224(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_mobilenet_v2_1.0_224.pb"
        )

        fname = Path("serde_tf1_mobilenet_v2_10_224.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_mobilenet_v2_14_224(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssd_mobilenet_v1_coco.pb"
        )

        fname = Path("serde_tf1_mobilenet_v2_14_224.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_vgg_16(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_vgg_16.pb"
        )

        fname = Path("serde_tf1_vgg_16.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_vgg_19(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_vgg_19.pb"
        )

        fname = Path("serde_tf1_vgg_19.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_ssd_mobilenet_v1_coco(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssd_mobilenet_v1_coco.pb"
        )

        fname = Path("serde_tf1_ssd_mobilenet_v1_coco.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_ssd_mobilenet_v2_coco(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssd_mobilenet_v2_coco.pb"
        )

        fname = Path("serde_tf1_ssd_mobilenet_v2_coco.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_yolov3_voc(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_yolov3_voc_model.pb"
        )

        fname = Path("serde_tf1_yolov3_voc.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_resnet34_ssd(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet34_ssd_code_model.pb"
        )

        fname = Path("serde_tf1_resnet34_ssd.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_inception_v2(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_inception_v2.pb"
        )

        fname = Path("serde_tf1_inception_v2.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_resnet_v2_50(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet_v2_50.pb"
        )

        fname = Path("serde_tf1_resnet_v2_50.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_resnet_v2_152(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet_v2_152.pb"
        )

        fname = Path("serde_tf1_resnet_v2_152.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_ssdlite_mobilenet_v2_coco(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssdlite_mobilenet_v2_coco.pb"
        )

        fname = Path("serde_tf1_ssdlite_mobilenet_v2_coco.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_ssd_inception_v2_coco(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssd_inception_v2_coco.pb"
        )

        fname = Path("serde_tf1_ssd_inception_v2_coco.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_MobileNetV2_city(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_MobileNetV2_city.pb"
        )

        fname = Path("serde_tf1_MobileNetV2_city.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_efficientNet_edgetpu_S(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_efficientNet_edgetpu_S.pb"
        )

        fname = Path("serde_tf1_efficientNet_edgetpu_S.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_efficientNet_edgetpu_M(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_efficientNet_edgetpu_M.pb"
        )

        fname = Path("serde_tf1_efficientNet_edgetpu_M.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_efficientNet_edgetpu_L(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_efficientNet_edgetpu_L.pb"
        )

        fname = Path("serde_tf1_efficientNet_edgetpu_L.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_resnet_v15(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_resnet_v1.5.pb"
        )

        fname = Path("serde_tf1_resnet_v15.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_MLPerf_pruned_resnet_v15_remain074_finetune(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_MLPerf_pruned_resnet_v1.5_remain0.74_finetune.pb"
        )

        fname = Path("serde_tf1_MLPerf_pruned_resnet_v1.5_remain0.74_finetune.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_mobilenet_edge_10(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_mobilenet_edge_1.0.pb"
        )

        fname = Path("serde_tf1_mobilenet_edge_1.0.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_mobilenet_edge_075(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_mobilenet_edge_0.75.pb"
        )

        fname = Path("serde_tf1_mobilenet_edge_0.75.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_RefineDet_Medical(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_RefineDet_Medical.pb"
        )

        fname = Path("serde_tf1_RefineDet_Medical.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_RefineDet(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_Refinedet.pb"
        )

        fname = Path("serde_tf1_RefineDet.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serde_tf1_ssd_resnet50_v1_fpn_coco(self):
        # path to tensorflow model
        nndct_model = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/nndct_xmodel/tf1_xmodel/tf1_ssd_resnet50_v1_fpn_coco.pb"
        )

        fname = Path("serde_tf1_ssd_resnet50_v1_fpn_coco.pb")

        # convert to xmodel
        XConverter.run(
            [nndct_model],
            model_type="nndct",
            layout="NHWC",
            dump_to=fname,
            target=TargetType.OPENIR,
        )

        self.assertTrue(fname.exists())

        # clean up
        print("clean up generated files.")
        fname.unlink()


if __name__ == "__main__":
    unittest.main()