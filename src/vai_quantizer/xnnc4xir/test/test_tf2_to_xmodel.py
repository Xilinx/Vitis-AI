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

import sys
from pathlib import Path

curr_path = Path(__file__).resolve()
PRJ_DIR = curr_path.parents[1]
sys.path.append(str(PRJ_DIR.resolve()))

import unittest

from xnnc.xconverter import XConverter


class HDF52XModelTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_inception_v3_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/inception_v3/quantized.h5"
        )

        dump_to = Path("tf2_inception_v3.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_resnet50_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/resnet50/quantized.h5"
        )

        dump_to = Path("tf2_resnet50.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_mobilenet_1_0_224_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/mobilenet_1_0_224/quantized.h5"
        )

        dump_to = Path("tf2_mobilenet_1_0_224.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_medical_seg_cell_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/medical_seg_cell/quantized.h5"
        )

        dump_to = Path("tf2_medical_seg_cell.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_semantic_seg_city_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/semantic_seg_cities/quantized.h5"
        )

        dump_to = Path("tf2_semantic_seg_city.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_ml2796_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/ml-2796-canon/quantized.h5"
        )

        dump_to = Path("tf2_ml2796.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_ml2875_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/ml-2875/quantized_functional_model.h5"
        )

        dump_to = Path("tf2_ml2875.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_ml3005_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/ML-3005/quant_ft_finetuned.h5"
        )

        dump_to = Path("tf2_ml3005.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_vai877_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/VAI-877/quant_ft.h5"
        )

        dump_to = Path("tf2_vai877.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_fluke_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/fluke/quantized_model.h5"
        )

        dump_to = Path("tf2_fluke.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_unet_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/unet/quantized.h5"
        )

        dump_to = Path("tf2_unet.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_vae_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/tf2_var_autoenc/q_model.h5"
        )

        dump_to = Path("tf2_vae.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_efficientnet_b0_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/Efficientnet-b0/20210517/quantized.h5"
        )

        dump_to = Path("tf2_efficientnet_b0.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_mobilenet_v3_small_1_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/Mobilenet_v3_small_1.0/quantized.h5"
        )

        dump_to = Path("tf2_mobilenet_v3_small_1.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 224, 224, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_vai_1259_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/vai-1259/fpga_classifier_models/quantized_model.h5"
        )

        dump_to = Path("tf2_vai_1259.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 224, 224, 20]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_Mnist_dnn_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/Mnist_dnn/quantized.h5"
        )

        dump_to = Path("tf2_Mnist_dnn.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 28, 28, 1]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_yolov4_tiny_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/yolov4_tiny/quantized.h5"
        )

        dump_to = Path("tf2_yolov4_tiny.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_vai_1451_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/vai_1451/quantized_stereo_encoder.h5"
        )

        dump_to = Path("tf2_vai_1451.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_vatis_14_ea_class01_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/vatis_14_ea/class01_finetune.h5"
        )

        dump_to = Path("tf2_vatis_14_ea_class01.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_vatis_14_ea_class22_r2_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/vatis_14_ea/class22_r2_finetune.h5"
        )

        dump_to = Path("tf2_vatis_14_ea_class22_r2.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf2_vai_1562_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/vai_1562/q_model.h5"
        )

        dump_to = Path("tf2_vai_1562.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow2",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()


if __name__ == "__main__":
    unittest.main()
