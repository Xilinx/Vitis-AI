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

# import logging
import unittest


class TF2XModelTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_resnetv1_50_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/resnet_v1_50_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("resnet_v1_50_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_resnetv1_101_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/resnet_v1_101_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("resnet_v1_101_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_resnet_v1_152_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/resnet_v1_152_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("resnet_v1_152_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_inception_v1_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/inception_v1_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("inception_v1_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_inception_v1_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/inception_v1_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("inception_v1_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_inception_v3_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/inception_v3_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("inception_v3_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_inception_v4_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/inception_v4_2016_09_09_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("inception_v4_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mobilenet_v1_025_128_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/mobilenet_v1_0.25_128/quantize_eval_model.pb"
        )

        dump_to = Path("mobilenet_v1_0.25_128.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mobilenet_v1_05_160_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/mobilenet_v1_0.5_160/quantize_eval_model.pb"
        )

        dump_to = Path("mobilenet_v1_0.5_160.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mobilenet_v1_10_224_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/mobilenet_v1_1.0_224/quantize_eval_model.pb"
        )

        dump_to = Path("mobilenet_v1_1.0_224.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mobilenet_v2_10_224_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/mobilenet_v2_1.0_224/quantize_eval_model.pb"
        )

        dump_to = Path("mobilenet_v2_1.0_224.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mobilenet_v2_14_224_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/mobilenet_v2_1.4_224/quantize_eval_model.pb"
        )

        dump_to = Path("mobilenet_v2_1.4_224.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_inception_resnet_v2_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/inception_resnet_v2_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("inception_resnet_v2_tf1.15.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_vgg_16_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/vgg_16/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_vgg_16.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_vgg_19_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/vgg_19/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_vgg_19.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_ssd_mobilenet_v1_coco_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ssd_mobilenet_v1_coco/quantize_eval_model.pb"
        )

        dump_to = Path("ssd_mobilenet_v1_coco.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_ssd_mobilenet_v2_coco_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ssd_mobilenet_v2_coco/quantize_eval_model.pb"
        )

        dump_to = Path("ssd_mobilenet_v2_coco.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel], model_type="tensorflow", layout="NHWC", dump_to=dump_to
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_ssd_resnet50_v1_fpn_coco_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ssd_resnet50_v1_fpn_coco/quantize_eval_model.pb"
        )

        dump_to = Path("ssd_resnet50_v1_fpn_coco.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 640, 640, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_yolov3_voc_model_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf_yolov3_voc_model/quantize_eval_model.pb"
        )

        dump_to = Path("tf_yolov3_voc_model.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 416, 416, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_resnet34_ssd_code_model_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/resnet34_ssd_code_model/quantize_eval_model.pb"
        )

        dump_to = Path("resnet34_ssd_code_model.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_resnet_v2_50_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_resnet_v2_50/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_resnet_v2_50.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_ssdlite_mobilenet_v2_coco_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_ssdlite_mobilenet_v2_coco/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_ssdlite_mobilenet_v2_coco.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 300, 300, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_ssd_inception_v2_coco_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_ssd_inception_v2_coco/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_ssd_inception_v2_coco.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 300, 300, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_MobileNetV2_city_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_MobileNetV2_city/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_MobileNetV2_city.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_efficientNet_edgetpu_S_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_efficientNet_edgetpu_S/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_efficientNet_edgetpu_S.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_efficientNet_edgetpu_M_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_efficientNet_edgetpu_M/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_efficientNet_edgetpu_M.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_efficientNet_edgetpu_L_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_efficientNet_edgetpu_L/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_efficientNet_edgetpu_L.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mlperf_resnet_v15_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/mlperf_resnet_v1.5/quantize_eval_model.pb"
        )

        dump_to = Path("mlperf_resnet_v1.5.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_MLPerf_pruned_resnet_v15_remain074_finetune_fix_to_xmodel(
        self,
    ):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_MLPerf_pruned_resnet_v1.5_remain0.74_finetune/quantize_eval_model.pb"
        )

        dump_to = Path(
            "tf1_MLPerf_pruned_resnet_v1.5_remain0.74_finetune.xmodel"
        ).resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mobilenet_edge_10_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_mobilenet_edge_1.0/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_mobilenet_edge_1.0.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_ml_at_edge_yolov3_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ml_at_edge_yolov3/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_ml_at_edge_yolov3.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 416, 416, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_mobilenet_edge_075_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf1_mobilenet_edge_0.75/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_mobilenet_edge_0.75.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_refinedet_medical_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/refinedet_medical_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_refinedet_medical.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_serialize_tf1_refinedet_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/refinedet_tf1.15/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_refinedet.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_ml2781_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ml-2781/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_ml2781.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_ml2767_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ml-2767/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_ml2767.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_ml2963_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ml-2963/everseen_quantize_eval_model.pb"
        )

        dump_to = Path("tf1_ml2963.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_variation_autoencoder_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/autoencoder/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_variation_autoencoder.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_fcn8_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/fcn8/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_fcn8.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_olympus48_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/olympus48/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_olympus48.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_olympus96_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/olympus96/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_olympus96.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_nasnet_a_large_fix_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/nasnet_a_large/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_nasnet_a_large.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_subaru_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/subaru/quantize_eval_model_2.pb"
        )

        dump_to = Path("tf1_subaru.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_fcn8hdtv_semseg_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/fcn8hdtv-semseg/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_fcn8hdtv_semseg.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_unet_500_500_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/unet_500_500/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_unet_500_500.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 500, 500, 4]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_yolov3_voc_416_416_6563G_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tf_yolov3_voc_416_416_65.63G_1.3/quantize_eval_model.pb"
        )

        dump_to = Path("tf1_yolov3_voc_416_416_6563G.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 416, 416, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_tf1_tensorrt_to_xmodel(self):
        # path to tensorflow model
        tfmodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/tensorrt/modified_quantize_eval_model.pb"
        )

        dump_to = Path("tf1_tensorrt.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [tfmodel],
            model_type="tensorflow",
            layout="NHWC",
            dump_to=dump_to,
            in_shapes=[[1, 416, 416, 3]],
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()


if __name__ == "__main__":
    unittest.main()
