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

import unittest


class Caffe2XModelTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_resnet50_baseline9213_ck_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/resnet50_baseline9213_ck/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/resnet50_baseline9213_ck/deploy.prototxt"
        )

        dump_to = Path("caffe_resnet50_baseline9213_ck.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            batchsize=10,
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_inception_v1_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v1/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v1/deploy.prototxt"
        )

        dump_to = Path("caffe_inception_v1.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_inception_v2_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v2/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v2/deploy.prototxt"
        )

        dump_to = Path("caffe_inception_v2.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_inception_v3_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v3/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v3/deploy.prototxt"
        )

        dump_to = Path("caffe_inception_v3.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_inception_v4_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v4/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/inception_v4/deploy.prototxt"
        )

        dump_to = Path("caffe_inception_v4.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_mobilenet_v2_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/mobilenet_v2/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/mobilenet_v2/deploy.prototxt"
        )

        dump_to = Path("caffe_mobilenet_v2.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_squeezenet_v11_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/squeezenet_v1.1/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/squeezenet_v1.1/deploy.prototxt"
        )

        dump_to = Path("caffe_squeezenet_v1.1.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_resnet_18_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/resnet_18/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/resnet_18/deploy.prototxt"
        )

        dump_to = Path("caffe_resnet_18.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ssd_pedestrain_detection_purn_097_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ssd_pedestrain_detection_purn_0.97/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ssd_pedestrain_detection_purn_0.97/deploy.prototxt"
        )

        dump_to = Path("caffe_ssd_pedestrain_detection_purn_0.97.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_refinedet_pruned_08_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_purn_0.8/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_purn_0.8/deploy.prototxt"
        )

        dump_to = Path("caffe_refinedet_pruned_0.8.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_refinedet_pruned_092_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_purn_0.92/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_purn_0.92/deploy.prototxt"
        )

        dump_to = Path("caffe_refinedet_pruned_0.92.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_refinedet_pruned_096_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_purn_0.96/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_purn_0.96/deploy.prototxt"
        )

        dump_to = Path("caffe_refinedet_pruned_0.96.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_SSD_adas_pruned_095_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/SSD_adas_pruned_0.95/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/SSD_adas_pruned_0.95/deploy.prototxt"
        )

        dump_to = Path("caffe_SSD_adas_pruned_0.95.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ssd_traffic_116G_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ssd_traffic_11.6G/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ssd_traffic_11.6G/deploy.prototxt"
        )

        dump_to = Path("caffe_ssd_traffic_11.6G.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_VPGnet_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VPGnet/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VPGnet/deploy.prototxt"
        )

        dump_to = Path("caffe_VPGnet.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ssd_mobilnet_v2_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ssd_mobilenet_v2/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ssd_mobilenet_v2/deploy.prototxt"
        )

        dump_to = Path("caffe_ssd_mobilnet_v2.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_seg_FPN_89G_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/seg_FPN_8.9G/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/seg_FPN_8.9G/deploy.prototxt"
        )

        dump_to = Path("caffe_seg_FPN_8.9G.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_openpose_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/openpose/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/openpose/deploy.prototxt"
        )

        dump_to = Path("caffe_openpose.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_poseestimation_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/poseestimation/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/poseestimation/deploy.prototxt"
        )

        dump_to = Path("caffe_poseestimation.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_densebox_320_320_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/densebox_320_320/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/densebox_320_320/deploy.prototxt"
        )

        dump_to = Path("caffe_densebox_320_320.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_densebox_640_360_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/densebox_640_360/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/densebox_640_360/deploy.prototxt"
        )

        dump_to = Path("caffe_densebox_640_360.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_landmark_sun_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/landmark_sun/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/landmark_sun/deploy.prototxt"
        )

        dump_to = Path("caffe_landmark_sun.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_reid_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/reid/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/reid/deploy.prototxt"
        )

        dump_to = Path("caffe_reid.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_multi_task_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/multi_task/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/multi_task/deploy.prototxt"
        )

        dump_to = Path("caffe_multi_task.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_yolov3_bdd_model_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov3_bdd_model/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov3_bdd_model/deploy.prototxt"
        )

        dump_to = Path("caffe_yolov3_bdd_model.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_yolov3_adas_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov3_adas/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov3_adas/deploy.prototxt"
        )

        dump_to = Path("caffe_yolov3_adas.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_yolov3_voc_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov3_voc/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov3_voc/deploy.prototxt"
        )

        dump_to = Path("caffe_yolov3_voc.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_YOLOv2_voc_baseline_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_baseline/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_baseline/deploy.prototxt"
        )

        dump_to = Path("caffe_YOLOv2_voc_baseline.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_YOLOv2_voc_prun_66_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_prun_66%/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_prun_66%/deploy.prototxt"
        )

        dump_to = Path("caffe_YOLOv2_voc_prun_66%.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_YOLOv2_voc_prun_71_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_prun_71%/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_prun_71%/deploy.prototxt"
        )

        dump_to = Path("caffe_YOLOv2_voc_prun_71%.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_YOLOv2_voc_prun_77_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_prun_77%/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/YOLOv2_voc_prun_77%/deploy.prototxt"
        )

        dump_to = Path("caffe_YOLOv2_voc_prun_77%.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_plate_detection_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/plate_detection/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/plate_detection/deploy.prototxt"
        )

        dump_to = Path("caffe_plate_detection.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_plate_recognition_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/plate_recognition/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/plate_recognition/deploy.prototxt"
        )

        dump_to = Path("caffe_plate_recognition.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_facerec_resnet20_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/facerec_resnet20/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/facerec_resnet20/deploy.prototxt"
        )

        dump_to = Path("caffe_facerec_resnet20.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_facerec_resnet64_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/facerec_resnet64/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/facerec_resnet64/deploy.prototxt"
        )

        dump_to = Path("caffe_facerec_resnet64.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_FPN_Res18_Medical_segmentation_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/FPN_Res18_Medical_segmentation/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/FPN_Res18_Medical_segmentation/deploy.prototxt"
        )

        dump_to = Path("caffe_FPN_Res18_Medical_segmentation.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_refinedet_baseline_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_baseline/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/refinedet_baseline/deploy.prototxt"
        )

        dump_to = Path("caffe_refinedet_baseline.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_retinaface_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/retinaface/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/retinaface/deploy.prototxt"
        )

        dump_to = Path("caffe_retinaface.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_tiny_yolov3_vmss_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/tiny_yolov3_vmss/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/tiny_yolov3_vmss/deploy.prototxt"
        )

        dump_to = Path("caffe_tiny_yolov3_vmss.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_yolov4_leaky_spp_m_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov4_leaky_spp_m/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/yolov4_leaky_spp_m/deploy.prototxt"
        )

        dump_to = Path("caffe_yolov4_leaky_spp_m.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_FPN_Res18_Endov_segmentation_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/FPN_Res18_Endov_segmentation/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/FPN_Res18_Endov_segmentation/deploy.prototxt"
        )

        dump_to = Path("caffe_FPN_Res18_Endov_segmentation.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_face_quality_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/face_quality/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/face_quality/deploy.prototxt"
        )

        dump_to = Path("caffe_face_quality.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_VAI_Caffe_ML_CATSvsDOGS_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-ML-CATSvsDOGS/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-ML-CATSvsDOGS/deploy.prototxt"
        )

        dump_to = Path("caffe_VAI-Caffe-ML-CATSvsDOGS.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_VAI_Caffe_ML_CATSvsDOGS_pruned_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-ML-CATSvsDOGS-pruned/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-ML-CATSvsDOGS-pruned/deploy.prototxt"
        )

        dump_to = Path("caffe_VAI-Caffe-ML-CATSvsDOGS-pruned.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ML_Caffe_Segmentation_Tutorial_enet_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-enet/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-enet/deploy.prototxt"
        )

        dump_to = Path("caffe_ML-Caffe-Segmentation-Tutorial-enet.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ML_Caffe_Segmentation_Tutorial_espnet_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-espnet/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-espnet/deploy.prototxt"
        )

        dump_to = Path("caffe_ML-Caffe-Segmentation-Tutorial-espnet.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ML_Caffe_Segmentation_Tutorial_FPN_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-FPN/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-FPN/deploy.prototxt"
        )

        dump_to = Path("caffe_ML-Caffe-Segmentation-Tutorial-FPN.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ML_Caffe_Segmentation_Tutorial_unet_full_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-unet-full/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-unet-full/deploy.prototxt"
        )

        dump_to = Path(
            "caffe_ML-Caffe-Segmentation-Tutorial-unet-full.xmodel"
        ).resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_ML_Caffe_Segmentation_Tutorial_unet_lite_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-unet-lite/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/ML-Caffe-Segmentation-Tutorial-unet-lite/deploy.prototxt"
        )

        dump_to = Path(
            "caffe_ML-Caffe-Segmentation-Tutorial-unet-lite.xmodel"
        ).resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_VAI_Caffe_SSD_Tutorial_VGG16_SSD_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-SSD-Tutorial-VGG16-SSD/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-SSD-Tutorial-VGG16-SSD/deploy.prototxt"
        )

        dump_to = Path("caffe_VAI-Caffe-SSD-Tutorial-VGG16-SSD.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_VAI_Caffe_SSD_Tutorial_Mobilenetv2_SSD_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-SSD-Tutorial-Mobilenetv2-SSD/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/VAI-Caffe-SSD-Tutorial-Mobilenetv2-SSD/deploy.prototxt"
        )

        dump_to = Path("caffe_VAI-Caffe-SSD-Tutorial-Mobilenetv2-SSD.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_jira_vai1131_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/vai_1131/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/vai_1131/deploy.prototxt"
        )

        dump_to = Path("caffe_VAI-Caffe-SSD-Tutorial-Mobilenetv2-SSD.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_peng_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/peng/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/peng/deploy.prototxt"
        )

        dump_to = Path("caffe_peng.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_baidu_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/baidu/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/jira/baidu/deploy.prototxt"
        )

        dump_to = Path("baidu.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_caffe_hourglass_fix_to_xmodel(self):
        # path to caffe model
        caffemodel = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/hourglass/deploy.caffemodel"
        )
        prototxt = Path(
            "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/hourglass/deploy.prototxt"
        )

        dump_to = Path("caffe_hourglass_14.xmodel").resolve()

        # convert to xmodel
        XConverter.run(
            [caffemodel, prototxt],
            model_type="caffe",
            layout="NCHW",
            dump_to=dump_to,
        )

        self.assertTrue(dump_to.exists())

        # clean up
        print("clean up generated files.")
        dump_to.unlink()


if __name__ == "__main__":
    unittest.main()
