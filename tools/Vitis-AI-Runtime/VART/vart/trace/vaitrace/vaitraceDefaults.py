
# Copyright 2019 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


builtInFunctions = {
    "trace_vitis-ai-library": [
        "vitis::ai::BCCImp::bcc_post_process@libvitis_ai_library-bcc.so",
        "vitis::ai::BCCImp::preprocess@libvitis_ai_library-bcc.so",
        "vitis::ai::BCCImp::run@libvitis_ai_library-bcc.so",
        "vitis::ai::CarPlateRecogImp::run@libvitis_ai_library-carplaterecog.so",
        "vitis::ai::CenterPointImp::run@libvitis_ai_library-centerpoint.so",
        "vitis::ai::ClassificationImp::run@libvitis_ai_library-classification.so",
        "vitis::ai::Covid19Segmentation8UC1::run@libvitis_ai_library-covid19segmentation.so",
        "vitis::ai::Covid19Segmentation8UC3::run@libvitis_ai_library-covid19segmentation.so",
        "vitis::ai::DetectImp::run@libvitis_ai_library-facedetect.so",
        "vitis::ai::DpuTaskImp::run@libvitis_ai_library-dpu_task.so",
        "vitis::ai::DpuTaskImp::setImageBGR@libvitis_ai_library-dpu_task.so",
        "vitis::ai::DpuTaskImp::setImageRGB@libvitis_ai_library-dpu_task.so",
        "vitis::ai::DpuTaskImp::setMeanScaleBGR@libvitis_ai_library-dpu_task.so",
        "vitis::ai::FaceDetectRecogImp::run@libvitis_ai_library-facedetectrecog.so",
        "vitis::ai::FaceFeatureImp::run@libvitis_ai_library-facefeature.so",
        "vitis::ai::FaceLandmarkImp::run@libvitis_ai_library-facelandmark.so",
        "vitis::ai::FaceQuality5ptImp::run@libvitis_ai_library-facequality5pt.so",
        "vitis::ai::FaceRecogImp::run@libvitis_ai_library-facerecog.so",
        "vitis::ai::HourglassImp::run@libvitis_ai_library-hourglass.so",
        "vitis::ai::MedicalDetectionImp::run@libvitis_ai_library-medicaldetection.so",
        "vitis::ai::MedicalSegcellImp::post_process@libvitis_ai_library-medicalsegcell.so",
        "vitis::ai::MedicalSegcellImp::run@libvitis_ai_library-medicalsegcell.so",
        "vitis::ai::MedicalSegmentationImp::run@libvitis_ai_library-medicalsegmentation.so",
        "vitis::ai::MnistClassificationImp::post_process@libvitis_ai_library-mnistclassification.so",
        "vitis::ai::MnistClassificationImp::pre_process@libvitis_ai_library-mnistclassification.so",
        "vitis::ai::MnistClassificationImp::run@libvitis_ai_library-mnistclassification.so",
        "vitis::ai::MultiTaskImp::run_8UC1@libvitis_ai_library-multitask.so",
        "vitis::ai::MultiTaskImp::run_8UC3@libvitis_ai_library-multitask.so",
        "vitis::ai::MultiTaskImp::run_it@libvitis_ai_library-multitask.so",
        "vitis::ai::MultiTaskv3Imp::run_8UC1@libvitis_ai_library-multitaskv3.so",
        "vitis::ai::MultiTaskv3Imp::run_8UC3@libvitis_ai_library-multitaskv3.so",
        "vitis::ai::MultiTaskv3Imp::run_it@libvitis_ai_library-multitaskv3.so",
        "vitis::ai::OpenPoseImp::run@libvitis_ai_library-openpose.so",
        "vitis::ai::PMGImp::pmg_post_process@libvitis_ai_library-pmg.so",
        "vitis::ai::PMGImp::run@libvitis_ai_library-pmg.so",
        "vitis::ai::PlateDetectImp::run@libvitis_ai_library-platedetect.so",
        "vitis::ai::PlateNumImp::run@libvitis_ai_library-platenum.so",
        "vitis::ai::PlateRecogImp::run@libvitis_ai_library-platerecog.so",
        "vitis::ai::PointPaintingImp::run@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPaintingImp::runPointPillars@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPaintingImp::runSegmentation@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPaintingImp::runSegmentationFusion@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPillarsImp::run@libvitis_ai_library-pointpillars.so",
        "vitis::ai::PointPillarsNuscenesImp::run@libvitis_ai_library-pointpillars_nuscenes.so",
        "vitis::ai::PointPillarsPost::post_process@libvitis_ai_library-pointpillars.so",
        "vitis::ai::PoseDetectImp::run@libvitis_ai_library-posedetect.so",
        "vitis::ai::RGBDsegmentationImp::run@libvitis_ai_library-RGBDsegmentation.so",
        "vitis::ai::RcanImp::run@libvitis_ai_library-rcan.so",
        "vitis::ai::RefineDetImp::run@libvitis_ai_library-refinedet.so",
        "vitis::ai::ReidImp::run@libvitis_ai_library-reid.so",
        "vitis::ai::RetinaFaceImp::run@libvitis_ai_library-retinaface.so",
        "vitis::ai::RoadLineImp::run@libvitis_ai_library-lanedetect.so",
        "vitis::ai::SSDImp::run@libvitis_ai_library-ssd.so",
        "vitis::ai::Segmentation3DImp::run@libvitis_ai_library-3Dsegmentation.so",
        "vitis::ai::SegmentationImp::run_8UC1@libvitis_ai_library-segmentation.so",
        "vitis::ai::SegmentationImp::run_8UC3@libvitis_ai_library-segmentation.so",
        "vitis::ai::TFSSDImp::run@libvitis_ai_library-tfssd.so",
        "vitis::ai::XmodelImageImp::run@libvitis_ai_library-xmodel_image.so",
        "vitis::ai::XmodelPostprocessorSingleBatch::process@libvitis_ai_library-xmodel_image.so",
        "vitis::ai::YOLOv2Imp::run@libvitis_ai_library-yolov2.so",
        "vitis::ai::YOLOv3Imp::run@libvitis_ai_library-yolov3.so",
        "vitis::ai::bev_preprocess@libvitis_ai_library-pointpillars.so",
        "vitis::ai::centerpoint::middle_process@libvitis_ai_library-centerpoint.so",
        "vitis::ai::centerpoint::preprocess3@libvitis_ai_library-centerpoint.so",
        "vitis::ai::efficientnet_preprocess@libvitis_ai_library-classification.so",
        "vitis::ai::inception_preprocess@libvitis_ai_library-classification.so",
        "vitis::ai::rgbdsegmentation::process_image_rgbd@libvitis_ai_library-RGBDsegmentation.so",
        "vitis::ai::vgg_preprocess@libvitis_ai_library-classification.so"
    ],
    "trace_xnnpp_post_process": [
        "vitis::ai::MedicalSegmentationPost::medicalsegmentation_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::post_process_seg@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_det@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_seg@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskv3PostProcessImp::post_process@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskv3PostProcessImp::process_depth@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskv3PostProcessImp::process_det@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskv3PostProcessImp::process_seg@libxnnpp-xnnpp.so",
        "vitis::ai::PointPillarsNuscenesPost::postprocess@libxnnpp-xnnpp.so",
        "vitis::ai::RefineDetPost::refine_det_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::RetinaFacePost::retinaface_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::RoadLinePost::road_line_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::SSDPost::ssd_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::TFSSDPost::ssd_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::convert_color@libxnnpp-xnnpp.so",
        "vitis::ai::dpssd::SSDdetector::detect@libxnnpp-xnnpp.so",
        "vitis::ai::medicaldetection::MedicalDetectionPost::medicaldetection_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::medicaldetection::SSDDetector::detect@libxnnpp-xnnpp.so",
        "vitis::ai::plate_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::plate_num_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::pose_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::post_process@libxnnpp-xnnpp.so",
        "vitis::ai::rcan_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::segmentation_post_process_8UC1@libxnnpp-xnnpp.so",
        "vitis::ai::softmax@libvitis_ai_library-math.so",
        "vitis::ai::tfrefinedet::TFRefineDetPost::tfrefinedet_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::transform_bgr@libvitis_ai_library-math.so",
        "vitis::ai::transform_mean_scale@libvitis_ai_library-math.so",
        "vitis::ai::transform_rgb@libvitis_ai_library-math.so",
        "vitis::ai::yuv2bgr@libvitis_ai_library-math.so"
    ],
    "trace_vart": [
        "convert_data_type@libvart-runner.so",
        "convert_tensors@libvart-runner.so",
        "vart::TensorBuffer::copy_from_host@libvart-runner.so",
        "vart::TensorBuffer::copy_tensor_buffer@libvart-runner.so",
        "vart::TensorBuffer::copy_to_host@libvart-runner.so",
        "xir::XrtCu::run@libvart-dpu-controller.so"
    ],
    "trace_opencv": [
        "cv::imread",
        "cv::imshow",
        "cv::resize"
    ]
}

trace_va_timeout = 30
trace_xat_timeout = 30
trace_max_timeout = 30
trace_fg_timeout = 10
trace_max_fg_timeout = 10

default_runmode = "normal"
#default_runmode = "debug"

traceCfgDefaule = {
    'collector': {},
    'trace': {
        "enable_trace_list": ["vitis-ai-library", "vart", "xnnpp_post_process", "custom"]
    },
    'control': {
        'cmd': None,
        'xat': {
            'compress': True,
            'filename': None
        },
        'config': None,
        'timeout': None,
    }
}
