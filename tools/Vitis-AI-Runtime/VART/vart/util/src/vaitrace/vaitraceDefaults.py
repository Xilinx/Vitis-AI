
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
        "vitis::ai::Cifar10ClassificationImp::post_process@libvitis_ai_library-cifar10classification.so ",
        "vitis::ai::Cifar10ClassificationImp::run@libvitis_ai_library-cifar10classification.so ",
        "vitis::ai::ClassificationImp::run@libvitis_ai_library-classification.so",
        "vitis::ai::ConfigurableDpuTaskImp::run@libvitis_ai_library-dpu_task.so",
        "vitis::ai::ConfigurableDpuTaskImp::setInputImageBGR@libvitis_ai_library-dpu_task.so",
        "vitis::ai::ConfigurableDpuTaskImp::setInputImageRGB@libvitis_ai_library-dpu_task.so",
        "vitis::ai::Covid19Segmentation8UC1::run@libvitis_ai_library-covid19segmentation.so ",
        "vitis::ai::Covid19Segmentation8UC3::run@libvitis_ai_library-covid19segmentation.so ",
        "vitis::ai::Covid19SegmentationImp::run_8UC1@libvitis_ai_library-covid19segmentation.so ",
        "vitis::ai::Covid19SegmentationImp::run_8UC3@libvitis_ai_library-covid19segmentation.so ",
        "vitis::ai::DetectImp::run@libvitis_ai_library-facedetect.so",
        "vitis::ai::DpuTaskImp::run@libvitis_ai_library-dpu_task.so",
        "vitis::ai::FaceDetectRecogImp::run@libvitis_ai_library-facedetectrecog.so ",
        "vitis::ai::FaceFeatureImp::run@libvitis_ai_library-facefeature.so ",
        "vitis::ai::FaceLandmarkImp::run@libvitis_ai_library-facelandmark.so",
        "vitis::ai::FaceQuality5ptImp::run@libvitis_ai_library-facequality5pt.so ",
        "vitis::ai::FaceRecogImp::_runNormalFixed@libvitis_ai_library-facerecog.so ",
        "vitis::ai::FaceRecogImp::run@libvitis_ai_library-facerecog.so ",
        "vitis::ai::FaceRecogImp::run_fixed@libvitis_ai_library-facerecog.so ",
        "vitis::ai::HourglassImp::run@libvitis_ai_library-hourglass.so ",
        "vitis::ai::MedicalDetectionImp::run@libvitis_ai_library-medicaldetection.so ",
        "vitis::ai::MedicalSegmentationImp::run@libvitis_ai_library-medicalsegmentation.so ",
        "vitis::ai::MnistClassificationImp::post_process@libvitis_ai_library-mnistclassification.so ",
        "vitis::ai::MnistClassificationImp::pre_process@libvitis_ai_library-mnistclassification.so ",
        "vitis::ai::MnistClassificationImp::run@libvitis_ai_library-mnistclassification.so ",
        "vitis::ai::MultiTaskImp::run_8UC1@libvitis_ai_library-multitask.so",
        "vitis::ai::MultiTaskImp::run_8UC3@libvitis_ai_library-multitask.so",
        "vitis::ai::MultiTaskImp::run_it@libvitis_ai_library-multitask.so",
        "vitis::ai::OpenPoseImp::run@libvitis_ai_library-openpose.so",
        "vitis::ai::PlateDetectImp::run@libvitis_ai_library-platedetect.so ",
        "vitis::ai::PlateNumImp::run@libvitis_ai_library-platenum.so ",
        "vitis::ai::PlateRecogImp::run@libvitis_ai_library-platerecog.so ",
        "vitis::ai::PointPillarsImp::do_pointpillar_display@libvitis_ai_library-pointpillars.so ",
        "vitis::ai::PointPillarsImp::run@libvitis_ai_library-pointpillars.so ",
        "vitis::ai::PointPillarsPost::post_process@libvitis_ai_library-pointpillars.so ",
        "vitis::ai::PoseDetectImp::run@libvitis_ai_library-posedetect.so",
        "vitis::ai::RefineDetImp::run@libvitis_ai_library-refinedet.so",
        "vitis::ai::ReidImp::run@libvitis_ai_library-reid.so",
        "vitis::ai::RetinaFaceImp::run@libvitis_ai_library-retinaface.so ",
        "vitis::ai::RoadLineImp::run@libvitis_ai_library-lanedetect.so",
        "vitis::ai::SSDImp::run@libvitis_ai_library-ssd.so",
        "vitis::ai::Segmentation8UC1::run@libvitis_ai_library-segmentation.so",
        "vitis::ai::Segmentation8UC3::run@libvitis_ai_library-segmentation.so",
        "vitis::ai::SegmentationImp::run_8UC1@libvitis_ai_library-segmentation.so",
        "vitis::ai::SegmentationImp::run_8UC3@libvitis_ai_library-segmentation.so",
        "vitis::ai::TFSSDImp::run@libvitis_ai_library-tfssd.so",
        "vitis::ai::YOLOv2Imp::run@libvitis_ai_library-yolov2.so",
        "vitis::ai::YOLOv3Imp::run@libvitis_ai_library-yolov3.so",
        "vitis::ai::inception_preprocess@libvitis_ai_library-classification.so",
        "vitis::ai::topk@libvitis_ai_library-pointpillars.so ",
        "vitis::ai::vgg_preprocess@libvitis_ai_library-classification.so"
    ],
    "trace_xnnpp_post_process": [
        "vitis::ai::MedicalSegmentationPost::medicalsegmentation_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::post_process_seg@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::post_process_seg_visualization@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_det@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_seg@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_seg_visualization@libxnnpp-xnnpp.so",
        "vitis::ai::RefineDetPost::refine_det_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::RetinaFacePost::retinaface_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::RetinaFacePost::retinaface_post_process_internal@libxnnpp-xnnpp.so",
        "vitis::ai::RoadLinePost::road_line_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::SSDPost::ssd_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::SSDPost::ssd_post_process_internal_uniform@libxnnpp-xnnpp.so",
        "vitis::ai::TFSSDPost::ssd_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::classification_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::face_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::face_feature_post_process_fixed@libxnnpp-xnnpp.so",
        "vitis::ai::face_feature_post_process_float@libxnnpp-xnnpp.so",
        "vitis::ai::face_landmark_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::face_quality5pt_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::hourglass_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::medicaldetection::MedicalDetectionPost::medicaldetection_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::open_pose_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::plate_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::plate_num_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::pose_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::reid_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::segmentation_post_process_8UC1@libxnnpp-xnnpp.so",
        "vitis::ai::segmentation_post_process_8UC3@libxnnpp-xnnpp.so",
        "vitis::ai::softmax@libvitis_ai_library-math.so",
        "vitis::ai::tfrefinedet::TFRefineDetPost::tfrefinedet_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::yolov2_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::yolov3_post_process@libxnnpp-xnnpp.so"
    ],
    "trace_vart": [
        "vart::TensorBuffer::copy_from_host@libvart-runner.so",
        "vart::TensorBuffer::copy_to_host@libvart-runner.so",
        "xir::XrtCu::run@libvart-dpu-controller.so"
    ],
    "trace_opencv": [
        "cv::imread",
        "cv::imshow",
        "cv::resize"
    ]
}

trace_va_timeout = 60
trace_xat_timeout = 5

traceCfgDefaule = {
    "options": {
        "runmodel": "normal"
    },
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
