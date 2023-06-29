# Copyright 2022-2023 Advanced Micro Devices Inc.

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
        "vitis::ai::ARFlowImp::run@libvitis_ai_library-arflow.so",
        "vitis::ai::BCCImp::bcc_post_process@libvitis_ai_library-bcc.so",
        "vitis::ai::BCCImp::preprocess@libvitis_ai_library-bcc.so",
        "vitis::ai::BCCImp::run@libvitis_ai_library-bcc.so",
        "vitis::ai::C2D2_liteImp::run@libvitis_ai_library-c2d2_lite.so",
        "vitis::ai::CarPlateRecogImp::run@libvitis_ai_library-carplaterecog.so",
        "vitis::ai::CenterPointImp::run@libvitis_ai_library-centerpoint.so",
        "vitis::ai::ClassificationImp::run@libvitis_ai_library-classification.so",
        "vitis::ai::ClocsImp::postprocess@libvitis_ai_library-clocs.so",
        "vitis::ai::ClocsImp::postprocess_kernel@libvitis_ai_library-clocs.so",
        "vitis::ai::ClocsImp::run@libvitis_ai_library-clocs.so",
        "vitis::ai::ClocsPointPillarsImp::postprocess@libvitis_ai_library-clocs.so",
        "vitis::ai::ClocsPointPillarsImp::run@libvitis_ai_library-clocs.so",
        "vitis::ai::ClocsPointPillarsImp::run_postprocess_t@libvitis_ai_library-clocs.so",
        "vitis::ai::ClocsPointPillarsImp::run_preprocess_t@libvitis_ai_library-clocs.so",
        "vitis::ai::ConfigurableDpuTaskImp::run@libvitis_ai_library-dpu_task.so",
        "vitis::ai::Covid19Segmentation8UC1::run@libvitis_ai_library-covid19segmentation.so",
        "vitis::ai::Covid19Segmentation8UC3::run@libvitis_ai_library-covid19segmentation.so",
        "vitis::ai::DetectImp::run@libvitis_ai_library-facedetect.so",
        "vitis::ai::DpuTaskImp::run@libvitis_ai_library-dpu_task.so",
        "vitis::ai::EfficientDetD2Imp::preprocess@libvitis_ai_library-efficientdet_d2.so",
        "vitis::ai::EfficientDetD2Imp::run@libvitis_ai_library-efficientdet_d2.so",
        "vitis::ai::FaceDetectRecogImp::run@libvitis_ai_library-facedetectrecog.so",
        "vitis::ai::FaceFeatureImp::run@libvitis_ai_library-facefeature.so",
        "vitis::ai::FaceLandmarkImp::run@libvitis_ai_library-facelandmark.so",
        "vitis::ai::FaceQuality5ptImp::run@libvitis_ai_library-facequality5pt.so",
        "vitis::ai::FaceRecogImp::run@libvitis_ai_library-facerecog.so",
        "vitis::ai::FairMotImp::run@libvitis_ai_library-fairmot.so",
        "vitis::ai::FusionCNNImp::postprocess@libvitis_ai_library-fusion_cnn.so",
        "vitis::ai::FusionCNNImp::preprocess@libvitis_ai_library-fusion_cnn.so",
        "vitis::ai::FusionCNNImp::run@libvitis_ai_library-fusion_cnn.so",
        "vitis::ai::HourglassImp::run@libvitis_ai_library-hourglass.so",
        "vitis::ai::MedicalDetectionImp::run@libvitis_ai_library-medicaldetection.so",
        "vitis::ai::MedicalSegcellImp::post_process@libvitis_ai_library-medicalsegcell.so",
        "vitis::ai::MedicalSegcellImp::run@libvitis_ai_library-medicalsegcell.so",
        "vitis::ai::MedicalSegmentationImp::run@libvitis_ai_library-medicalsegmentation.so",
        "vitis::ai::MnistClassificationImp::post_process@libvitis_ai_library-mnistclassification.so",
        "vitis::ai::MnistClassificationImp::pre_process@libvitis_ai_library-mnistclassification.so",
        "vitis::ai::MnistClassificationImp::run@libvitis_ai_library-mnistclassification.so",
        "vitis::ai::MultiTaskv3Imp::run_it@libvitis_ai_library-multitaskv3.so",
        "vitis::ai::OCRImp::preprocess@libvitis_ai_library-ocr.so",
        "vitis::ai::OCRImp::run@libvitis_ai_library-ocr.so",
        "vitis::ai::OFAYOLOImp::run@libvitis_ai_library-ofa_yolo.so",
        "vitis::ai::OpenPoseImp::run@libvitis_ai_library-openpose.so",
        "vitis::ai::PMGImp::pmg_post_process@libvitis_ai_library-pmg.so",
        "vitis::ai::PMGImp::run@libvitis_ai_library-pmg.so",
        "vitis::ai::PlateDetectImp::run@libvitis_ai_library-platedetect.so",
        "vitis::ai::PlateNumImp::run@libvitis_ai_library-platenum.so",
        "vitis::ai::PlateRecogImp::run@libvitis_ai_library-platerecog.so",
        "vitis::ai::PointPaintingImp::fusion@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPaintingImp::run@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPaintingImp::runPointPillars@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPaintingImp::runSegmentation@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPaintingImp::runSegmentationFusion@libvitis_ai_library-pointpainting.so",
        "vitis::ai::PointPillarsImp::run@libvitis_ai_library-pointpillars.so",
        "vitis::ai::PointPillarsNuscenesImp::run@libvitis_ai_library-pointpillars_nuscenes.so",
        "vitis::ai::PointPillarsNuscenesImp::sweepsFusionFilter@libvitis_ai_library-pointpillars_nuscenes.so",
        "vitis::ai::PointPillarsPost::post_process@libvitis_ai_library-pointpillars.so",
        "vitis::ai::PolypSegmentationImp::run@libvitis_ai_library-polypsegmentation.so",
        "vitis::ai::PoseDetectImp::run@libvitis_ai_library-posedetect.so",
        "vitis::ai::RGBDsegmentationImp::run@libvitis_ai_library-RGBDsegmentation.so",
        "vitis::ai::RcanImp::run@libvitis_ai_library-rcan.so",
        "vitis::ai::RefineDetImp::run@libvitis_ai_library-refinedet.so",
        "vitis::ai::ReidImp::run@libvitis_ai_library-reid.so",
        "vitis::ai::RetinaFaceImp::run@libvitis_ai_library-retinaface.so",
        "vitis::ai::RoadLineImp::run@libvitis_ai_library-lanedetect.so",
        "vitis::ai::SSDImp::run@libvitis_ai_library-ssd.so",
        "vitis::ai::Segmentation3DImp::run@libvitis_ai_library-3Dsegmentation.so",
        "vitis::ai::Segmentation3DPost::post_prec@libvitis_ai_library-3Dsegmentation.so",
        "vitis::ai::SoloImp::run@libvitis_ai_library-solo.so",
        "vitis::ai::TFSSDImp::run@libvitis_ai_library-tfssd.so",
        "vitis::ai::TextMountainImp::run@libvitis_ai_library-textmountain.so",
        "vitis::ai::UltraFastImp::run@libvitis_ai_library-ultrafast.so",
        "vitis::ai::VehicleClassificationImp::run@libvitis_ai_library-vehicleclassification.so",
        "vitis::ai::YOLOv2Imp::run@libvitis_ai_library-yolov2.so",
        "vitis::ai::YOLOv3Imp::run@libvitis_ai_library-yolov3.so",
        "vitis::ai::YOLOvXImp::run@libvitis_ai_library-yolovx.so",
        "vitis::ai::bev_preprocess@libvitis_ai_library-pointpillars.so",
        "vitis::ai::centerpoint::middle_process@libvitis_ai_library-centerpoint.so",
        "vitis::ai::centerpoint::preprocess3@libvitis_ai_library-centerpoint.so",
        "vitis::ai::fairmot_post_process@libvitis_ai_library-fairmot.so"
    ],
    "trace_xnnpp_post_process": [
        "vitis::ai::EfficientDetD2Post::postprocess@libxnnpp-xnnpp.so",
        "vitis::ai::EfficientDetD2Post::postprocess_kernel@libxnnpp-xnnpp.so",
        "vitis::ai::MedicalSegmentationPost::medicalsegmentation_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::post_process_seg@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_det@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_seg@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskPostProcessImp::process_seg_visualization@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskv3PostProcessImp::post_process@libxnnpp-xnnpp.so",
        "vitis::ai::MultiTaskv3PostProcessImp::post_process_visualization@libxnnpp-xnnpp.so",
        "vitis::ai::OCRPostImp::process@libxnnpp-xnnpp.so",
        "vitis::ai::PointPillarsNuscenesPost::postprocess@libxnnpp-xnnpp.so",
        "vitis::ai::RefineDetPost::refine_det_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::RetinaFacePost::retinaface_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::RoadLinePost::road_line_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::SSDPost::ssd_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::TFSSDPost::ssd_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::TextMountainPostImp::process@libxnnpp-xnnpp.so",
        "vitis::ai::UltraFastPostImp::post_process@libxnnpp-xnnpp.so",
        "vitis::ai::X_Autonomous3DPost::process@libxnnpp-xnnpp.so",
        "vitis::ai::classification_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::face_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::face_feature_post_process_fixed@libxnnpp-xnnpp.so",
        "vitis::ai::face_feature_post_process_float@libxnnpp-xnnpp.so",
        "vitis::ai::face_landmark_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::face_quality5pt_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::hourglass_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::medicaldetection::MedicalDetectionPost::medicaldetection_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::ofa_yolo_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::open_pose_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::plate_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::plate_num_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::pose_detect_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::post_process@libxnnpp-xnnpp.so",
        "vitis::ai::process_kernel@libxnnpp-xnnpp.so",
        "vitis::ai::rcan_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::reid_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::segmentation_post_process_8UC1@libxnnpp-xnnpp.so",
        "vitis::ai::segmentation_post_process_8UC3@libxnnpp-xnnpp.so",
        "vitis::ai::solo_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::solo_post_process_batch@libxnnpp-xnnpp.so",
        "vitis::ai::tfrefinedet::TFRefineDetPost::tfrefinedet_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::vehicleclassification_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::yolov2_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::yolov3_post_process@libxnnpp-xnnpp.so",
        "vitis::ai::yolovx_post_process@libxnnpp-xnnpp.so"
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

trace_va_timeout = 60
trace_xat_timeout = 60
trace_max_timeout = 120
trace_fg_timeout = 30
trace_max_fg_timeout = 60

default_runmode = "normal"
#default_runmode = "debug"

traceCfgDefaule = {
    'collector': {},
    'tracer': {
        'power': {
            'disable': True,
        },
        'sched': {
            'disable': True,
        },
    },
    'trace': {
        "enable_trace_list": ["vitis-ai-library", "vart", "xnnpp_post_process", "opencv", "custom"]
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
