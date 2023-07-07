#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(XilinxCommon)
include(XilinxTest)
include(CheckIncludeFileCXX)
include(CheckIncludeFile)

option(BUILD_DOXYGEN_DOC "enable doxygen or not" OFF)
set(SAMPLE_INATLL_PATH share/${PROJECT_NAME}/samples)
set(SAMPLE_ONNX_INSTALL_PATH share/${PROJECT_NAME}/samples_onnx)
set(DEMO_INATLL_PATH share/${PROJECT_NAME}/apps)

find_path(DRM_PATH drm)
if(NOT ${DRM_PATH} EQUAL "DRM_PATH-NOTFOUND")
  set(CMAKE_REQUIRED_INCLUDES ${DRM_PATH}/drm)
endif()

# 3Dsegmentation
vai_overview_add_test(3Dsegmentation 3Dsegmentation Segmentation3D NO_VIDEO
                      NO_CONFIG)
vai_overview_add_performance(3Dsegmentation 3Dsegmentation Segmentation3D
                             NO_CONFIG)
vai_overview_add_accuracy(3Dsegmentation 3Dsegmentation)
# centerpoint
vai_overview_add_test(
  test_file_centerpoint
  centerpoint
  CenterPoint
  NO_VIDEO
  NO_CONFIG
  NAME
  test_file_centerpoint)
vai_overview_add_performance(centerpoint centerpoint CenterPoint NO_VIDEO
                             NO_CONFIG)
vai_overview_add_accuracy(centerpoint centerpoint)

# bcc
vai_overview_add_test(bcc bcc BCC)
vai_overview_add_performance(bcc bcc BCC)
vai_overview_add_accuracy(bcc bcc)
vai_overview_add_accuracy(bcc_mt bcc)

# bts
if(ENABLE_BTS AND (NOT DISABLE_ALL))
  vai_overview_add_test(bts bts BTS)
  vai_overview_add_performance(bts bts BTS)
  vai_overview_add_accuracy(bts bts)
endif()

# classification
vai_overview_add_test(classification classification Classification)
vai_overview_add_performance(classification classification Classification)
vai_overview_add_accuracy(classification classification)
vai_overview_add_accuracy(classification_mt classification)

vai_overview_add_test(classification_squeezenet classification Classification
                      NO_VIDEO NO_CONFIG)
vai_overview_add_performance(classification_squeezenet classification
                             Classification NO_CONFIG)
vai_overview_add_accuracy(classification_squeezenet classification)

# vehicleclassification
vai_overview_add_test(vehicleclassification vehicleclassification
                      VehicleClassification)
vai_overview_add_performance(vehicleclassification vehicleclassification
                             VehicleClassification)
vai_overview_add_accuracy(vehicleclassification vehicleclassification)
vai_overview_add_accuracy(vehicleclassification_mt vehicleclassification)

# covid19segmentation
vai_overview_add_test(covid19segmentation covid19segmentation
                      Covid19Segmentation NO_VIDEO NO_CONFIG)
vai_overview_add_performance(covid19segmentation covid19segmentation
                             Covid19Segmentation NO_CONFIG)
vai_overview_add_accuracy(covid19segmentation covid19segmentation)

# facedetect
vai_overview_add_test(facedetect facedetect FaceDetect)
vai_overview_add_performance(facedetect facedetect FaceDetect)
vai_overview_add_accuracy(facedetect facedetect)
vai_overview_add_accuracy(facedetect_mt facedetect)

# facefeature
vai_overview_add_test(facefeature facefeature FaceFeature NO_VIDEO)
vai_overview_add_performance(facefeature facefeature FaceFeature)
vai_overview_add_accuracy(facefeature facefeature)
vai_overview_add_accuracy(facerec_pretrain_res20 facefeature)

# facelandmark
vai_overview_add_test(facelandmark facelandmark FaceLandmark NO_VIDEO)
vai_overview_add_performance(facelandmark facelandmark FaceLandmark)
vai_overview_add_accuracy(facelandmark facelandmark)
vai_overview_add_accuracy(facelandmark_mt facelandmark)

# facequality5pt
vai_overview_add_test(facequality5pt facequality5pt FaceQuality5pt NO_VIDEO)
vai_overview_add_performance(facequality5pt facequality5pt FaceQuality5pt)
vai_overview_add_accuracy(facequality5pt facequality5pt)
vai_overview_add_accuracy(facequality5pt_mt facequality5pt)

# hourglass
vai_overview_add_test(hourglass hourglass Hourglass)
vai_overview_add_performance(hourglass hourglass Hourglass)
vai_overview_add_accuracy(hourglass hourglass REQUIRE ${PROJECT_NAME}::hourglass
                                                      json-c)

# movenet
vai_overview_add_test(movenet movenet Movenet)
vai_overview_add_performance(movenet movenet Movenet)
vai_overview_add_accuracy(movenet movenet REQUIRE ${PROJECT_NAME}::movenet
                                                  json-c)
# lanedetect
vai_overview_add_test(lanedetect lanedetect RoadLine)
vai_overview_add_performance(lanedetect lanedetect RoadLine)
vai_overview_add_accuracy(lanedetect lanedetect)

# medicaldetection
vai_overview_add_test(medicaldetection medicaldetection MedicalDetection)
vai_overview_add_performance(medicaldetection medicaldetection MedicalDetection)
vai_overview_add_accuracy(medicaldetection medicaldetection)
vai_overview_add_accuracy(medicaldetection_mt medicaldetection)

# medicalsegcell
vai_overview_add_test(medicalsegcell medicalsegcell MedicalSegcell)
vai_overview_add_performance(medicalsegcell medicalsegcell MedicalSegcell)
vai_overview_add_accuracy(medicalsegcell medicalsegcell)
vai_overview_add_accuracy(medicalsegcell_mt medicalsegcell)

# medicalsegmentation
vai_overview_add_test(medicalsegmentation medicalsegmentation
                      MedicalSegmentation NO_VIDEO)
vai_overview_add_performance(medicalsegmentation medicalsegmentation
                             MedicalSegmentation)
vai_overview_add_accuracy(medicalsegmentation medicalsegmentation)
vai_overview_add_accuracy(medicalsegmentation_mt medicalsegmentation)

# centerpoint

# mnistclassification
vai_overview_add_test(mnistclassification mnistclassification
                      MnistClassification NO_VIDEO NO_CONFIG)
vai_overview_add_performance(mnistclassification mnistclassification
                             MnistClassification NO_CONFIG)

# multitask
vai_overview_add_test(multitask multitask MultiTask8UC3)
vai_overview_add_performance(multitask multitask MultiTask8UC3)
vai_overview_add_accuracy(multitask multitask)
vai_overview_add_accuracy(multitask_mt multitask)

# multitaskv3
vai_overview_add_test(multitaskv3 multitaskv3 MultiTaskv38UC3)
vai_overview_add_performance(multitaskv3 multitaskv3 MultiTaskv38UC1)
vai_overview_add_accuracy(multitaskv3_detection multitaskv3)
vai_overview_add_accuracy(multitaskv3_segmentation multitaskv3)

# openpose
vai_overview_add_test(openpose openpose OpenPose)
vai_overview_add_performance(openpose openpose OpenPose)
vai_overview_add_accuracy(openpose openpose REQUIRE ${PROJECT_NAME}::openpose
                                                    json-c)
vai_overview_add_accuracy(openpose_mt openpose REQUIRE ${PROJECT_NAME}::openpose
                                                       json-c)

# platedetect
vai_overview_add_test(platedetect platedetect PlateDetect)
vai_overview_add_performance(platedetect platedetect PlateDetect)
vai_overview_add_accuracy(platedetect platedetect)
vai_overview_add_accuracy(platedetect_mt platedetect)

# platenum
vai_overview_add_test(platenum platenum PlateNum)
vai_overview_add_performance(platenum platenum PlateNum)
vai_overview_add_accuracy(platenum platenum)
vai_overview_add_accuracy(platenum_mt platenum)

# pmg
vai_overview_add_test(pmg pmg PMG)
vai_overview_add_performance(pmg pmg PMG)
vai_overview_add_accuracy(pmg pmg)

# pointpillars
vai_overview_add_test(
  test_bin_pointpillars
  pointpillars
  PointPillars
  NO_VIDEO
  NO_CONFIG
  NAME
  test_bin_pointpillars)
vai_overview_add_performance(pointpillars pointpillars PointPillars NO_CONFIG)
vai_overview_add_accuracy(pointpillars pointpillars)

# pointpillars_nuscenes
vai_overview_add_test(
  test_bin_pointpillars_nuscenes
  pointpillars_nuscenes
  PointPillarsNuscenes
  NO_VIDEO
  NO_CONFIG
  NAME
  test_bin_pointpillars_nuscenes)
vai_overview_add_performance(pointpillars_nuscenes pointpillars_nuscenes
                             PointPillarsNuscenes NO_VIDEO NO_CONFIG)
vai_overview_add_accuracy(pointpillars_nuscenes pointpillars_nuscenes)

# pointpainting
vai_overview_add_test(
  test_bin_pointpainting
  pointpainting
  PointPainting
  NO_VIDEO
  NO_CONFIG
  NAME
  test_bin_pointpainting)
vai_overview_add_performance(pointpainting pointpainting PointPainting NO_VIDEO
                             NO_CONFIG)
vai_overview_add_accuracy(pointpainting_seg pointpainting)
vai_overview_add_accuracy(pointpainting_nuscenes pointpainting)
vai_overview_add_accuracy(pointpainting pointpainting)

# x_autonomous3d
if(ENABLE_X_AUTONOMOUS3D AND (NOT DISABLE_ALL))
  vai_overview_add_test(
    test_bin_x_autonomous3d
    x_autonomous3d
    X_Autonomous3D
    NO_VIDEO
    NO_CONFIG
    NAME
    test_bin_x_autonomous3d)
  vai_overview_add_performance(x_autonomous3d x_autonomous3d X_Autonomous3D
                               NO_VIDEO NO_CONFIG)
  # vai_overview_add_accuracy(x_autonomous3d x_autonomous3d)
endif()

# clocs
vai_overview_add_test(
  test_bin_clocs
  clocs
  Clocs
  NO_VIDEO
  NO_CONFIG
  NAME
  test_bin_clocs)
vai_overview_add_performance(clocs clocs Clocs NO_VIDEO NO_CONFIG)
vai_overview_add_accuracy(clocs clocs)

# posedetect
vai_overview_add_test(posedetect posedetect PoseDetect)
vai_overview_add_performance(posedetect posedetect PoseDetect)
vai_overview_add_accuracy(posedetect posedetect)
vai_overview_add_accuracy(posedetect_mt posedetect)

vai_overview_add_test(
  posedetect_with_ssd
  posedetect
  PoseDetect
  NO_CONFIG
  REQUIRE
  ${PROJECT_NAME}::posedetect
  ${PROJECT_NAME}::ssd)

# refinedet
vai_overview_add_test(refinedet refinedet RefineDet)
vai_overview_add_performance(refinedet refinedet RefineDet)
vai_overview_add_accuracy(refinedet refinedet)
vai_overview_add_accuracy(refinedet_mt refinedet)

vai_overview_add_test(refinedet_tf refinedet RefineDet NO_VIDEO NO_CONFIG)

# reid
vai_overview_add_test(reid reid Reid NO_VIDEO NO_CONFIG)
vai_overview_add_performance(reid reid Reid)
vai_overview_add_accuracy(reid reid)

# rcan
vai_overview_add_test(rcan rcan Rcan)
vai_overview_add_performance(rcan rcan Rcan)
vai_overview_add_accuracy(rcan rcan Rcan)

# solo
vai_overview_add_performance(solo solo Solo)
vai_overview_add_test(solo solo Solo)
# fairmot
vai_overview_add_test(fairmot fairmot FairMot)
vai_overview_add_performance(fairmot fairmot FairMot)
vai_overview_add_accuracy(fairmot fairmot REQUIRE ${PROJECT_NAME}::fairmot
                                                  ${PROJECT_NAME}::reidtracker)
# vai_overview_add_accuracy(fairmot fairmot REQUIRE ${PROJECT_NAME}::FairMot
# ${PROJECT_NAME}::ReidTracker)

# retinaface
vai_overview_add_test(retinaface retinaface RetinaFace)
vai_overview_add_performance(retinaface retinaface RetinaFace)
vai_overview_add_accuracy(retinaface retinaface)

# segmentation
vai_overview_add_test(segmentation segmentation Segmentation8UC3)
vai_overview_add_performance(segmentation segmentation Segmentation8UC3)
foreach(TMPMODEL segmentation segmentation_mt endov unet)
  vai_overview_add_accuracy(${TMPMODEL} segmentation
                            REQUIRE ${PROJECT_NAME}::segmentation json-c)
endforeach(TMPMODEL)

# RGBDsegmentation
vai_overview_add_test(RGBDsegmentation RGBDsegmentation RGBDsegmentation
                      NO_VIDEO NO_CONFIG)
vai_overview_add_performance(RGBDsegmentation RGBDsegmentation RGBDsegmentation
                             NO_CONFIG)
vai_overview_add_accuracy(RGBDsegmentation RGBDsegmentation)

# pmrid
vai_overview_add_test(pmrid pmrid pmrid NO_VIDEO NO_CONFIG)
vai_overview_add_performance(pmrid pmrid pmrid NO_CONFIG)
vai_overview_add_accuracy(pmrid pmrid)

# C2D2_lite
vai_overview_add_test(c2d2_lite c2d2_lite C2D2_lite NO_VIDEO NO_CONFIG)
vai_overview_add_performance(c2d2_lite c2d2_lite C2D2_lite NO_CONFIG)

# ssd
vai_overview_add_test(ssd ssd SSD)
vai_overview_add_performance(ssd ssd SSD)
foreach(TMPMODEL ssd_mt ssd_pedestrian_pruned_0_97 ssd_traffic_pruned_0_9
                 ssd_adas_pruned_0_95 ssd_mobilenet_v2 mlperf_ssd_resnet34_tf)
  vai_overview_add_accuracy(${TMPMODEL} ssd)
endforeach(TMPMODEL)

# tfssd
vai_overview_add_test(tfssd tfssd TFSSD)
vai_overview_add_performance(tfssd tfssd TFSSD)
vai_overview_add_accuracy(tfssd tfssd REQUIRE ${PROJECT_NAME}::tfssd json-c)
vai_overview_add_accuracy(tfssd_mt tfssd REQUIRE ${PROJECT_NAME}::tfssd json-c)

# yolov2
vai_overview_add_test(yolov2 yolov2 YOLOv2)
vai_overview_add_performance(yolov2 yolov2 YOLOv2)
vai_overview_add_accuracy(yolov2_mt yolov2)

# yolov3
vai_overview_add_test(yolov3 yolov3 YOLOv3)
vai_overview_add_performance(yolov3 yolov3 YOLOv3)
vai_overview_add_accuracy(yolov3_mt yolov3)
foreach(TMPMODEL yolov3_adas_pruned_0_9 yolov3_voc yolov3_voc_tf yolov3_bdd
                 tiny_yolov3_vmss yolov3_coco_416_tf2_mt)
  vai_overview_add_accuracy(${TMPMODEL} yolov3)
endforeach(TMPMODEL)

# polypsegmentation
vai_overview_add_test(polypsegmentation polypsegmentation PolypSegmentation)
vai_overview_add_performance(polypsegmentation polypsegmentation
                             PolypSegmentation)
vai_overview_add_accuracy(polypsegmentation_mt polypsegmentation)
vai_overview_add_accuracy(polypsegmentation polypsegmentation)

# ofa_yolo
vai_overview_add_test(ofa_yolo ofa_yolo OFAYOLO)
vai_overview_add_performance(ofa_yolo ofa_yolo OFAYOLO)
vai_overview_add_accuracy(ofa_yolo_mt ofa_yolo)

# yolovx
vai_overview_add_test(yolovx yolovx YOLOvX)
vai_overview_add_performance(yolovx yolovx YOLOvX)
vai_overview_add_accuracy(yolovx_mt yolovx)
vai_overview_add_accuracy(yolovx_nano_mt yolovx)
vai_overview_add_accuracy(yolovx_coco_mt yolovx REQUIRE ${PROJECT_NAME}::yolovx
 json-c)


# yolov6
vai_overview_add_test(yolov6 yolov6 YOLOv6)
vai_overview_add_accuracy(yolov6_mt yolov6)
vai_overview_add_performance(yolov6 yolov6 YOLOv6 NO_CONFIG REQUIRE
                             ${PROJECT_NAME}::yolov6)

# bevdet
vai_overview_add_test(bevdet bevdet bevdet NO_VIDEO NO_CONFIG)
vai_overview_add_performance(bevdet bevdet bevdet NO_CONFIG)
vai_overview_add_accuracy(bevdet_mt bevdet)

# arflow
vai_overview_add_test(arflow arflow ARFlow NO_VIDEO NO_CONFIG)
vai_overview_add_performance(arflow arflow ARFlow NO_CONFIG)
vai_overview_add_accuracy(arflow arflow)

# yolov4
vai_overview_add_test(yolov4 yolov4 YOLOv3 NO_CONFIG REQUIRE
                      ${PROJECT_NAME}::yolov3)
vai_overview_add_performance(yolov4 yolov4 YOLOv3 NO_CONFIG REQUIRE
                             ${PROJECT_NAME}::yolov3)
vai_overview_add_accuracy(yolov4_mt yolov4 REQUIRE ${PROJECT_NAME}::yolov3)

# yolov5
vai_overview_add_test(yolov5 yolov5 YOLOv3 NO_CONFIG REQUIRE
                      ${PROJECT_NAME}::yolov3)
vai_overview_add_performance(yolov5 yolov5 YOLOv3 NO_CONFIG REQUIRE
                             ${PROJECT_NAME}::yolov3)
vai_overview_add_accuracy(yolov5_mt yolov5 REQUIRE ${PROJECT_NAME}::yolov3)

# ultrafast
vai_overview_add_test(ultrafast ultrafast UltraFast NO_VIDEO)
vai_overview_add_performance(ultrafast ultrafast UltraFast)
vai_overview_add_accuracy(ultrafast ultrafast)

# ocr
vai_overview_add_test(ocr ocr OCR NO_VIDEO)
vai_overview_add_performance(ocr ocr OCR)
vai_overview_add_accuracy(ocr ocr)

# textmountain
vai_overview_add_test(textmountain textmountain TextMountain NO_VIDEO)
vai_overview_add_performance(textmountain textmountain TextMountain)
vai_overview_add_accuracy(textmountain textmountain)

# efficientdet_d2
vai_overview_add_test(efficientdet_d2 efficientdet_d2 EfficientDetD2)
vai_overview_add_performance(efficientdet_d2 efficientdet_d2 EfficientDetD2)
vai_overview_add_accuracy(efficientdet_d2 efficientdet_d2)

# bosch related: bosch_ssdobjdet bosch_fcnsemseg
if( ENABLE_BOSCH_FCNSEMSEG  AND (NOT DISABLE_ALL))
  vai_overview_add_performance(bosch_fcnsemseg bosch_fcnsemseg Bosch_Fcnsemseg)
endif()
if( ENABLE_BOSCH_SSDOBJDET  AND (NOT DISABLE_ALL))
  vai_overview_add_performance(bosch_ssdobjdet bosch_ssdobjdet Bosch_Ssdobjdet)
endif()

# monodepth2
vai_overview_add_test(monodepth2 monodepth2 Monodepth2 NO_VIDEO)
vai_overview_add_performance(monodepth2 monodepth2 Monodepth2)
vai_overview_add_accuracy(monodepth2 monodepth2)

#cflownet
vai_overview_add_test(test_bin_cflownet cflownet  Cflownet NO_VIDEO NO_CONFIG NAME test_bin_cflownet)
vai_overview_add_performance(cflownet cflownet  Cflownet NO_CONFIG)
vai_overview_add_accuracy(cflownet cflownet)

# demos demo_yolov3
vai_overview_add_demo(
  NAME
  yolov3
  SRCS
  samples/dpu_task/yolov3/demo_yolov3.cpp
  REQUIRE
  ${PROJECT_NAME}::dpu_task
  ${PROJECT_NAME}::xnnpp
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FILES
  samples/dpu_task/yolov3/demo_yolov3.cpp
  samples/dpu_task/yolov3/readme
  VAI_INSTALL_BUILD
  samples/dpu_task/yolov3/build.sh)

vai_overview_add_dpu_task(
  NAME
  fadnet
  SRCS
  samples/dpu_task/fadnet/fadnet.cpp
  REQUIRE
  ${PROJECT_NAME}::dpu_task
  ${PROJECT_NAME}::xnnpp
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FOLDER
  samples/dpu_task/fadnet
  VAI_INSTALL_BUILD
  samples/dpu_task/fadnet/build.sh)

vai_overview_add_dpu_task(
  NAME
  fadnet_corr
  SRCS
  samples/dpu_task/fadnet_corr/fadnet_corr.cpp
  REQUIRE
  ${PROJECT_NAME}::dpu_task
  ${PROJECT_NAME}::xnnpp
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FOLDER
  samples/dpu_task/fadnet_corr
  VAI_INSTALL_BUILD
  samples/dpu_task/fadnet_corr/build.sh)

vai_overview_add_dpu_task(
  NAME
  ssr
  SRCS
  samples/dpu_task/ssr/ssr.cpp
  REQUIRE
  ${PROJECT_NAME}::dpu_task
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FOLDER
  samples/dpu_task/ssr
  VAI_INSTALL_BUILD
  samples/dpu_task/ssr/build.sh)

vai_overview_add_dpu_task(
  NAME
  superpoint
  SRCS
  samples/dpu_task/superpoint/superpoint.cpp
  REQUIRE
  ${PROJECT_NAME}::dpu_task
  ${PROJECT_NAME}::math
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FOLDER
  samples/dpu_task/superpoint
  VAI_INSTALL_BUILD
  samples/dpu_task/superpoint/build.sh)

vai_overview_add_dpu_task(
  NAME
  hfnet
  SRCS
  samples/dpu_task/hfnet/hfnet.cpp
  REQUIRE
  ${PROJECT_NAME}::dpu_task
  ${PROJECT_NAME}::math
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FOLDER
  samples/dpu_task/hfnet
  VAI_INSTALL_BUILD
  samples/dpu_task/hfnet/build.sh)

if(ENABLE_PSMNET AND (NOT DISABLE_ALL))
  vai_overview_add_dpu_task(
    NAME
    psmnet
    SRCS
    samples/dpu_task/psmnet/psmnet.cpp
    samples/dpu_task/psmnet/cpu_op.cpp
    REQUIRE
    ${PROJECT_NAME}::dpu_task
    ${PROJECT_NAME}::math
    ${PROJECT_NAME}::benchmark
    VAI_INSTALL_FOLDER
    samples/dpu_task/psmnet
    VAI_INSTALL_BUILD
    samples/dpu_task/psmnet/build.sh)
endif()

# apps seg_and_pose_detect
vai_overview_add_app(
  NAME
  seg_and_pose_detect
  SRCS
  apps/seg_and_pose_detect/seg_and_pose_detect.cpp
  REQUIRE
  ${PROJECT_NAME}::multitask
  ${PROJECT_NAME}::posedetect
  ${PROJECT_NAME}::ssd
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FILES
  apps/seg_and_pose_detect/seg_and_pose_detect.cpp
  apps/seg_and_pose_detect/readme
  VAI_INSTALL_BUILD
  apps/seg_and_pose_detect/build.sh
  apps/seg_and_pose_detect/builddrm.sh)

vai_overview_add_app(
  NAME
  multitask_v3_quad_windows
  SRCS
  apps/multitaskv3with4windows/multitaskv3_quad_windows.cpp
  REQUIRE
  ${PROJECT_NAME}::multitaskv3
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FILES
  apps/multitaskv3with4windows/multitaskv3_quad_windows.cpp
  apps/multitaskv3with4windows/readme
  VAI_INSTALL_BUILD
  apps/multitaskv3with4windows/build.sh
  apps/multitaskv3with4windows/builddrm.sh)

# segs_and_roadline_detect
vai_overview_add_app(
  NAME
  segs_and_roadline_detect
  SRCS
  apps/segs_and_roadline_detect/segs_and_roadline_detect.cpp
  REQUIRE
  ${PROJECT_NAME}::multitask
  ${PROJECT_NAME}::lanedetect
  ${PROJECT_NAME}::ssd
  ${PROJECT_NAME}::benchmark
  VAI_INSTALL_FILES
  apps/segs_and_roadline_detect/segs_and_roadline_detect.cpp
  apps/segs_and_roadline_detect/readme
  VAI_INSTALL_BUILD
  apps/segs_and_roadline_detect/build.sh
  apps/segs_and_roadline_detect/builddrm.sh)

# tutorials
if(ENABLE_TUTORIAL AND (NOT DISABLE_ALL))
  install(FILES tutorials/tf2_flow/app_mt.py
          DESTINATION share/vitis_ai_library/tutorials/tf2_flow/)
  install(
    FILES tutorials/04_tensorflow_ai_optimizer/app_mt.py
    DESTINATION share/vitis_ai_library/tutorials/04_tensorflow_ai_optimizer/)
  install(FILES tutorials/pytorch-subgraphs/main_graph_runner.py
          DESTINATION share/vitis_ai_library/tutorials/pytorch-subgraphs/)
  install(FILES tutorials/tf2_quant_fine_tune/app_mt.py
          DESTINATION share/vitis_ai_library/tutorials/tf2_quant_fine_tune/)
  install(FILES tutorials/05-tf1_keras_optimizer/app_mt.py
          DESTINATION share/vitis_ai_library/tutorials/05-tf1_keras_optimizer/)
  install(FILES tutorials/11-tf2_var_autoenc/app_mt.py
          DESTINATION share/vitis_ai_library/tutorials/11-tf2_var_autoenc/)
  install(FILES tutorials/01-caffe_cats_vs_dogs/app_mt.py
          DESTINATION share/vitis_ai_library/tutorials/01-caffe_cats_vs_dogs/)
  install(
    FILES tutorials/10-RF_modulation_recognition/app_mt.py
    DESTINATION share/vitis_ai_library/tutorials/10-RF_modulation_recognition/)
endif()

# demo
if(ENABLE_DEMO AND (NOT DISABLE_ALL))
  add_subdirectory(demo)
endif()

# onnx sample
if(ENABLE_ONNX AND (NOT DISABLE_ALL))
  add_subdirectory(samples_onnx)
endif()
