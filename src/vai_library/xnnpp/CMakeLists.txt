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

set(PBF_NAME
    anchor_generator
    grid_anchor_generator
    mean_stddev_box_coder
    region_similarity_calculator
    argmax_matcher
    hyperparams
    model
    square_box_coder
    bipartite_matcher
    eval
    image_resizer
    multiscale_anchor_generator
    ssd_anchor_generator
    box_coder
    faster_rcnn_box_coder
    input_reader
    optimizer
    ssd
    box_predictor
    faster_rcnn
    keypoint_box_coder
    pipeline
    string_int_label_map
    calibration
    flexible_grid_anchor_generator
    losses
    post_processing
    graph_rewriter
    matcher
    preprocessor
    train)

foreach(PBF ${PBF_NAME})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.cc
           ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.h
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/include/object_detection/protos/${PBF}.proto
    COMMAND
      protobuf::protoc ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/include/object_detection/protos/${PBF}.proto
    COMMENT "Running C++ protocol buffer compiler on proto ${PBF}"
    VERBATIM)
  list(APPEND LIST_PBSRC_CC
       ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.cc)
  list(APPEND LIST_PBSRC_H
       ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.h)
  list(APPEND LIST_PROTO_DST include/object_detection/protos/${PBF}.proto)
endforeach()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/word_list.txt
    ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/word_list.txt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cifar10_label.inc
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/cifar10_label.txt
    ${CMAKE_CURRENT_BINARY_DIR}/cifar10_label.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/cifar10_label.txt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fmnist_label.inc
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/fmnist_label.txt
    ${CMAKE_CURRENT_BINARY_DIR}/fmnist_label.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/fmnist_label.txt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/orien_label.inc
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/orien_label.txt
    ${CMAKE_CURRENT_BINARY_DIR}/orien_label.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/orien_label.txt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/word_131.inc
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/word_131.txt
    ${CMAKE_CURRENT_BINARY_DIR}/word_131.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/word_131.txt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/car_color_chen.inc
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/car_color_chen.txt
    ${CMAKE_CURRENT_BINARY_DIR}/car_color_chen.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/car_color_chen.txt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vehicle_make.inc
  COMMAND
    bash -ex
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/vehicle_make.txt
    ${CMAKE_CURRENT_BINARY_DIR}/vehicle_make.inc
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/vehicle_make.txt)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vehicle_type.inc
  COMMAND
    bash -ex
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/vehicle_type.txt
    ${CMAKE_CURRENT_BINARY_DIR}/vehicle_type.inc
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vehicleclassification/vehicle_type.txt)

vai_add_library(
  SRCS
  src/segmentation.cpp
  src/centerpoint/centerpoint.cpp
  src/centerpoint/centerpointutil.hpp
  src/apply_nms.cpp
  src/platedetect.cpp
  src/platenum.cpp
  src/facedetect.cpp
  src/facefeature.cpp
  src/classification/classification.cpp
  src/vehicleclassification/vehicleclassification.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc
  ${CMAKE_CURRENT_BINARY_DIR}/cifar10_label.inc
  ${CMAKE_CURRENT_BINARY_DIR}/fmnist_label.inc
  ${CMAKE_CURRENT_BINARY_DIR}/orien_label.inc
  ${CMAKE_CURRENT_BINARY_DIR}/word_131.inc
  ${CMAKE_CURRENT_BINARY_DIR}/car_color_chen.inc
  ${CMAKE_CURRENT_BINARY_DIR}/vehicle_make.inc
  ${CMAKE_CURRENT_BINARY_DIR}/vehicle_type.inc
  src/openpose.cpp
  src/hourglass.cpp
  src/facequality5pt.cpp
  src/facelandmark.cpp
  src/multitask/multitask_imp.cpp
  src/multitask/multitask.cpp
  src/multitask/prior_boxes.cpp
  src/multitask/ssd_detector.cpp
  src/multitaskv3/multitaskv3_imp.cpp
  src/multitaskv3/multitaskv3.cpp
  src/multitaskv3/prior_boxes.cpp
  src/multitaskv3/ssd_detector.cpp
  src/yolov2.cpp
  src/yolov3.cpp
  src/ofa_yolo.cpp
  src/yolovx.cpp
  src/yolov6.cpp
  src/solo.cpp
  src/posedetect.cpp
  src/reid.cpp
  src/rcan.cpp
  src/refinedet/refinedet.cpp
  src/refinedet/refine_det_post.cpp
  src/refinedet/prior_boxes.cpp
  src/refinedet/ssd_detector.cpp
  src/tfrefinedet/tfrefinedet.cpp
  src/tfrefinedet/tfrefinedet_post.cpp
  src/tfrefinedet/ssd_detector.cpp
  src/roadline/roadline.cpp
  src/roadline/road_line_post.cpp
  src/roadline/predict.cpp
  src/roadline/ipm_info.cpp
  src/tfssd/tfssd.cpp
  src/tfssd/tfssd_post.cpp
  src/tfssd/tfssd_detector.cpp
  src/tfssd/anchor/ssd_anchor.cpp
  src/tfssd/anchor/grid_anchor.cpp
  src/tfssd/anchor/multiscale_anchor.cpp
  src/tfssd/anchor/flexiblegrid_anchor.cpp
  src/tfssd/anchor/anchorbase.cpp
  ${LIST_PBSRC_CC}
  ${LIST_PBSRC_H}
  src/ssd/ssd.cpp
  src/ssd/ssd_post.cpp
  src/ssd/ssd_detector.cpp
  src/ssd/prior_boxes.cpp
  src/medicalsegmentation/medicalsegmentation.cpp
  src/medicalsegmentation/medicalsegmentation_post.cpp
  src/medicaldetection/medicaldetection.cpp
  src/medicaldetection/medicaldetection_post.cpp
  src/medicaldetection/priorbox.cpp
  src/medicaldetection/ssd_detector.cpp
  src/retinaface/retinaface.cpp
  src/retinaface/retinaface_post.cpp
  src/retinaface/retinaface_detector.cpp
  src/retinaface/anchor.cpp
  src/pointpillars_nuscenes/pointpillars_nuscenes.cpp
  src/pointpillars_nuscenes/pointpillars_nuscenes_post.cpp
  src/pointpillars_nuscenes/anchor.cpp
  src/pointpillars_nuscenes/utils.cpp
  src/pointpillars_nuscenes/sigmoid_table.cpp
  src/ultrafast/ultrafast.cpp
  src/ultrafast/ultrafast_postimp.cpp
  src/ocr/ocr.cpp
  src/ocr/ocr_postimp.cpp
  src/x_autonomous3d/x_autonomous3d.cpp
  src/x_autonomous3d/x_autonomous3d_post.hpp
  src/x_autonomous3d/x_autonomous3d_post.cpp
  src/x_autonomous3d/utils.hpp
  src/x_autonomous3d/utils.cpp
  src/textmountain/textmountain.cpp
  src/textmountain/textmountain_postimp.cpp
  src/efficientdet_d2/efficientdet_d2.cpp
  src/efficientdet_d2/anchor.cpp
  src/efficientdet_d2/object_detection_base.cpp
  src/efficientdet_d2/postprocess.cpp
  PUBLIC_REQUIRE
  vart::util
  ${PROJECT_NAME}::math
  ${PROJECT_NAME}::model_config
  glog::glog
  ${OpenCV_LIBS})

target_include_directories(
  ${COMPONENT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include>)
install(FILES ${LIST_PROTO_DST} ${LIST_PBSRC_H}
        DESTINATION include/object_detection/protos)
