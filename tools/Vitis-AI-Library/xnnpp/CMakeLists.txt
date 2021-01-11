#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required(VERSION 3.5)
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

#dpu_model_param
set(PBF_NAME anchor_generator  grid_anchor_generator  mean_stddev_box_coder  region_similarity_calculator  argmax_matcher  hyperparams  model  square_box_coder  bipartite_matcher  eval  image_resizer  multiscale_anchor_generator  ssd_anchor_generator  box_coder  faster_rcnn_box_coder  input_reader  optimizer  ssd  box_predictor  faster_rcnn  keypoint_box_coder  pipeline  string_int_label_map  calibration  flexible_grid_anchor_generator  losses  post_processing  graph_rewriter  matcher  preprocessor  train)

foreach(PBF ${PBF_NAME})
  add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.h
    DEPENDS   ${CMAKE_CURRENT_SOURCE_DIR}/include/object_detection/protos/${PBF}.proto
    COMMAND  protobuf::protoc
    ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR}
    --proto_path ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/object_detection/protos/${PBF}.proto
    COMMENT "Running C++ protocol buffer compiler on proto ${PBF}"
    VERBATIM )
  list(APPEND LIST_PBSRC_CC ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.cc)
  list(APPEND LIST_PBSRC_H ${CMAKE_CURRENT_BINARY_DIR}/object_detection/protos/${PBF}.pb.h)
  list(APPEND LIST_PROTO_DST include/object_detection/protos/${PBF}.proto)
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND Other_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
endif()
include(XilinxCommon)
include(XilinxVersion)
include(CheckIncludeFileCXX)
check_include_file_cxx(arm_neon.h HAVE_NEON)
if(HAVE_NEON)
  add_definitions(-DENABLE_NEON)
	add_definitions(-DHAVE_NEON)
  endif(HAVE_NEON)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc
  COMMAND bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/word_list.txt ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/word_list.txt
  )

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cifar10_label.inc
  COMMAND bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/cifar10_label.txt ${CMAKE_CURRENT_BINARY_DIR}/cifar10_label.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/cifar10_label.txt
  )

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fmnist_label.inc
  COMMAND bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/fmnist_label.txt ${CMAKE_CURRENT_BINARY_DIR}/fmnist_label.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/convert_word_list.sh ${CMAKE_CURRENT_SOURCE_DIR}/src/classification/fmnist_label.txt
  )

add_library(${COMPONENT_NAME} SHARED
  include/vitis/ai/nnpp/segmentation.hpp
  src/segmentation.cpp
  include/vitis/ai/nnpp/apply_nms.hpp
  src/apply_nms.cpp
  include/vitis/ai/nnpp/platedetect.hpp
  src/platedetect.cpp
  include/vitis/ai/nnpp/platenum.hpp
  src/platenum.cpp
  include/vitis/ai/nnpp/facedetect.hpp
  src/facedetect.cpp
  include/vitis/ai/nnpp/facefeature.hpp
  src/facefeature.cpp
  include/vitis/ai/nnpp/classification.hpp
  src/classification/classification.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc
  ${CMAKE_CURRENT_BINARY_DIR}/cifar10_label.inc
  ${CMAKE_CURRENT_BINARY_DIR}/fmnist_label.inc
  include/vitis/ai/nnpp/openpose.hpp
  src/openpose.cpp
  include/vitis/ai/nnpp/hourglass.hpp
  src/hourglass.cpp
  #  include/vitis/ai/nnpp/facequality.hpp
  #  src/facequality.cpp
  include/vitis/ai/nnpp/facequality5pt.hpp
  src/facequality5pt.cpp
  include/vitis/ai/nnpp/facelandmark.hpp
  src/facelandmark.cpp
  #  include/vitis/ai/nnpp/multitask.hpp

  src/multitask/multitask_imp.cpp
  src/multitask/multitask.cpp
  src/multitask/prior_boxes.cpp
  src/multitask/ssd_detector.cpp
  include/vitis/ai/nnpp/yolov2.hpp
  src/yolov2.cpp
  include/vitis/ai/nnpp/yolov3.hpp
  src/yolov3.cpp
  include/vitis/ai/nnpp/posedetect.hpp
  src/posedetect.cpp
  include/vitis/ai/nnpp/reid.hpp
  src/reid.cpp
  include/vitis/ai/nnpp/refinedet.hpp
  src/refinedet/refinedet.cpp
  src/refinedet/refine_det_post.cpp
  src/refinedet/prior_boxes.cpp
  src/refinedet/ssd_detector.cpp
  include/vitis/ai/nnpp/tfrefinedet.hpp
  src/tfrefinedet/tfrefinedet.cpp
  src/tfrefinedet/tfrefinedet_post.cpp
  src/tfrefinedet/ssd_detector.cpp
  include/vitis/ai/nnpp/lanedetect.hpp
  src/roadline/roadline.cpp
  src/roadline/road_line_post.cpp
  src/roadline/predict.cpp
  src/roadline/ipm_info.cpp
  include/vitis/ai/nnpp/tfssd.hpp
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
  include/vitis/ai/nnpp/ssd.hpp
  src/ssd/ssd.cpp
  src/ssd/ssd_post.cpp
  src/ssd/ssd_detector.cpp
  src/ssd/prior_boxes.cpp
  include/vitis/ai/nnpp/medicalsegmentation.hpp
  src/medicalsegmentation/medicalsegmentation.cpp
  src/medicalsegmentation/medicalsegmentation_post.cpp
    include/vitis/ai/nnpp/medicaldetection.hpp
  src/medicaldetection/medicaldetection.cpp
  src/medicaldetection/medicaldetection_post.cpp
  src/medicaldetection/priorbox.cpp
  src/medicaldetection/ssd_detector.cpp
  include/vitis/ai/nnpp/retinaface.hpp
  src/retinaface/retinaface.cpp
  src/retinaface/retinaface_post.cpp
  src/retinaface/retinaface_detector.cpp
  src/retinaface/anchor.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/version.c
  )
add_library(${PROJECT_NAME}::${COMPONENT_NAME} ALIAS ${COMPONENT_NAME})

target_include_directories(${COMPONENT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(${COMPONENT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(${COMPONENT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${COMPONENT_NAME} vart::util
  ${PROJECT_NAME}::math
  ${PROJECT_NAME}::model_config glog::glog
  ${OpenCV_LIBS})
set_target_properties(${COMPONENT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  LIBRARY_OUTPUT_NAME  ${COMPONENT_NAME}-${COMPONENT_NAME}
  )
install(
  TARGETS ${COMPONENT_NAME}
  EXPORT ${COMPONENT_NAME}-targets
  RUNTIME DESTINATION ${INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${INSTALL_LIB_DIR})

install(
  EXPORT ${COMPONENT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${INSTALL_CMAKE_DIR})

install(FILES
  include/vitis/ai/nnpp/classification.hpp
  include/vitis/ai/nnpp/openpose.hpp
  include/vitis/ai/nnpp/multitask.hpp
  include/vitis/ai/nnpp/platedetect.hpp
  include/vitis/ai/nnpp/platenum.hpp
  include/vitis/ai/nnpp/facedetect.hpp
  include/vitis/ai/nnpp/facefeature.hpp
  #include/vitis/ai/nnpp/facequality.hpp
  include/vitis/ai/nnpp/facequality5pt.hpp
  include/vitis/ai/nnpp/facelandmark.hpp
  include/vitis/ai/nnpp/segmentation.hpp
  include/vitis/ai/nnpp/yolov2.hpp
  include/vitis/ai/nnpp/yolov3.hpp
  include/vitis/ai/nnpp/posedetect.hpp
  include/vitis/ai/nnpp/refinedet.hpp
  include/vitis/ai/nnpp/lanedetect.hpp
  include/vitis/ai/nnpp/ssd.hpp
  include/vitis/ai/nnpp/tfssd.hpp
  include/vitis/ai/nnpp/reid.hpp
  include/vitis/ai/nnpp/medicalsegmentation.hpp
  include/vitis/ai/nnpp/medicaldetection.hpp
  include/vitis/ai/nnpp/retinaface.hpp
  include/vitis/ai/nnpp/hourglass.hpp
  DESTINATION include/vitis/ai/nnpp)


install(FILES
  ${LIST_PROTO_DST}
  ${LIST_PBSRC_H}
  DESTINATION include/object_detection/protos)
