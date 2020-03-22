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
option(BUILD_DOXYGEN_DOC "enable doxygen or not" OFF)
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
set(Pthread_LIB )
set(SAMPLE_INATLL_PATH share/${PROJECT_NAME}/samples)
set(DEMO_INATLL_PATH share/${PROJECT_NAME}/demo)

if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
  list(APPEND Pthread_LIB pthread)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(XilinxCommon)
include(XilinxTest)
include(CheckIncludeFileCXX)
include(CheckIncludeFile)
if(BUILD_DOXYGEN_DOC)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
  find_package(Doxygen REQUIRED)
  add_custom_target(doc ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/Logo.png ${CMAKE_BINARY_DIR}/
    COMMAND cp -av ${CMAKE_CURRENT_SOURCE_DIR}/images ${CMAKE_BINARY_DIR}/c-doc/latex
    COMMAND make -C ${CMAKE_BINARY_DIR}/c-doc/latex
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )

#  install(FILES ${CMAKE_BINARY_DIR}/c-doc/latex/refman.pdf DESTINATION share/XILINX_AI_SDK/doc
#    RENAME ug1355-xilinx-ai-sdk-programming-guide.pdf)

endif(BUILD_DOXYGEN_DOC)

find_path(DRM_PATH drm)
if(NOT ${DRM_PATH} EQUAL "DRM_PATH-NOTFOUND")
    set(CMAKE_REQUIRED_INCLUDES ${DRM_PATH}/drm)
endif()

#  check_include_file(xf86drm.h HAVE_DRM)
#  if(HAVE_DRM)
#     message(STATUS "Using DRM.")
#     include_directories(${DRM_PATH}/drm)
#     add_definitions(-DUSE_DRM=1)
#  endif(HAVE_DRM)

check_include_file_cxx(eigen3/Eigen/Dense HAVE_EIGEN)
if(HAVE_EIGEN)
  add_definitions(-DHAVE_EIGEN=1)
endif(HAVE_EIGEN)

XILINX_AI_SDK_TEST_BY_NAME(ssd SSD)
XILINX_AI_SDK_TEST_BY_NAME(yolov3 YOLOv3)
XILINX_AI_SDK_TEST_BY_NAME(yolov2 YOLOv2)
XILINX_AI_SDK_TEST_BY_NAME(facedetect FaceDetect)
XILINX_AI_SDK_TEST_BY_NAME(facelandmark FaceLandmark NO_VIDEO)
XILINX_AI_SDK_TEST_BY_NAME(lanedetect RoadLine)

XILINX_AI_SDK_TEST_BY_NAME(classification Classification)
XILINX_AI_SDK_add_accuracy(classification classification)

#XILINX_AI_SDK_TEST_BY_NAME(segmentation Segmentation8UC1)
XILINX_AI_SDK_TEST_BY_NAME(segmentation Segmentation8UC3)

#XILINX_AI_SDK_TEST_BY_NAME(multitask MultiTask8UC1)
XILINX_AI_SDK_TEST_BY_NAME(multitask MultiTask8UC3)
XILINX_AI_SDK_TEST_BY_NAME(posedetect PoseDetect)


add_executable(test_jpeg_reid  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_jpeg_reid.cpp)
target_link_libraries(test_jpeg_reid  ${OpenCV_LIBS} ${PROJECT_NAME}::reid vart::util glog ${Pthread_LIB})

add_executable(test_accuracy_reid  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_accuracy_reid.cpp)
target_link_libraries(test_accuracy_reid  ${OpenCV_LIBS} ${PROJECT_NAME}::reid vart::util glog ${Pthread_LIB})

add_executable(test_performance_reid  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_performance_reid.cpp)
target_link_libraries(test_performance_reid  ${OpenCV_LIBS} ${PROJECT_NAME}::reid vart::util glog ${Pthread_LIB})
install(TARGETS test_jpeg_reid test_accuracy_reid test_performance_reid DESTINATION ${SAMPLE_INATLL_PATH}/reid/)

#install(DIRECTORY
#  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/images
#  DESTINATION ${SAMPLE_INATLL_PATH}/reid)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_jpeg_reid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_accuracy_reid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_performance_reid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/build.sh
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/readme
  #${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/sample_reid_001.jpg
  #${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/sample_reid_002.jpg
  #${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_performance_reid.list
  DESTINATION ${SAMPLE_INATLL_PATH}/reid/)

add_executable(test_jpeg_posedetect_with_ssd  ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/test_jpeg_posedetect_with_ssd.cpp)
target_link_libraries(test_jpeg_posedetect_with_ssd  ${OpenCV_LIBS} ${PROJECT_NAME}::ssd ${PROJECT_NAME}::posedetect glog ${Pthread_LIB})
if(HAVE_DRM)
  target_link_libraries(test_jpeg_posedetect_with_ssd drm)
endif(HAVE_DRM)

add_executable(test_video_posedetect_with_ssd  ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/test_video_posedetect_with_ssd.cpp)
target_link_libraries(test_video_posedetect_with_ssd  ${OpenCV_LIBS} ${PROJECT_NAME}::ssd ${PROJECT_NAME}::posedetect glog ${Pthread_LIB})
if(HAVE_DRM)
  target_link_libraries(test_video_posedetect_with_ssd drm)
endif(HAVE_DRM)


install(TARGETS test_jpeg_posedetect_with_ssd test_video_posedetect_with_ssd DESTINATION ${SAMPLE_INATLL_PATH}/posedetect/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/test_jpeg_posedetect_with_ssd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/test_video_posedetect_with_ssd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/ssd_posedetect.hpp
  DESTINATION ${SAMPLE_INATLL_PATH}/posedetect/)

XILINX_AI_SDK_TEST_BY_NAME(refinedet RefineDet)
XILINX_AI_SDK_TEST_BY_NAME(openpose OpenPose)

XILINX_AI_SDK_TEST_BY_NAME(tfssd TFSSD)

XILINX_AI_SDK_add_accuracy(tfssd tfssd)

#ssd accuracy
foreach(TMPMODEL ssd_pedestrain_pruned_0_97 ssd_traffic_pruned_0_9 ssd_adas_pruned_0_95 ssd_mobilenet_v2 mlperf_ssd_resnet34_tf)
  XILINX_AI_SDK_add_accuracy(ssd ${TMPMODEL})
endforeach(TMPMODEL)

#openpose accuracy
XILINX_AI_SDK_add_accuracy(openpose openpose)
XILINX_AI_SDK_add_accuracy(multitask multitask)

#yolov2 accuracy

XILINX_AI_SDK_add_accuracy(yolov2 yolov2)


#yolov3 accuracy
foreach(TMPMODEL yolov3_adas_pruned_0_9 yolov3_voc yolov3_voc_tf yolov3_bdd)
  XILINX_AI_SDK_add_accuracy(yolov3 ${TMPMODEL})
endforeach(TMPMODEL)

#refinedet accuracy
XILINX_AI_SDK_add_accuracy(refinedet refinedet)
XILINX_AI_SDK_add_accuracy(facedetect facedetect)
XILINX_AI_SDK_add_accuracy(posedetect posedetect)
XILINX_AI_SDK_add_accuracy(segmentation segmentation)
XILINX_AI_SDK_add_accuracy(lanedetect lanedetect)
XILINX_AI_SDK_add_accuracy(facelandmark facelandmark)

add_executable(seg_and_pose_detect ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/seg_and_pose_detect.cpp)
target_link_libraries(seg_and_pose_detect ${OpenCV_LIBS} ${PROJECT_NAME}::multitask ${PROJECT_NAME}::posedetect ${PROJECT_NAME}::ssd glog ${Pthread_LIB})
if(HAVE_DRM)
  target_link_libraries(seg_and_pose_detect drm)
endif(HAVE_DRM)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/ DESTINATION ${DEMO_INATLL_PATH}/seg_and_pose_detect/)

add_executable(segs_and_roadline_detect ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/segs_and_roadline_detect.cpp)
target_link_libraries(segs_and_roadline_detect ${OpenCV_LIBS} ${PROJECT_NAME}::multitask ${PROJECT_NAME}::lanedetect ${PROJECT_NAME}::ssd glog ${Pthread_LIB})
if(HAVE_DRM)
  target_link_libraries(segs_and_roadline_detect drm)
endif(HAVE_DRM)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/ DESTINATION ${DEMO_INATLL_PATH}/segs_and_roadline_detect/)


if(HAVE_DRM)
  target_link_libraries(segs_and_roadline_detect drm)
endif(HAVE_DRM)

# demos
#compile failed
find_package(xir REQUIRED)
find_package(unilog REQUIRED)
find_package(vart REQUIRED COMPONENTS util runner xrt-device-handle dpu-controller dpu-runner)
add_executable(demo_classification ${CMAKE_CURRENT_SOURCE_DIR}/demo/classification/demo_classification.cpp)
target_link_libraries(demo_classification ${OpenCV_LIBS} glog ${PROJECT_NAME}::model_config ${PROJECT_NAME}::dpu_task vart::dpu-runner)

###install(TARGETS demo_classification DESTINATION ${DEMO_INATLL_PATH}/classification/)
###install(FILES
###  demo/classification/build.sh
###  demo/classification/demo_classification.cpp
###  demo/classification/readme
###  demo/classification/demo_classification.jpg
###  DESTINATION ${DEMO_INATLL_PATH}/classification/)

add_executable(demo_yolov3 ${CMAKE_CURRENT_SOURCE_DIR}/demo/yolov3/demo_yolov3.cpp)
target_link_libraries(demo_yolov3 ${OpenCV_LIBS} glog protobuf::libprotobuf ${PROJECT_NAME}::model_config ${PROJECT_NAME}::dpu_task xnnpp)
install(TARGETS demo_yolov3 DESTINATION ${DEMO_INATLL_PATH}/yolov3/)
install(FILES
  demo/yolov3/build.sh
  demo/yolov3/readme
  #demo/yolov3/demo_yolov3.jpg
  demo/yolov3/demo_yolov3.cpp
  DESTINATION ${DEMO_INATLL_PATH}/yolov3/)


add_executable(test_jpeg_classification_squeezenet ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_jpeg_classification_squeezenet.cpp)
add_executable(test_performance_classification_squeezenet ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_performance_classification_squeezenet.cpp)
add_executable(test_accuracy_classification_squeezenet ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_accuracy_classification_squeezenet.cpp)
target_link_libraries(test_jpeg_classification_squeezenet ${OpenCV_LIBS} glog ${PROJECT_NAME}::model_config ${PROJECT_NAME}::dpu_task ${PROJECT_NAME}::math)
target_link_libraries(test_performance_classification_squeezenet ${OpenCV_LIBS} glog ${PROJECT_NAME}::model_config ${PROJECT_NAME}::dpu_task ${PROJECT_NAME}::math vart::util ${Pthread_LIB})
target_link_libraries(test_accuracy_classification_squeezenet ${OpenCV_LIBS} glog ${PROJECT_NAME}::model_config ${PROJECT_NAME}::dpu_task ${PROJECT_NAME}::math)
install(TARGETS test_jpeg_classification_squeezenet DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(TARGETS test_performance_classification_squeezenet DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(TARGETS test_accuracy_classification_squeezenet DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(FILES
  samples/classification/test_jpeg_classification_squeezenet.cpp
  samples/classification/test_performance_classification_squeezenet.cpp
  samples/classification/test_accuracy_classification_squeezenet.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
