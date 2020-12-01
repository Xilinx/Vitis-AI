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
XILINX_AI_SDK_TEST_BY_NAME(platedetect PlateDetect)
XILINX_AI_SDK_TEST_BY_NAME(platenum PlateNum)
XILINX_AI_SDK_TEST_BY_NAME(facelandmark FaceLandmark NO_VIDEO)
XILINX_AI_SDK_TEST_BY_NAME(facefeature FaceFeature NO_VIDEO)
XILINX_AI_SDK_TEST_BY_NAME(facequality5pt FaceQuality5pt NO_VIDEO)
XILINX_AI_SDK_TEST_BY_NAME(lanedetect RoadLine)
XILINX_AI_SDK_TEST_BY_NAME(retinaface RetinaFace)

XILINX_AI_SDK_TEST_BY_NAME(classification Classification)
XILINX_AI_SDK_add_accuracy(classification classification)
add_executable(test_accuracy_classification_mt  ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_accuracy_classification_mt.cpp)
target_link_libraries(test_accuracy_classification_mt  ${OpenCV_LIBS} ${PROJECT_NAME}::classification vart::util glog::glog ${Pthread_LIB})
install(TARGETS test_accuracy_classification_mt DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_accuracy_classification_mt.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/classification/)

#XILINX_AI_SDK_TEST_BY_NAME(segmentation Segmentation8UC1)
XILINX_AI_SDK_TEST_BY_NAME(segmentation Segmentation8UC3)

#XILINX_AI_SDK_TEST_BY_NAME(multitask MultiTask8UC1)
XILINX_AI_SDK_TEST_BY_NAME(multitask MultiTask8UC3)
XILINX_AI_SDK_TEST_BY_NAME(posedetect PoseDetect)


add_executable(test_jpeg_reid  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_jpeg_reid.cpp)
target_link_libraries(test_jpeg_reid  ${OpenCV_LIBS} ${PROJECT_NAME}::reid vart::util glog::glog ${Pthread_LIB})

add_executable(test_accuracy_reid  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_accuracy_reid.cpp)
target_link_libraries(test_accuracy_reid  ${OpenCV_LIBS} ${PROJECT_NAME}::reid vart::util glog::glog ${Pthread_LIB})

add_executable(test_performance_reid  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_performance_reid.cpp)
target_link_libraries(test_performance_reid  ${OpenCV_LIBS} ${PROJECT_NAME}::reid vart::util glog::glog ${Pthread_LIB})
install(TARGETS test_jpeg_reid test_accuracy_reid test_performance_reid DESTINATION ${SAMPLE_INATLL_PATH}/reid/)

#install(DIRECTORY
#  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/images
#  DESTINATION ${SAMPLE_INATLL_PATH}/reid)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_jpeg_reid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_accuracy_reid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_performance_reid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/readme
  #${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/sample_reid_001.jpg
  #${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/sample_reid_002.jpg
  #${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/test_performance_reid.list
  DESTINATION ${SAMPLE_INATLL_PATH}/reid/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/reid/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${SAMPLE_INATLL_PATH}/reid/)

add_executable(test_jpeg_posedetect_with_ssd  ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/test_jpeg_posedetect_with_ssd.cpp)
target_link_libraries(test_jpeg_posedetect_with_ssd  ${OpenCV_LIBS} ${PROJECT_NAME}::ssd ${PROJECT_NAME}::posedetect glog::glog ${Pthread_LIB})
if(HAVE_DRM)
  target_link_libraries(test_jpeg_posedetect_with_ssd drm)
endif(HAVE_DRM)

add_executable(test_video_posedetect_with_ssd  ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/test_video_posedetect_with_ssd.cpp)
target_link_libraries(test_video_posedetect_with_ssd  ${OpenCV_LIBS} ${PROJECT_NAME}::ssd ${PROJECT_NAME}::posedetect glog::glog ${Pthread_LIB})
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
add_executable(test_jpeg_refinedet_tf ${CMAKE_CURRENT_SOURCE_DIR}/samples/refinedet/test_jpeg_refinedet_tf.cpp)
target_link_libraries(test_jpeg_refinedet_tf ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::refinedet -lglog protobuf::libprotobuf)
install(TARGETS test_jpeg_refinedet_tf DESTINATION ${SAMPLE_INATLL_PATH}/refinedet/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/samples/refinedet/test_jpeg_refinedet_tf.cpp DESTINATION ${SAMPLE_INATLL_PATH}/refinedet/)


XILINX_AI_SDK_TEST_BY_NAME(openpose OpenPose)
XILINX_AI_SDK_TEST_BY_NAME(hourglass Hourglass)
XILINX_AI_SDK_add_accuracy(hourglass hourglass)

XILINX_AI_SDK_TEST_BY_NAME(tfssd TFSSD)

XILINX_AI_SDK_add_accuracy(tfssd tfssd)

add_executable(test_accuracy_tfssd_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/tfssd/test_accuracy_tfssd_mt.cpp)
target_link_libraries(test_accuracy_tfssd_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::tfssd -lpthread -lglog json-c protobuf::libprotobuf)
install(TARGETS test_accuracy_tfssd_mt DESTINATION ${SAMPLE_INATLL_PATH}/tfssd/)


XILINX_AI_SDK_TEST_BY_NAME(medicalsegmentation MedicalSegmentation NO_VIDEO)
XILINX_AI_SDK_add_accuracy( medicalsegmentation medicalsegmentation )
XILINX_AI_SDK_TEST_BY_NAME(medicaldetection MedicalDetection )
XILINX_AI_SDK_add_accuracy( medicaldetection medicaldetection )

add_executable(test_accuracy_medicaldetection_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicaldetection/test_accuracy_medicaldetection_mt.cpp)
target_link_libraries(test_accuracy_medicaldetection_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::medicaldetection -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_medicaldetection_mt DESTINATION ${SAMPLE_INATLL_PATH}/medicaldetection/)


#ssd accuracy
foreach(TMPMODEL ssd_pedestrian_pruned_0_97 ssd_traffic_pruned_0_9 ssd_adas_pruned_0_95 ssd_mobilenet_v2 mlperf_ssd_resnet34_tf)
  XILINX_AI_SDK_add_accuracy(ssd ${TMPMODEL})
endforeach(TMPMODEL)

add_executable(test_accuracy_ssd_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/ssd/test_accuracy_ssd_mt.cpp)
target_link_libraries(test_accuracy_ssd_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::ssd -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_ssd_mt DESTINATION ${SAMPLE_INATLL_PATH}/ssd/)

#openpose accuracy
XILINX_AI_SDK_add_accuracy(openpose openpose)
XILINX_AI_SDK_add_accuracy(multitask multitask)
add_executable(test_accuracy_multitask_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/multitask/test_accuracy_multitask_mt.cpp)
target_link_libraries(test_accuracy_multitask_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::multitask -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_multitask_mt DESTINATION ${SAMPLE_INATLL_PATH}/multitask/)
#yolov2 accuracy
XILINX_AI_SDK_add_accuracy(yolov2 yolov2)
add_executable(test_accuracy_yolov2_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/yolov2/test_accuracy_yolov2_mt.cpp)
target_link_libraries(test_accuracy_yolov2_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::yolov2 -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_yolov2_mt DESTINATION ${SAMPLE_INATLL_PATH}/yolov2/)

#yolov3 accuracy
foreach(TMPMODEL yolov3_adas_pruned_0_9 yolov3_voc yolov3_voc_tf yolov3_bdd)
  XILINX_AI_SDK_add_accuracy(yolov3 ${TMPMODEL})
endforeach(TMPMODEL)

add_executable(test_accuracy_yolov3_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/yolov3/test_accuracy_yolov3_mt.cpp)
target_link_libraries(test_accuracy_yolov3_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::yolov3 -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_yolov3_mt DESTINATION ${SAMPLE_INATLL_PATH}/yolov3/)

add_executable(test_accuracy_tiny_yolov3_vmss ${CMAKE_CURRENT_SOURCE_DIR}/samples/yolov3/test_accuracy_tiny_yolov3_vmss.cpp)
target_link_libraries(test_accuracy_tiny_yolov3_vmss ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::yolov3 -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_tiny_yolov3_vmss DESTINATION ${SAMPLE_INATLL_PATH}/yolov3/)

add_executable(test_jpeg_yolov4 ${CMAKE_CURRENT_SOURCE_DIR}/samples/yolov3/test_jpeg_yolov4.cpp)
target_link_libraries(test_jpeg_yolov4 ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::yolov3 -lglog protobuf::libprotobuf)
add_executable(test_accuracy_yolov4 ${CMAKE_CURRENT_SOURCE_DIR}/samples/yolov3/test_accuracy_yolov4_mt.cpp)
target_link_libraries(test_accuracy_yolov4 ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::yolov3 -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_jpeg_yolov4 test_accuracy_yolov4 DESTINATION ${SAMPLE_INATLL_PATH}/yolov3/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/yolov3/test_jpeg_yolov4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/yolov3/test_accuracy_yolov4_mt.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/yolov3/)


#refinedet accuracy
XILINX_AI_SDK_add_accuracy(refinedet refinedet)

add_executable(test_accuracy_refinedet_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/refinedet/test_accuracy_refinedet_mt.cpp)
target_link_libraries(test_accuracy_refinedet_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::refinedet -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_refinedet_mt DESTINATION ${SAMPLE_INATLL_PATH}/refinedet/)

#platenum accuracy
XILINX_AI_SDK_add_accuracy(platenum platenum)

add_executable(test_accuracy_platenum_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/platenum/test_accuracy_platenum_mt.cpp)
target_link_libraries(test_accuracy_platenum_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::platenum -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_platenum_mt DESTINATION ${SAMPLE_INATLL_PATH}/platenum/)

#platedetect accuracy
XILINX_AI_SDK_add_accuracy(platedetect platedetect)

add_executable(test_accuracy_platedetect_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/platedetect/test_accuracy_platedetect_mt.cpp)
target_link_libraries(test_accuracy_platedetect_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::platedetect -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_platedetect_mt DESTINATION ${SAMPLE_INATLL_PATH}/platedetect/)

#segmentation accuracy
XILINX_AI_SDK_add_accuracy(segmentation segmentation)

add_executable(test_accuracy_segmentation_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/segmentation/test_accuracy_segmentation_mt.cpp)
target_link_libraries(test_accuracy_segmentation_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::segmentation -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_segmentation_mt DESTINATION ${SAMPLE_INATLL_PATH}/segmentation/)

add_executable(test_accuracy_covid19segmentation ${CMAKE_CURRENT_SOURCE_DIR}/samples/covid19segmentation/test_accuracy_covid19segmentation.cpp)
target_link_libraries(test_accuracy_covid19segmentation ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::covid19segmentation -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_covid19segmentation DESTINATION ${SAMPLE_INATLL_PATH}/covid19segmentation/)

add_executable(test_accuracy_endov ${CMAKE_CURRENT_SOURCE_DIR}/samples/segmentation/test_accuracy_endov.cpp)
target_link_libraries(test_accuracy_endov ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::segmentation -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_endov DESTINATION ${SAMPLE_INATLL_PATH}/segmentation/)

add_executable(test_accuracy_unet ${CMAKE_CURRENT_SOURCE_DIR}/samples/segmentation/test_accuracy_unet.cpp)
target_link_libraries(test_accuracy_unet ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::segmentation -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_unet DESTINATION ${SAMPLE_INATLL_PATH}/segmentation/)

#posedetect accuracy
XILINX_AI_SDK_add_accuracy(posedetect posedetect)

add_executable(test_accuracy_posedetect_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/posedetect/test_accuracy_posedetect_mt.cpp)
target_link_libraries(test_accuracy_posedetect_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::posedetect -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_posedetect_mt DESTINATION ${SAMPLE_INATLL_PATH}/posedetect/)

XILINX_AI_SDK_add_accuracy(facedetect facedetect)
add_executable(test_accuracy_facedetect_mt  ${CMAKE_CURRENT_SOURCE_DIR}/samples/facedetect/test_accuracy_facedetect_mt.cpp)
target_link_libraries(test_accuracy_facedetect_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::facedetect -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_facedetect_mt DESTINATION ${SAMPLE_INATLL_PATH}/facedetect/)


XILINX_AI_SDK_add_accuracy(facefeature facefeature)
XILINX_AI_SDK_add_accuracy(lanedetect lanedetect)
XILINX_AI_SDK_add_accuracy(facelandmark facelandmark)
add_executable(test_accuracy_facelandmark_mt  ${CMAKE_CURRENT_SOURCE_DIR}/samples/facelandmark/test_accuracy_facelandmark_mt.cpp)
target_link_libraries(test_accuracy_facelandmark_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::facelandmark -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_facelandmark_mt DESTINATION ${SAMPLE_INATLL_PATH}/facelandmark/)


XILINX_AI_SDK_add_accuracy(facequality5pt facequality5pt)
XILINX_AI_SDK_add_accuracy(retinaface retinaface)

add_executable(test_accuracy_facequality5pt_mt ${CMAKE_CURRENT_SOURCE_DIR}/samples/facequality5pt/test_accuracy_facequality5pt_mt.cpp)
target_link_libraries(test_accuracy_facequality5pt_mt ${COMPONENT_NAME} ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::facequality5pt -lpthread -lglog protobuf::libprotobuf)
install(TARGETS test_accuracy_facequality5pt_mt DESTINATION ${SAMPLE_INATLL_PATH}/facequality5pt/)


add_executable(seg_and_pose_detect ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/seg_and_pose_detect.cpp)
target_link_libraries(seg_and_pose_detect ${OpenCV_LIBS} ${PROJECT_NAME}::multitask ${PROJECT_NAME}::posedetect ${PROJECT_NAME}::ssd glog::glog ${Pthread_LIB})
if(HAVE_DRM)
  target_link_libraries(seg_and_pose_detect drm)
endif(HAVE_DRM)
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/ DESTINATION ${DEMO_INATLL_PATH}/seg_and_pose_detect/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/readme
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/seg_and_pose_detect.cpp
  DESTINATION ${DEMO_INATLL_PATH}/seg_and_pose_detect/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/build.sh
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/seg_and_pose_detect/builddrm.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${DEMO_INATLL_PATH}/seg_and_pose_detect/)

add_executable(segs_and_roadline_detect ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/segs_and_roadline_detect.cpp)
target_link_libraries(segs_and_roadline_detect ${OpenCV_LIBS} ${PROJECT_NAME}::multitask ${PROJECT_NAME}::lanedetect ${PROJECT_NAME}::ssd glog::glog ${Pthread_LIB})
if(HAVE_DRM)
  target_link_libraries(segs_and_roadline_detect drm)
endif(HAVE_DRM)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/ DESTINATION ${DEMO_INATLL_PATH}/segs_and_roadline_detect/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/readme
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/segs_and_roadline_detect.cpp
  DESTINATION  ${DEMO_INATLL_PATH}/segs_and_roadline_detect/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/build.sh
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/segs_and_roadline_detect/builddrm.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${DEMO_INATLL_PATH}/segs_and_roadline_detect/)


if(HAVE_DRM)
  target_link_libraries(segs_and_roadline_detect drm)
endif(HAVE_DRM)

# demos
#compile failed
##find_package(xir REQUIRED)
##find_package(unilog REQUIRED)
##find_package(vart REQUIRED COMPONENTS util runner xrt-device-handle dpu-controller dpu-runner)
add_executable(demo_classification ${CMAKE_CURRENT_SOURCE_DIR}/demo/classification/demo_classification.cpp)
target_link_libraries(demo_classification ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::model_config ${PROJECT_NAME}::dpu_task vart::dpu-runner)

install(TARGETS demo_classification DESTINATION ${DEMO_INATLL_PATH}/classification/)
install(FILES
  demo/classification/demo_classification.cpp
  demo/classification/readme
  DESTINATION ${DEMO_INATLL_PATH}/classification/)
install(FILES
  demo/classification/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${DEMO_INATLL_PATH}/classification/)

add_executable(demo_yolov3 ${CMAKE_CURRENT_SOURCE_DIR}/demo/yolov3/demo_yolov3.cpp)
target_link_libraries(demo_yolov3 ${OpenCV_LIBS} glog::glog protobuf::libprotobuf ${PROJECT_NAME}::model_config ${PROJECT_NAME}::dpu_task xnnpp)
install(TARGETS demo_yolov3 DESTINATION ${DEMO_INATLL_PATH}/yolov3/)
install(FILES
  demo/yolov3/readme
  #demo/yolov3/demo_yolov3.jpg
  demo/yolov3/demo_yolov3.cpp
  DESTINATION ${DEMO_INATLL_PATH}/yolov3/)
install(FILES
  demo/yolov3/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${DEMO_INATLL_PATH}/yolov3/)

add_executable(test_jpeg_covid19segmentation ${CMAKE_CURRENT_SOURCE_DIR}/samples/covid19segmentation/test_jpeg_covid19segmentation.cpp)
target_link_libraries(test_jpeg_covid19segmentation ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::covid19segmentation)
install(TARGETS test_jpeg_covid19segmentation DESTINATION ${SAMPLE_INATLL_PATH}/covid19segmentation/)
add_executable(test_performance_covid19segmentation ${CMAKE_CURRENT_SOURCE_DIR}/samples/covid19segmentation/test_performance_covid19segmentation.cpp)
target_link_libraries(test_performance_covid19segmentation ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::covid19segmentation)
install(TARGETS test_performance_covid19segmentation DESTINATION ${SAMPLE_INATLL_PATH}/covid19segmentation/)
install(FILES
  samples/covid19segmentation/readme
  samples/covid19segmentation/test_jpeg_covid19segmentation.cpp
  samples/covid19segmentation/test_performance_covid19segmentation.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/covid19segmentation/)
install(FILES
  samples/covid19segmentation/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${SAMPLE_INATLL_PATH}/covid19segmentation/)

add_executable(test_jpeg_3Dsegmentation ${CMAKE_CURRENT_SOURCE_DIR}/samples/3Dsegmentation/test_jpeg_3Dsegmentation.cpp)
target_link_libraries(test_jpeg_3Dsegmentation ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::3Dsegmentation)
install(TARGETS test_jpeg_3Dsegmentation DESTINATION ${SAMPLE_INATLL_PATH}/3Dsegmentation/)
add_EXecutable(test_performance_3Dsegmentation ${CMAKE_CURRENT_SOURCE_DIR}/samples/3Dsegmentation/test_performance_3Dsegmentation.cpp)
target_link_libraries(test_performance_3Dsegmentation ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::3Dsegmentation)
install(TARGETS test_performance_3Dsegmentation DESTINATION ${SAMPLE_INATLL_PATH}/3Dsegmentation/)
install(FILES
  samples/3Dsegmentation/readme
  samples/3Dsegmentation/test_jpeg_3Dsegmentation.cpp
  samples/3Dsegmentation/test_performance_3Dsegmentation.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/3Dsegmentation/)
install(FILES
  samples/3Dsegmentation/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${SAMPLE_INATLL_PATH}/3Dsegmentation/)

add_executable(test_jpeg_classification_squeezenet ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_jpeg_classification_squeezenet.cpp)
add_executable(test_jpeg_classification_cifar ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_jpeg_classification_cifar.cpp)
add_executable(test_performance_classification_squeezenet ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_performance_classification_squeezenet.cpp)
add_executable(test_accuracy_classification_squeezenet ${CMAKE_CURRENT_SOURCE_DIR}/samples/classification/test_accuracy_classification_squeezenet.cpp)
target_link_libraries(test_jpeg_classification_squeezenet ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::classification)
target_link_libraries(test_jpeg_classification_cifar ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::classification)
target_link_libraries(test_performance_classification_squeezenet ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::classification vart::util ${Pthread_LIB})
target_link_libraries(test_accuracy_classification_squeezenet ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::model_config ${PROJECT_NAME}::classification)
install(TARGETS test_jpeg_classification_squeezenet DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(TARGETS test_jpeg_classification_cifar DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(TARGETS test_performance_classification_squeezenet DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(TARGETS test_accuracy_classification_squeezenet DESTINATION ${SAMPLE_INATLL_PATH}/classification/)
install(FILES
  samples/classification/find_model.hpp
  samples/classification/test_jpeg_classification_squeezenet.cpp
  samples/classification/test_jpeg_classification_cifar.cpp
  samples/classification/test_performance_classification_squeezenet.cpp
  samples/classification/test_accuracy_classification_squeezenet.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/classification/)

#facefeature : model facerec_pretrain_res20
add_executable(test_accuracy_facerec_pretrain_res20 samples/facefeature/test_accuracy_facerec_pretrain_res20.cpp)
target_link_libraries(test_accuracy_facerec_pretrain_res20 ${OpenCV_LIBS} ${PROJECT_NAME}::facefeature)
install(FILES
  samples/facefeature/test_accuracy_facerec_pretrain_res20.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/facefeature/)
install(TARGETS test_accuracy_facerec_pretrain_res20 DESTINATION ${SAMPLE_INATLL_PATH}/facefeature/)

#pointpillars
add_executable(test_bin_pointpillars ${CMAKE_CURRENT_SOURCE_DIR}/samples/pointpillars/test_bin_pointpillars.cpp)
add_executable(test_performance_pointpillars ${CMAKE_CURRENT_SOURCE_DIR}/samples/pointpillars/test_performance_pointpillars.cpp)
add_executable(test_accuracy_pointpillars ${CMAKE_CURRENT_SOURCE_DIR}/samples/pointpillars/test_accuracy_pointpillars.cpp)
target_link_libraries(test_bin_pointpillars ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::pointpillars)
target_link_libraries(test_performance_pointpillars ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::pointpillars vart::util ${Pthread_LIB})
target_link_libraries(test_accuracy_pointpillars ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::model_config ${PROJECT_NAME}::pointpillars)
install(TARGETS test_bin_pointpillars DESTINATION ${SAMPLE_INATLL_PATH}/pointpillars/)
install(TARGETS test_performance_pointpillars DESTINATION ${SAMPLE_INATLL_PATH}/pointpillars/)
install(TARGETS test_accuracy_pointpillars DESTINATION ${SAMPLE_INATLL_PATH}/pointpillars/)
install(FILES
  samples/pointpillars/readme
  samples/pointpillars/test_bin_pointpillars.cpp
  samples/pointpillars/test_performance_pointpillars.cpp
  samples/pointpillars/test_accuracy_pointpillars.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/pointpillars/)
install(FILES
  samples/pointpillars/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${SAMPLE_INATLL_PATH}/pointpillars/)

#cifar10classification
add_executable(test_jpeg_cifar10classification ${CMAKE_CURRENT_SOURCE_DIR}/samples/cifar10classification/test_jpeg_cifar10classification.cpp)
add_executable(test_performance_cifar10classification ${CMAKE_CURRENT_SOURCE_DIR}/samples/cifar10classification/test_performance_cifar10classification.cpp)
target_link_libraries(test_jpeg_cifar10classification ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::cifar10classification)
target_link_libraries(test_performance_cifar10classification ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::cifar10classification vart::util ${Pthread_LIB})
install(TARGETS test_jpeg_cifar10classification DESTINATION ${SAMPLE_INATLL_PATH}/cifar10classification/)
install(TARGETS test_performance_cifar10classification DESTINATION ${SAMPLE_INATLL_PATH}/cifar10classification/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/cifar10classification/readme
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/cifar10classification/process_result.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/cifar10classification/test_jpeg_cifar10classification.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/cifar10classification/test_performance_cifar10classification.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/cifar10classification/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/cifar10classification/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${SAMPLE_INATLL_PATH}/cifar10classification/)

#mnistclassification
add_executable(test_jpeg_mnistclassification ${CMAKE_CURRENT_SOURCE_DIR}/samples/mnistclassification/test_jpeg_mnistclassification.cpp)
add_executable(test_performance_mnistclassification ${CMAKE_CURRENT_SOURCE_DIR}/samples/mnistclassification/test_performance_mnistclassification.cpp)
target_link_libraries(test_jpeg_mnistclassification ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::mnistclassification)
target_link_libraries(test_performance_mnistclassification ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::mnistclassification vart::util ${Pthread_LIB})
install(TARGETS test_jpeg_mnistclassification DESTINATION ${SAMPLE_INATLL_PATH}/mnistclassification/)
install(TARGETS test_performance_mnistclassification DESTINATION ${SAMPLE_INATLL_PATH}/mnistclassification/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/mnistclassification/readme
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/mnistclassification/test_jpeg_mnistclassification.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/mnistclassification/test_performance_mnistclassification.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/mnistclassification/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/mnistclassification/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${SAMPLE_INATLL_PATH}/mnistclassification/)

#medicalsegcell
add_executable(test_jpeg_medicalsegcell  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_jpeg_medicalsegcell.cpp)
add_executable(test_video_medicalsegcell ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_video_medicalsegcell.cpp)
add_executable(test_performance_medicalsegcell ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_performance_medicalsegcell.cpp)
add_executable(test_accuracy_medicalsegcell ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_accuracy_medicalsegcell.cpp)
target_link_libraries(test_jpeg_medicalsegcell ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::medicalsegcell)
target_link_libraries(test_video_medicalsegcell ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::medicalsegcell)
target_link_libraries(test_performance_medicalsegcell ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::medicalsegcell vart::util ${Pthread_LIB})
target_link_libraries(test_accuracy_medicalsegcell ${OpenCV_LIBS} glog::glog ${PROJECT_NAME}::medicalsegcell vart::util ${Pthread_LIB})
install(TARGETS test_jpeg_medicalsegcell DESTINATION ${SAMPLE_INATLL_PATH}/medicalsegcell/)
install(TARGETS test_video_medicalsegcell DESTINATION ${SAMPLE_INATLL_PATH}/medicalsegcell/)
install(TARGETS test_performance_medicalsegcell DESTINATION ${SAMPLE_INATLL_PATH}/medicalsegcell/)
install(TARGETS test_accuracy_medicalsegcell DESTINATION ${SAMPLE_INATLL_PATH}/medicalsegcell/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/readme
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/process_result.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_jpeg_medicalsegcell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_video_medicalsegcell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_performance_medicalsegcell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/test_accuracy_medicalsegcell.cpp
  DESTINATION ${SAMPLE_INATLL_PATH}/medicalsegcell/)
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/samples/medicalsegcell/build.sh
  PERMISSIONS OWNER_READ  OWNER_WRITE  OWNER_EXECUTE  GROUP_READ  GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE 
  DESTINATION ${SAMPLE_INATLL_PATH}/medicalsegcell/)

