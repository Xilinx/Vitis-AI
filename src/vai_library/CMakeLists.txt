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

cmake_minimum_required(VERSION 3.5)

if(DEFINED ENV{BUILD_VER})
  project(
    vitis_ai_library
    VERSION $ENV{BUILD_VER}
    LANGUAGES C CXX)
else()
  project(
    vitis_ai_library
    VERSION 3.0.0
    LANGUAGES C CXX)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/XilinxCommon.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/vitis_ai_library_common.cmake)

# options
option(DISABLE_ALL "disable all libraries" OFF)
option(ENABLE_CLASSIFICATION "enable classification" OFF)
option(ENABLE_VEHICLECLASSIFICATION "enable vehicleclassification" OFF)
option(ENABLE_TFSSD "enable tfssd" OFF)
option(ENABLE_SSD "enable ssd" OFF)
option(ENABLE_FACEDETECT "enable face detection" OFF)
option(ENABLE_FACERECOG "enable face recog" OFF)
option(ENABLE_FACEDETECTRECOG "enable face detect recog" OFF)
option(ENABLE_FACEFEATURE "enable face feature" OFF)
option(ENABLE_FACEQUALITY5PT "enable face quality 5pt" OFF)
option(ENABLE_LANEDETECT "enable lane detection" OFF)
option(ENABLE_YOLOV2 "enable Yolov2" OFF)
option(ENABLE_YOLOV3 "enable Yolov3" OFF)
option(ENABLE_OFAYOLO "enable OFA Yolo" OFF)
option(ENABLE_YOLOVX "enable YolovX" OFF)
option(ENABLE_YOLOV6 "enable Yolov6" OFF)
option(ENABLE_PMRID "enable pmrid" OFF)
option(ENABLE_ARFLOW "enable ARFlow" OFF)
option(ENABLE_SOLO "enable Solo" OFF)
option(ENABLE_C2D2_LITE "enable c2d2_lite" OFF)
option(ENABLE_BEVDET "enable bevdet" OFF)
option(ENABLE_RGBDSEGMENTATION "enable RGBDsegmentation" OFF)
option(ENABLE_POLYPSEGMENTATION "enable polypsegmentation" OFF)
option(ENABLE_LANDMARK "enable landmark face-5pt" OFF)
option(ENABLE_SEGMENTATION "enable segmentation" OFF)
option(ENABLE_COVID19SEGMENTATION "enable covid19segmentation" OFF)
option(ENABLE_3DSEGMENTATION "enable 3dsegmentation" OFF)
option(ENABLE_REFINEDET "enable refinedet" OFF)
option(ENABLE_OPENPOSE "enable openpose" OFF)
option(ENABLE_HOURGLASS "enable hourglass" OFF)
option(ENABLE_MOVENET "enable movenet" OFF)
option(ENABLE_POSEDETECT "enable pose detect" OFF)
option(ENABLE_REID "enable reid" OFF)
option(ENABLE_RCAN "enable rcan" OFF)
option(ENABLE_SOLO "enable solo" OFF)
option(ENABLE_FAIRMOT "enable fairmot" OFF)
option(ENABLE_REIDTRACKER "enable reidtracker" OFF)
option(ENABLE_MULTITASK "enable multitask" OFF)
option(ENABLE_MULTITASKV3 "enable multitaskv3" OFF)
option(ENABLE_PLATEDETECT "enable platedetect" OFF)
option(ENABLE_PLATENUM "enable platenum" OFF)
option(ENABLE_PLATERECOG "enable platerecog" OFF)
option(ENABLE_CARPLATERECOG "enable carplaterecog" OFF)
option(ENABLE_PSMNET "enable psmnet" OFF)
if(IS_DIRECTORY overview)
  option(ENABLE_OVERVIEW "enable overview" ON)
else(DIRECTORY overview)
  option(ENABLE_OVERVIEW "enable overview" OFF)
endif(IS_DIRECTORY overview)
message(STATUS "ENABLE_OVERVIEW=${ENABLE_OVERVIEW}")
option(ENABLE_GENERAL "enable general" OFF)
option(ENABLE_MEDICALSEGMENTATION "enable medicalsegmentation" OFF)
option(ENABLE_MEDICALDETECTION "enable medicaldetection" OFF)
option(ENABLE_POINTPILLARS "enable pointpillars" OFF)
option(ENABLE_POINTPILLARS_NUSCENES "enable pointpillars-nuscenes" OFF)
option(ENABLE_POINTPAINTING "enable pointpainting" OFF)
option(ENABLE_CENTERPOINTS "enable centerpoints" OFF)
option(ENABLE_RETINAFACE "enable retinaface" OFF)
option(ENABLE_MNISTCLASSIFICATION "enable mnistclassification" OFF)
option(ENABLE_MEDICALSEGCELL "enable medicalsegcell" OFF)
option(ENABLE_BCC "enable bcc" OFF)
option(ENABLE_PMG "enable pmg" OFF)
option(ENABLE_ULTRAFAST "enable ultrafast" OFF)
option(ENABLE_OCR "enable ocr" OFF)
option(ENABLE_TEXTMOUNTAIN "enable textmountain" ON)
option(ENABLE_BTS "enable bts" OFF)
option(ENABLE_FUSION_CNN "enable fusion_cnn " OFF)
option(ENABLE_CLOCS "enable clocs " OFF)
option(ENABLE_X_AUTONOMOUS3D "enable x_autonomous3d" OFF)
option(ENABLE_EFFICIENTDET_D2 "enable efficientdet_d2 " ON)
option(ENABLE_DPUCADX8G_RUNNER "enable DPUV1 Runner" OFF)
option(ENABLE_DPUCADF8H_RUNNER "enable DPUV3INT8 Runner" OFF)
option(ENABLE_BOSCH_FCNSEMSEG "enable bosch_fcnsemseg" OFF)
option(ENABLE_BOSCH_SSDOBJDET "enable bosch_ssdobjdet" OFF)
option(ENABLE_MONODEPTH2 "enable monodepth2" OFF)
option(ENABLE_CFLOWNET "enable cflownet" OFF)
option(ENABLE_CENTERNET "enable centernet" OFF)

# variables
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}/cmake")
set(INSTALL_LIB_DIR lib)
set(INSTALL_BIN_DIR bin)
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_CMAKE_DIR share/cmake/${PROJECT_NAME})

if(ENABLE_DPUCADX8G_RUNNER)
  # Enble supported models for DPUV1
  unset(DISABLE_ALL CACHE)
  unset(ENABLE_CLASSIFICATION CACHE)
  unset(ENABLE_FACEDETECT CACHE)
  unset(ENABLE_YOLOV3 CACHE)
  unset(ENABLE_LANDMARK CACHE)
  unset(ENABLE_FACEQUALITY5PT CACHE)
  unset(ENABLE_REID CACHE)
  unset(ENABLE_RETINAFACE CACHE)
  option(DISABLE_ALL "disable all libraries" ON)
  option(ENABLE_CLASSIFICATION "enable classification" ON)
  option(ENABLE_FACEDETECT "enable face detection" ON)
  option(ENABLE_FACEFEATURE "enable face feature" ON)
  option(ENABLE_YOLOV3 "enable Yolov3" ON)
  option(ENABLE_LANDMARK "enable landmark face-5pt" ON)
  option(ENABLE_FACEQUALITY5PT "enable face quality 5pt" ON)
  option(ENABLE_REID "enable reid" ON)
  option(ENABLE_RETINAFACE "enable face detection" ON)

  # Enable DPUV1 flag
  add_definitions(-DENABLE_DPUCADX8G_RUNNER)
endif()

if(ENABLE_DPUCADF8H_RUNNER)
  # Enble supported models for DPUCADF8H
  unset(DISABLE_ALL CACHE)
  unset(ENABLE_CLASSIFICATION CACHE)
  option(DISABLE_ALL "disable all libraries" ON)
  option(ENABLE_CLASSIFICATION "enable classification" ON)
endif()

# find packages
find_package(unilog)
find_package(xir)
find_package(
  vart
  COMPONENTS util
             xrt-device-handle
             buffer-object
             dpu-controller
             runner
             dpu-runner
             runner-assistant
             mem-manager
             trace
  REQUIRED)
find_package(Protobuf 3.4.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Xrt MODULE)
find_package(OpenCV COMPONENTS opencv_core opencv_video opencv_videoio
                               opencv_imgproc opencv_imgcodecs opencv_highgui)
if(NOT OpenCV_FOUND)
  find_package(
    OpenCV_LIBS
    opencv_core
    opencv_video
    opencv_videoio
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui)
endif(NOT OpenCV_FOUND)
find_package(PkgConfig REQUIRED)
pkg_search_module(JSONC REQUIRED IMPORTED_TARGET json-c)

enable_testing()

# add subirectories
add_subdirectory(usefultools)
add_subdirectory(benchmark)

set(PACKAGE_COMPONENTS
    cxir
    cvart
    extra_ops
    model_config
    math
    runner_helper
    dpu_task
    cpu_task
    graph_runner
    xnnpp
    multi_runner
    xmodel_jit
    xmodel_image)

if(ENABLE_CLASSIFICATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS classification)
endif()

if(ENABLE_VEHICLECLASSIFICATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS vehicleclassification)
endif()

if(ENABLE_TFSSD OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS tfssd)
endif()

if(ENABLE_FACEDETECT OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS facedetect)
endif()

if(ENABLE_FACEFEATURE OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS facefeature)
endif()

if(ENABLE_LANEDETECT OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS lanedetect)
endif()

if(ENABLE_YOLOV2 OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS yolov2)
endif()

if(ENABLE_YOLOV3 OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS yolov3)
endif()

if(ENABLE_OFAYOLO OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS ofa_yolo)
endif()

if(ENABLE_YOLOVX OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS yolovx)
endif()

if(ENABLE_YOLOV6 OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS yolov6)
endif()


if(ENABLE_PMRID OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS pmrid)
endif()

if(ENABLE_ARFLOW OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS arflow)
endif()

if(ENABLE_SOLO OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS solo)
endif()

if(ENABLE_C2D2_LITE OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS c2d2_lite)
endif()

if(ENABLE_BEVDET OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS bevdet)
endif()

if(ENABLE_RGBDSEGMENTATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS RGBDsegmentation)
endif()

if(ENABLE_POLYPSEGMENTATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS polypsegmentation)
endif()

if(ENABLE_LANDMARK OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS facelandmark)
endif()

if(ENABLE_FACEQUALITY5PT OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS facequality5pt)
endif()

if(ENABLE_SSD OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS ssd)
endif()

if(ENABLE_SEGMENTATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS segmentation)
endif()

if(ENABLE_COVID19SEGMENTATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS covid19segmentation)
endif()

if(ENABLE_3DSEGMENTATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS 3Dsegmentation)
endif()

if(ENABLE_REFINEDET OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS refinedet)
endif()

if(ENABLE_OPENPOSE OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS openpose)
endif()

if(ENABLE_HOURGLASS OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS hourglass)
endif()

if(ENABLE_MOVENET OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS movenet)
endif()

if(ENABLE_POSEDETECT OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS posedetect)
endif()

if(ENABLE_REID OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS reid)
endif()

if(ENABLE_RCAN OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS rcan)
endif()

if(ENABLE_FAIRMOT OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS fairmot)
endif()

if(ENABLE_REIDTRACKER OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS reidtracker)
endif()

if(ENABLE_MULTITASK OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS multitask)
endif()

if(ENABLE_MULTITASKV3 OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS multitaskv3)
endif()

if(ENABLE_PLATEDETECT OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS platedetect)
endif()

if(ENABLE_PLATENUM OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS platenum)
endif()

if(ENABLE_PLATERECOG OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS platerecog)
endif()

if(ENABLE_CARPLATERECOG OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS carplaterecog)
endif()

if(ENABLE_MEDICALSEGMENTATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS medicalsegmentation)
endif()

if(ENABLE_MEDICALDETECTION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS medicaldetection)
endif()

if(ENABLE_FACERECOG OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS facerecog)
endif()

if(ENABLE_FACEDETECTRECOG OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS facedetectrecog)
endif()

if(ENABLE_POINTPILLARS OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS pointpillars)
endif()

if(ENABLE_POINTPILLARS_NUSCENES OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS pointpillars_nuscenes)
endif()

if(ENABLE_POINTPAINTING OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS pointpainting)
endif()

if(ENABLE_CENTERPOINT OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS centerpoint)
endif()

if(ENABLE_RETINAFACE OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS retinaface)
endif()

if(ENABLE_MNISTCLASSIFICATION OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS mnistclassification)
endif()

if(ENABLE_MEDICALSEGCELL OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS medicalsegcell)
endif()

if(ENABLE_BCC OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS bcc)
endif()

if(ENABLE_PMG OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS pmg)
endif()

if(ENABLE_ULTRAFAST OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS ultrafast)
endif()

if(ENABLE_OCR OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS ocr)
endif()

if(ENABLE_TEXTMOUNTAIN OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS textmountain)
endif()

if(ENABLE_BTS AND (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS bts)
endif()

if(ENABLE_FUSION_CNN OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS fusion_cnn)
endif()

if(ENABLE_CLOCS OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS clocs)
endif()

if(ENABLE_X_AUTONOMOUS3D AND (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS x_autonomous3d)
endif()

if(ENABLE_EFFICIENTDET_D2 OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS efficientdet_d2)
endif()

if(ENABLE_BOSCH_FCNSEMSEG AND (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS bosch_fcnsemseg)
endif()
if(ENABLE_BOSCH_SSDOBJDET AND (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS bosch_ssdobjdet)
endif()
if(ENABLE_MONODEPTH2 OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS monodepth2)
endif()
if(ENABLE_CFLOWNET OR (NOT DISABLE_ALL))
  list(APPEND PACKAGE_COMPONENTS cflownet)
endif()
if(ENABLE_CENTERNET )
  list(APPEND PACKAGE_COMPONENTS centernet)
endif()

# NOTE: general1 refer to many other libs, so it must be the last components in
# the list
if(ENABLE_GENERAL OR (NOT DISABLE_ALL))
  # use general1 instead of general because general is a special target name in
  # cmake
  list(APPEND PACKAGE_COMPONENTS general1)
endif()

foreach(COMPONENT ${PACKAGE_COMPONENTS})
  add_subdirectory(${COMPONENT})
endforeach()

if(ENABLE_OVERVIEW AND (NOT DISABLE_ALL))
  add_subdirectory(overview)
endif()

# install cmake config files
file(RELATIVE_PATH REL_INCLUDE_DIR
     ${CMAKE_INSTALL_PREFIX}/share/cmake/${PROJECT_NAME}
     ${CMAKE_INSTALL_PREFIX}/include)
set(CONF_INCLUDE_DIRS "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/config.cmake.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake" @ONLY)
write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
        DESTINATION share/cmake/${PROJECT_NAME})

# cpack
set(CPACK_PACKAGE_NAME "libvitis_ai_library")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A runtime runner for vitis")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/Copyright.txt")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "Hui Zhang<huizhang@xilinx.com>")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
set(CPACK_DEB_COMPONENT_INSTALL False)
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
set(CPACK_DEBIAN_COMPRESSION_TYPE "gzip")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libvart (>=0.0.1)")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
if(DEFINED ENV{BUILD_NUMBER})
  set(CPACK_DEBIAN_PACKAGE_RELEASE "r$ENV{BUILD_NUMBER}")
endif()
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortexa72-cortexa53")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
else()
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
endif()
set(CPACK_RPM_COMPONENT_INSTALL False)
set(CPACK_RPM_PACKAGE_LICENSE "Apache")
set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_PACKAGE_REQUIRES "libvart >= 0.0.1")
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
if(DEFINED ENV{BUILD_NUMBER})
  set(CPACK_RPM_PACKAGE_RELEASE "r$ENV{BUILD_NUMBER}")
endif()
set(CPACK_RPM_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "cortexa72-cortexa53")
  set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
endif()

include(CPack)
