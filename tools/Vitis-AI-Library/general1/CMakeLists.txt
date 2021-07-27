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

vai_add_library(
  SRCS include/vitis/ai/general.hpp src/general_adapter.hpp src/general.cpp
  PUBLIC_REQUIRE
    glog::glog
    ${OpenCV_LIBS}
    protobuf::libprotobuf
    ${PROJECT_NAME}::model_config
    ${PROJECT_NAME}::refinedet
    ${PROJECT_NAME}::facefeature
    ${PROJECT_NAME}::segmentation
    ${PROJECT_NAME}::reid
    ${PROJECT_NAME}::multitask
    ${PROJECT_NAME}::medicalsegmentation
    ${PROJECT_NAME}::posedetect
    ${PROJECT_NAME}::platenum
    ${PROJECT_NAME}::platedetect
    ${PROJECT_NAME}::facelandmark
    ${PROJECT_NAME}::lanedetect
    ${PROJECT_NAME}::ssd
    ${PROJECT_NAME}::tfssd
    ${PROJECT_NAME}::yolov2
    ${PROJECT_NAME}::yolov3
    ${PROJECT_NAME}::classification
    ${PROJECT_NAME}::facedetect
    ${PROJECT_NAME}::facedetectrecog)

vai_add_test(test_general)
vai_add_test(test_general_batch)
vai_add_test(generate_proto)
vai_add_test(compare_proto)
vai_add_test(show_proto)
vai_add_sample(
  general_example_0
  SRCS samples/process_image.hpp samples/xmodel_result_to_string.hpp
       samples/general1_example_0.cpp)
