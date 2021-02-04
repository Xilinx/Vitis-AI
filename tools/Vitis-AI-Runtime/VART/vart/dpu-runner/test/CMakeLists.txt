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
aux_source_directory(. TEST_SRC)

execute_process(
  COMMAND
    bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/convert_word_list.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/word_list.txt
    ${CMAKE_CURRENT_BINARY_DIR}/word_list.inc)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

find_package(OpenCV REQUIRED)
foreach(FNAME resnet50.cpp test_dpu_runner.cpp test_dpu_runner_mt.cpp)
  get_filename_component(F_PREFIX ${FNAME} NAME_WE)
  set(ELF ${F_PREFIX})

  add_executable(${ELF} ${FNAME})
  target_link_libraries(
    ${ELF}
    ${COMPONENT_NAME}
    xir::xir
    glog::glog
    opencv_core
    opencv_video
    opencv_videoio
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    util
    pthread
    mem-manager)
endforeach()
install(TARGETS test_dpu_runner_mt DESTINATION bin)

add_executable(show_kernel show_kernel.cpp)
target_link_libraries(show_kernel xir::xir)
install(
  CODE "execute_process (
    COMMAND env CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} bash ${CMAKE_CURRENT_SOURCE_DIR}/pack_for_trd.sh
)")

add_executable(test_dpu_runner_c test_dpu_runner_c.c)
target_link_libraries(test_dpu_runner_c ${COMPONENT_NAME})

# add_executable (resnet50_zero_copy ../samples/resnet50_zero_copy/resnet50.cpp)
# target_link_libraries (resnet50_zero_copy ${COMPONENT_NAME}
# ${PROJECT_NAME}::runner dpu-controller opencv_core opencv_video opencv_videoio
# opencv_imgproc opencv_imgcodecs opencv_highgui XRT::XRT )
