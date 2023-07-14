#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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
aux_source_directory(. TEST_SRC)
foreach(FNAME ${TEST_SRC})
  get_filename_component(F_PREFIX ${FNAME} NAME_WE)
  set(ELF ${F_PREFIX})

  add_executable(${ELF} ${FNAME})
  target_link_libraries(${ELF}
    ${COMPONENT_NAME} xir::xir glog::glog ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endforeach()
