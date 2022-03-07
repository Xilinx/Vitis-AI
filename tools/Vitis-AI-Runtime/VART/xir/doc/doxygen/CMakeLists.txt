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
include(FindDoxygen)

# doxygen setting
set(DOXYGEN_CONFIG_DIR ${PROJECT_SOURCE_DIR}/doc/doxygen)
configure_file(${DOXYGEN_CONFIG_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ALL
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
  COMMAND cp ${DOXYGEN_CONFIG_DIR}/Logo.png ${CMAKE_BINARY_DIR}/
  # optional, only necessary if you refer some images in your document.
  # COMMAND cp -av ${DOXYGEN_CONFIG_DIR}/images ${CMAKE_BINARY_DIR}/c-doc/latex
  COMMAND make -C ${CMAKE_BINARY_DIR}/c-doc/latex
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Generating Documentation with Doxygen" VERBATIM
  )
install(FILES ${CMAKE_BINARY_DIR}/c-doc/latex/refman.pdf DESTINATION ${PROJECT_SOURCE_DIR}/doc
  RENAME xir-reference-manual.pdf)
