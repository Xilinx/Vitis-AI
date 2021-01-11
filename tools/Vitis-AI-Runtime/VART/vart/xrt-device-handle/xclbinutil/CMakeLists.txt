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
get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
find_package (Boost COMPONENTS filesystem)
find_package (OpenSSL)
add_library(${COMPONENT_NAME} SHARED
  "FormattedOutput.cxx"
  "ParameterSectionData.cxx"
  "XclBinUtilMain.cxx"
  "XclBinClass.cxx"
  "SectionHeader.cxx"
  "XclBinUtilities.cxx"
  "Section.cxx"
  "SectionBitstream.cxx"
  "SectionClearBitstream.cxx"
  "SectionDNACertificate.cxx"
  "SectionEmbeddedMetadata.cxx"
  "SectionManagementFW.cxx"
  "SectionDebugData.cxx"
  "SectionSchedulerFW.cxx"
  "SectionMemTopology.cxx"
  "SectionConnectivity.cxx"
  "SectionIPLayout.cxx"
  "SectionDebugIPLayout.cxx"
  "SectionDesignCheckPoint.cxx"
  "SectionClockFrequencyTopology.cxx"
  "SectionMCS.cxx"
  "SectionBMC.cxx"
  "SectionBuildMetadata.cxx"
  "SectionKeyValueMetadata.cxx"
  "SectionUserMetadata.cxx"
  "SectionPDI.cxx"
  "SectionBitstreamPartialPDI.cxx"
  "SectionPartitionMetadata.cxx"
  "SectionEmulationData.cxx"
  "SectionSystemMetadata.cxx"
  "SectionSoftKernel.cxx"
  "DTC.cxx"
  "DTCStringsBlock.cxx"
  "FDTNode.cxx"
  "FDTProperty.cxx"
  "XclBinSignature.cxx")
target_compile_options(${COMPONENT_NAME} PRIVATE "-Wno-maybe-uninitialized")
set_target_properties(${COMPONENT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  LIBRARY_OUTPUT_NAME ${PROJECT_NAME}-${COMPONENT_NAME}
  BUILD_RPATH "\$ORIGIN/../util"
  )
target_link_libraries(${COMPONENT_NAME}
  PRIVATE pthread
  PUBLIC XRT::XRT util Boost::filesystem OpenSSL::Crypto)
target_include_directories(${COMPONENT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

install(
  TARGETS ${COMPONENT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)
