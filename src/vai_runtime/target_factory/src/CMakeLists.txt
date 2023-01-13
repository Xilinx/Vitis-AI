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

# get the git repo branch and commit id and write them into config.hpp
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE PROJECT_GIT_BRANCH_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE PROJECT_GIT_COMMIT_ID
  OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(config.hpp.in config.hpp @ONLY)

# run protoc
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO TARGET_FACTORY_DLLESPEC target.proto)
#set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-unused-variable)
if(BUILD_SHARED_LIBS)
    if(MSVC)
        set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_DEFINITIONS "TARGET_FACTORY_DLLESPEC=__declspec(dllexport)")
    else(MSVC)
        set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_DEFINITIONS "TARGET_FACTORY_DLLESPEC=__attribute__((visibility(\"default\")))")
    endif(MSVC)
else(BUILD_SHARED_LIBS)
    set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_DEFINITIONS "TARGET_FACTORY_DLLESPEC=")
endif(BUILD_SHARED_LIBS)

if(NOT MSVC)
    set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-array-bounds)
endif(NOT MSVC)
# process target files
file(GLOB PROTOTXT_FILE_LIST ${PROJECT_SOURCE_DIR}/targets/*.prototxt)
if(NOT ENABLE_IPU)
    file(GLOB AIE2_SRCS ${PROJECT_SOURCE_DIR}/targets/*AMD_AIE2*.prototxt)
    list(REMOVE_ITEM PROTOTXT_FILE_LIST ${AIE2_SRCS})
endif(NOT ENABLE_IPU)
set(TARGET_PROTOTXT_LST "")
set(TARGET_PROTOTXT_CODE "")
foreach(PROTOTXT_FILE ${PROTOTXT_FILE_LIST})
  file(READ ${PROTOTXT_FILE} _hex HEX)
  # Separate into individual bytes.
  string(REGEX MATCHALL "([A-Za-z0-9][A-Za-z0-9])" _separated_hex "${_hex}")
  list(APPEND _separated_hex "00")
  # Append the "0x" to each byte.
  list(JOIN _separated_hex ",0x" _formatted_hex)
  # JOIN misses the first byte's "0x", so add it here.
  string(PREPEND _formatted_hex "0x")
  string(MAKE_C_IDENTIFIER ${PROTOTXT_FILE} _c_name)
  string(APPEND TARGET_PROTOTXT_CODE "static constexpr char ${_c_name} [] = { ${_formatted_hex} };\n")
  list(APPEND TARGET_PROTOTXT_LST "(const char *)${_c_name}")
endforeach()
list(JOIN TARGET_PROTOTXT_LST "," TARGET_PROTOTXTS)
configure_file(${PROJECT_SOURCE_DIR}/src/target_list.hpp.in target_list.hpp
               ESCAPE_QUOTES @ONLY)

# build library
aux_source_directory(. CPP_SRCS)
add_library(${PROJECT_NAME}
  ../include/vitis/ai/target_factory.hpp
  ${PROTO_SRCS} ${CPP_SRCS})
# Enable Export of public APIs
target_compile_definitions(${PROJECT_NAME} PRIVATE -DTARGET_FACTORY_EXPORT
	# for an unknown reason, g++ 9.3 produces compilation errors related to PRI64X
	-D__STDC_FORMAT_MACROS=1)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DTARGET_FACTORY_USE_DLL=1)
else(BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DTARGET_FACTORY_USE_DLL=0)
endif(BUILD_SHARED_LIBS)

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC protobuf::libprotobuf PRIVATE unilog::unilog)

# add python api
if(BUILD_PYTHON)
    add_subdirectory(python_api)
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  FILES ${PROTO_HDRS} ${PROJECT_SOURCE_DIR}/include/vitis/ai/target_factory.hpp
  DESTINATION include/vitis/ai)
install(
  EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/cmake/${PROJECT_NAME})
