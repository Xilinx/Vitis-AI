# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Distributed under the Apache License, Version 2.0 (the "License"); you may not
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
set(VITIS_AI_LIBRARY_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR})

include(GNUInstallDirs)
set(INSTALL_LIBDIR
    ${CMAKE_INSTALL_LIBDIR}
    CACHE PATH "Installation directory for libraries")
set(INSTALL_BINDIR
    ${CMAKE_INSTALL_BINDIR}
    CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDEDIR
    ${CMAKE_INSTALL_INCLUDEDIR}
    CACHE PATH "Installation directory for headr files")
set(INSTALL_CMAKEDIR
    ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    CACHE PATH "Installation directory for cmake files")

add_library(gcc_atomic INTERFACE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(microblazeel)")
  # for a unknown reason, gcc on microblaze requires libatomic
  # message(FATAL_ERROR "HELLO")
  target_link_libraries(gcc_atomic INTERFACE -latomic)
endif()

function(vai_add_library)
  set(options STATIC SHARED MODULE NOT_GLOB_SRC)
  set(oneValueArgs NAME INCLUDE_DIR SRC_DIR TEST_DIR)
  set(multiValueArgs PUBLIC_REQUIRE PRIVATE_REQUIRE PUBLIC_HEADER SRCS TESTS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  # start check type
  if(ARG_STATIC)
    set(_type STATIC)
  elseif(ARG_SHARED)
    set(_type STATIC)
  elseif(ARG_MODULE)
    set(_type MODULE)
  else()
    set(_type "")
  endif()
  # end check type

  # start to check SRC
  # ~~~
  # if(NOT ARG_SRCS)
  #   set(ARG_SRCS ""
  # endif(NOT ARG_SRCS)
  # ~~~
  # end check SRC

  # start check include dir
  if(NOT ARG_INCLUDE_DIR)
    set(ARG_INCLUDE_DIR "include")
  endif(NOT ARG_INCLUDE_DIR)
  # end check include dir

  # start to check src dir
  if(NOT ARG_SRC_DIR)
    set(ARG_SRC_DIR "src")
  endif(NOT ARG_SRC_DIR)
  # end check src dir

  # start to check test dir
  if(NOT ARG_TEST_DIR)
    set(ARG_TEST_DIR "test")
  endif(NOT ARG_TEST_DIR)
  # end check test dir

  # start check target name
  if(NOT ARG_NAME)
    get_filename_component(ARG_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    set(COMPONENT_NAME
        ${ARG_NAME}
        PARENT_SCOPE)
  endif(NOT ARG_NAME)
  # end check target name

  # create the target
  message(STATUS "create target ${ARG_NAME} ${_type} ${ARG_SRCS}")
  add_library(${ARG_NAME} ${_type} ${ARG_SRCS})
  # add source codes it is not recommended in cmake doc to glob source codes
  #
  # ~~~
  # if(NOT ARG_NOT_GLOB_SRC)
  #   file(GLOB_RECURSE _srcs ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC_DIR} "*.cpp"
  #        "*.cc")
  #   message(
  #     STATUS
  #       "auto detect source code for ${ARG_NAME} from ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC_DIR}: ${_srcs}"
  #   )
  #   foreach(_src ${_srcs})
  #     file(RELATIVE_PATH _src ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC_DIR}
  #          ${_src})
  #     message(STATUS "\t\t${_src}")
  #   endforeach(_src)
  #   target_sources(${ARG_NAME} PRIVATE ${_srcs})
  # endif(NOT ARG_NOT_GLOB_SRC)
  # ~~~
  #
  # create alias
  add_library(${PROJECT_NAME}::${ARG_NAME} ALIAS ${ARG_NAME})
  # include version info
  config_vitis_ai_lib_target(${ARG_NAME})
  # add dependencies
  if(ARG_PUBLIC_REQUIRE)
    target_link_libraries(${ARG_NAME} PUBLIC ${ARG_PUBLIC_REQUIRE})
  endif(ARG_PUBLIC_REQUIRE)
  if(ARG_PRIVATE_REQUIRE)
    target_link_libraries(${ARG_NAME} PRIVATE ${ARG_PRIVATE_REQUIRE})
  endif(ARG_PRIVATE_REQUIRE)
  # add public headers
  if(ARG_PUBLIC_HEADER)
    # suppress warning contains relative path in its INTERFACE_SOURCES:
    cmake_policy(SET CMP0076 NEW)
    # cmake cannot preserve header file directory heirarachy
    # set_target_properties(${ARG_NAME} PROPERTIES # PUBLIC_HEADER
    # "${ARG_PUBLIC_HEADER}" )
    target_sources(${ARG_NAME} PRIVATE ${ARG_PUBLIC_HEADER})
  endif(ARG_PUBLIC_HEADER)
  target_include_directories(
    ${ARG_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>)
  # Prepare RPATH
  file(RELATIVE_PATH _rpath ${CMAKE_INSTALL_PREFIX}/${INSTALL_BINDIR}
       ${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBDIR})
  if(APPLE)
    set(_rpath "@loader_path/${_rpath}")
  else()
    set(_rpath "\$ORIGIN/${_rpath}")
  endif()
  file(TO_NATIVE_PATH "${_rpath}/${INSTALL_LIBDIR}" ${_rpath})
  # set all properties
  set_target_properties(
    ${ARG_NAME}
    PROPERTIES VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               LIBRARY_OUTPUT_NAME ${PROJECT_NAME}-${ARG_NAME}
               MACOSX_RPATH ON
               POSITION_INDEPENDENT_CODE 1
               SKIP_BUILD_RPATH ON
               BUILD_WITH_INSTALL_RPATH ON
               INSTALL_RPATH "${_rpath}"
               INSTALL_RPATH_USE_LINK_PATH ON)
  install(
    TARGETS ${ARG_NAME}
    EXPORT ${ARG_NAME}-targets
    RUNTIME DESTINATION ${INSTALL_BINDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
            # cmake cannot reserve public header file heirarachy so that we
            # have install them one by on
            # PUBLIC_HEADER DESTINATION
            # ${INSTALL_INCLUDEDIR}
  )
  file(GLOB _headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
  foreach(_header ${_headers})
    if(IS_DIRECTORY ${_header})
      message("install_header ${_header}")
      install(
        DIRECTORY ${_header}
        DESTINATION ${INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h*"
        PATTERN "*.inc")
    else()
      install(FILES ${_header} DESTINATION ${INSTALL_INCLUDEDIR})
    endif()
  endforeach(_header)
  install(
    EXPORT ${ARG_NAME}-targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKEDIR})
  # ~~~
  # if(ARG_TESTS)
  #   set(_test_options "")
  #   set(_test_single_options "")
  #   set(_test_multi_options "")
  #   cmake_parse_arguments(_TEST_ARG ${_test_options} ${_test_single_options}
  #     ${_test_multi_options} ${ARG_TESTS})
  # endif(ARG_TESTS)
  # ~~~
endfunction(vai_add_library)

function(vai_add_test name)
  set(options "")
  set(oneValueArgs ENABLE_IF)
  set(multiValueArgs REQUIRE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(_enable TRUE)
  if(ARG_ENABLE_IF)
    set(_enable ${${ARG_ENABLE_IF}})
  endif(ARG_ENABLE_IF)
  if(_enable)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/${name}.cpp)
      add_executable(${name} test/${name}.cpp)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/${name}.c)
      add_executable(${name} test/${name}.c)
    else()
      message(
        FATAL_ERROR "cannot find either test/${name}.c or test/${name}.cpp")
    endif()
    target_link_libraries(${name} ${PROJECT_NAME}::${COMPONENT_NAME})
    if(ARG_REQUIRE)
      target_link_libraries(${name} ${ARG_REQUIRE})
    endif(ARG_REQUIRE)
    install(
      TARGETS ${name}
      DESTINATION
        ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/test/${COMPONENT_NAME})
  endif(_enable)
endfunction(vai_add_test)

function(vai_add_sample name)
  set(options "")
  set(oneValueArgs DESTINATION)
  set(multiValueArgs REQUIRE SRCS BUILD)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_executable(${name} ${ARG_SRCS})
  target_link_libraries(${name} ${PROJECT_NAME}::${COMPONENT_NAME})
  if(ARG_REQUIRE)
    target_link_libraries(${name} ${ARG_REQUIRE})
  endif(ARG_REQUIRE)
  if(NOT ARG_DESTINATION)
    set(ARG_DESTINATION ".")
  endif(NOT ARG_DESTINATION)
  install(
    TARGETS ${name}
    DESTINATION
      ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples/${COMPONENT_NAME}/${ARG_DESTINATION}
  )
  install(
    FILES ${ARG_SRCS}
    DESTINATION
      ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples/${COMPONENT_NAME}/${ARG_DESTINATION}
  )
  install(
    FILES ${ARG_BUILDS}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ 
        GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION
      ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples/${COMPONENT_NAME}/${ARG_DESTINATION}
  )
endfunction(vai_add_sample)

function(config_vitis_ai_lib_target target_name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "${target_name} is not a valid target")
  endif(NOT TARGET ${target_name})
  config_vitis_ai_lib_target_version_support(${target_name})
endfunction()

execute_process(COMMAND ${CMAKE_COMMAND} --help-property-list
                OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

function(config_vitis_ai_lib_target_version_support target_name)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version2.c.kv
       "# export variables for versio info" \n)
  _get_all_properties(${target_name} _properties)
  _k(PROJECT_NAME)
  _kv(COMPONENT_NAME ${target_name})
  _kv(BUILD_ID "$ENV{BUILD_ID}")
  _kv(BUILD_USER "$ENV{USER}")
  _kv(CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  _k(CMAKE_CXX_FLAGS)
  _k(CMAKE_CXX_FLAGS_DEBUG)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    _k(${_variableName})
  endforeach()
  foreach(
    _info
    OS_NAME
    OS_RELEASE
    OS_VERSION
    OS_PLATFORM
    FQDN
    PROCESSOR_SERIAL_NUMBER
    PROCESSOR_NAME
    PROCESSOR_DESCRIPTION)
    cmake_host_system_information(RESULT _r QUERY ${_info})
    _kv(${_info} ${_r} "")
  endforeach(_info)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version.c
    COMMAND ${CMAKE_COMMAND} -DCURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} -P
            ${VITIS_AI_LIBRARY_CMAKE_DIR}/git-version.cmake
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version.c.kv
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  target_sources(${target_name}
                 PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version.c)
endfunction(config_vitis_ai_lib_target_version_support)

function(_get_all_properties target_name)
  set(_RET "")
  foreach(_prop ${CMAKE_PROPERTY_LIST})
    string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" _prop ${_prop})
    if(_prop STREQUAL "LOCATION"
       OR _prop MATCHES "^LOCATION_"
       OR _prop MATCHES "_LOCATION$")
      continue()
    endif()
    get_property(
      _propval
      TARGET ${target_name}
      PROPERTY ${_prop}
      SET)
    if(_propval)
      get_target_property(_propval ${target_name} ${_prop})
      _kv(${_prop} "${_propval}")
    else(_propval)
      _kv(${_prop} "${_propval}")
    endif(_propval)
  endforeach(_prop)
  set(${result}
      ${_RET}
      PARENT_SCOPE)
endfunction(_get_all_properties)

function(_kv k v)
  # message("hello ${k} = ${v} ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version.c.kv")
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version.c.kv
       "set(" ${k} " [=[" "${v}" "]=])" \n)
endfunction(_kv)

function(_k k)
  # message("hello ${k} = ${v} ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version.c.kv")
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/xilinx_version.c.kv
       "set(" ${k} " [=[" ${${k}} "]=])" \n)
endfunction(_k)
