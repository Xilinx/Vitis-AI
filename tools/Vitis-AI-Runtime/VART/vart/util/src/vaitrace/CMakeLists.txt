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
project(vaitrace VERSION 1.2.0)

get_filename_component(COMPONENT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
	# g++ -shared apm.cpp apm_shell.cpp -fPIC -o libxapm.so -I./include
	set(XAPM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vaitraceTools/xapm)
	include_directories(${XAPM_DIR}/include)
	
	add_library(xapm SHARED
		${XAPM_DIR}/apm.cpp
		${XAPM_DIR}/apm_shell.cpp
	)

	target_link_libraries(xapm pthread)
	set_target_properties(xapm PROPERTIES
		VERSION "${PROJECT_VERSION}"
		SOVERSION "${PROJECT_VERSION_MAJOR}")

	install(TARGETS xapm DESTINATION lib/)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION bin/xlnx
	FILES_MATCHING
	PATTERN "*.py"
	PATTERN .git EXCLUDE
	PATTERN script EXCLUDE
	PATTERN test_cfg EXCLUDE
	PATTERN time_sync EXCLUDE
	PATTERN build EXCLUDE
	PATTERN vaitrace_py EXCLUDE
	PATTERN vaitraceTools/xapm EXCLUDE
	PATTERN vaitrace.py PERMISSIONS
	WORLD_EXECUTE WORLD_READ 
	GROUP_EXECUTE GROUP_READ
	OWNER_EXECUTE OWNER_READ
	)


execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ./xlnx/${COMPONENT_NAME}/vaitrace.py
        ${CMAKE_CURRENT_BINARY_DIR}/vaitrace
    )
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vaitrace DESTINATION bin)

if(BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Development)
    set(PYTHON_SITE_PACKAGES_USER "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
    message("${PYTHON_SITE_PACKAGES_USER}")
    install(DIRECTORY python/vaitrace_py DESTINATION ${PYTHON_SITE_PACKAGES_USER}
	    FILES_MATCHING
	    PATTERN "*.py"
    )

endif()
