# Copyright 2019 Xilinx Inc.
#
# Distributed under the Apache License, Version 2.0 (the "License");
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
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_C_STANDARD 99)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -ggdb -O0 -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wall -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_EXE "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

set(CMAKE_MACOSX_RPATH 1)

if(CMAKE_BUILD_TYPE MATCHES "Release")
  ADD_DEFINITIONS(-DUNI_LOG_NDEBUG)
endif()

include(CMakePackageConfigHelpers)

if(CMAKE_CROSSCOMPILING)
  file(RELATIVE_PATH RPATH_RELATIVE ${CMAKE_SYSROOT} ${CMAKE_INSTALL_PREFIX})
  if(RPATH_RELATIVE MATCHES "^\\.\\.")
    message(FATAL_ERROR "Install prefix should be in SYSROOT:${CMAKE_SYSROOT}, but ${CMAKE_INSTALL_PREFIX} is specified. RPATH_REALTIVE=(${RPATH_RELATIVE})")
  else()
    set(CMAKE_INSTALL_RPATH "/${RPATH_RELATIVE}/lib")
    list(APPEND CMAKE_PREFIX_PATH "/${RPATH_RELATIVE}/")
  endif()
else()
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()
