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

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_C_STANDARD 99)

if(IS_DISTRIBUTION)
  set(extra_cxx_flags "-Wno-maybe-uninitialized")
endif(IS_DISTRIBUTION)

if (MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
   # too many warning about DLL interface because of protobuf/stubs/status.h
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
   # too many warning about type conversion.
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267")
   # conversion from '_Ty' to 'int', possible loss of data
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244")

else(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -ggdb -O0 -fno-inline")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -Werror ${extra_cxx_flags}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
	set(CMAKE_EXE "${CMAKE_C_FLAGS} -Wall -Werror")
	set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
	set(CMAKE_MACOSX_RPATH 1)
endif(MSVC)

if(CMAKE_BUILD_TYPE MATCHES "Release")
  ADD_DEFINITIONS(-DUNI_LOG_NDEBUG)
endif()

include(CMakePackageConfigHelpers)

if(NOT CMAKE_CROSSCOMPILING)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)
endif()
