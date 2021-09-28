/**********
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
**********/

#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>

#ifdef XILINX_XRT
#include <CL/cl_ext_xilinx.h>
#else
#include <CL/cl_ext.h>
#endif

// When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood
// User ptr is used if and only if it is properly aligned (page aligned). When not
// aligned, runtime has no choice but to create its own host side buffer that backs
// user ptr. This in turn implies that all operations that move data to and from
// device incur an extra memcpy to move data to/from runtime's own host buffer
// from/to user pointer. So it is recommended to use this allocator if user wish to
// Create Buffer/Memory Object with CL_MEM_USE_HOST_PTR to align user buffer to the
// page boundary. It will ensure that user buffer will be used when user create
// Buffer/Mem Object with CL_MEM_USE_HOST_PTR.
template <typename T>
struct aligned_allocator {
    using value_type = T;
    T* allocate(std::size_t num) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t num) { free(p); }
};

namespace xcl {
std::vector<cl::Device> get_xil_devices();
std::vector<cl::Device> get_devices(const std::string& vendor_name);
/* find_xclbin_file
 *
 *
 * Description:
 *   Find precompiled program (as commonly created by the Xilinx OpenCL
 *   flow). Using search path below.
 *
 *   Search Path:
 *      $XCL_BINDIR/<name>.<target>.<device>.xclbin
 *      $XCL_BINDIR/<name>.<target>.<device_versionless>.xclbin
 *      $XCL_BINDIR/binary_container_1.xclbin
 *      $XCL_BINDIR/<name>.xclbin
 *      xclbin/<name>.<target>.<device>.xclbin
 *      xclbin/<name>.<target>.<device_versionless>.xclbin
 *      xclbin/binary_container_1.xclbin
 *      xclbin/<name>.xclbin
 *      ../<name>.<target>.<device>.xclbin
 *      ../<name>.<target>.<device_versionless>.xclbin
 *      ../binary_container_1.xclbin
 *      ../<name>.xclbin
 *      ./<name>.<target>.<device>.xclbin
 *      ./<name>.<target>.<device_versionless>.xclbin
 *      ./binary_container_1.xclbin
 *      ./<name>.xclbin
 *
 * Inputs:
 *   _device_name - Targeted Device name
 *   xclbin_name - base name of the xclbin to import.
 *
 * Returns:
 *   An opencl program Binaries object that was created from xclbin_name file.
 */
std::string find_binary_file(const std::string& _device_name, const std::string& xclbin_name);
cl::Program::Binaries import_binary_file(std::string xclbin_file_name);
bool is_emulation();
bool is_hw_emulation();
bool is_xpr_device(const char* device_name);
}
