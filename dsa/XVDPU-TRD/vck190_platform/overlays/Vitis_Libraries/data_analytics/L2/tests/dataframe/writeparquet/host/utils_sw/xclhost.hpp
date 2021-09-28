/*
 * Copyright 2018 Xilinx, Inc.
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
 */

#ifndef XCLHOST_HPP
#define XCLHOST_HPP

// Include xilinx header first, to define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl_ext_xilinx.h>
#include <CL/cl.h>

#include <cstddef>
#include <cstdlib>
#include <new>

#define MSTR_(m) #m
#define MSTR(m) MSTR_(m)

namespace xclhost {
cl_int init_hardware(cl_context* context,
                     cl_device_id* device_id,
                     cl_command_queue* cmd_queue,
                     cl_command_queue_properties queue_props,
                     const char* dsa_name);

cl_int load_binary(cl_program* program, cl_context context, cl_device_id device_id, const char* xclbin);

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}
} // namespace xclhost

#endif // XCLHOST_HPP
