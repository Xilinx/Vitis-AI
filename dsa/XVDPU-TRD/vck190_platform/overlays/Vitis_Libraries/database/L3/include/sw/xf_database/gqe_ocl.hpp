/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef GQE_OCL_HPP
#define GQE_OCL_HPP
#include <CL/cl_ext_xilinx.h>
#include <CL/cl.h>

#include <new>
#include <cstddef>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace xf {
namespace database {
namespace gqe {

cl_int init_hardware(cl_context* context,
                     cl_device_id* device_id,
                     cl_command_queue* cmd_queue,
                     cl_command_queue_properties queue_props);

cl_int load_binary(cl_program* program, cl_context context, cl_device_id device_id, const char* xclbin);

} // gqe
} // database

} // namespace xf
#endif // GQE_OCL_HPP
