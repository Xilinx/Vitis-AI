/*
 * Copyright 2019 Xilinx Inc.
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
 * */

#include "tools_extra_ops.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/target_factory.hpp>
#include <xir/device_memory.hpp>
#include <xir/util/tool_function.hpp>
#include <xir/xrt_device_handle.hpp>

namespace py = pybind11;
PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "extra ops bindings";
  m.def("remove_xfix", &xir::remove_xfix);
  m.def("xdputil_query", &xdputil_query);
  m.def("xdputil_status", &xdputil_status);
  m.def("xmodel_to_txt", &xmodel_to_txt);
  m.def("xilinx_version", &xilinx_version);
  m.def("test_dpu_runner_mt", &test_dpu_runner_mt);
  m.def("get_reg_id_to_parameter", &get_reg_id_to_parameter);
  m.def("mem_read", [](unsigned long addr, unsigned long size) {
    auto device_memory = xir::DeviceMemory::create((size_t)0ull);
    std::vector<uint8_t> ret(size);
    device_memory->download(ret.data(), addr, size);
    return ret;
  });
  m.def("mem_write", [](std::vector<uint8_t> data, unsigned long addr) {
    auto device_memory = xir::DeviceMemory::create((size_t)0ull);
    device_memory->upload(data.data(), addr, data.size());
  });
}
