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

#include <vitis/ai/target_factory.hpp>
#include <xir/device_memory.hpp>
#include <xir/util/tool_function.hpp>
#include <xir/xrt_device_handle.hpp>

namespace py = pybind11;
PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "extra ops bindings";
  // py::module::import("xir");
  m.def("remove_xfix", &xir::remove_xfix);
  m.def("xmodel_to_txt", &xmodel_to_txt);
  m.def("read_register", &read_register);
  m.def("xilinx_version", &xilinx_version);
  m.def("get_target_factory_id", &vitis::ai::TargetFactory::get_lib_id);
  m.def("get_target_factory_name", &vitis::ai::TargetFactory::get_lib_name);
  m.def("test_dpu_runner_mt", &test_dpu_runner_mt);
  m.def("device_info", []() {
    auto h = xir::XrtDeviceHandle::get_instance();
    auto cu_name = std::string("");
    std::vector<py::dict> result;
    for (auto i = 0u; i < h->get_num_of_cus(cu_name); ++i) {
      py::dict res_i;
      res_i["cu_idx"] = i;
      res_i["device_id"] = h->get_device_id(cu_name, i);
      res_i["cu_name"] = h->get_cu_full_name(cu_name, i);
      res_i["fingerprint"] = h->get_fingerprint(cu_name, i);
      res_i["cu_handle"] = (uint64_t)h->get_handle(cu_name, i);
      res_i["cu_mask"] = h->get_cu_mask(cu_name, i);
      res_i["cu_addr"] = h->get_cu_addr(cu_name, i);
      result.push_back(res_i);
    }
    return result;
  });
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
  m.def("get_target_type", [](uint64_t fingerprint) {
    auto target = vitis::ai::target_factory()->create(fingerprint);
    return target.type();
  });
  m.def("get_target_name", [](uint64_t fingerprint) {
    auto target = vitis::ai::target_factory()->create(fingerprint);
    return target.name();
  });
  m.def("get_reg_id_to_parameter", [](const xir::Subgraph* s) {
    return s->get_attr<std::map<std::string, std::vector<char>>>(
        "reg_id_to_parameter_value");
  });
}
