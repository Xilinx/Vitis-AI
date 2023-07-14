/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#include <xir/cxir.h>

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/target_factory.hpp>
#include <xir/device_memory.hpp>
#include <xir/util/tool_function.hpp>
#include <xir/xrt_device_handle.hpp>
static std::vector<int> calculate_strides(const std::vector<int>& shape,
                                          size_t size_of_elt) {
  auto ret = std::vector<int>(shape.size(), 1);
  for (int i = ((int)shape.size()) - 2; i >= 0; --i) {
    ret[i] = ret[i + 1] * shape[i + 1];
  }
  for (auto& x : ret) {
    x = x * size_of_elt;
  }
  return ret;
}

static inline xir_string_t conv_to_xir_string(const std::string& s) {
  return xir_string_t{s.data(), s.size()};
}

static inline std::string conv_to_std_string(xir_string_t k) {
  std::string v(k.data, k.data + k.size);
  return v;
}

namespace py = pybind11;
PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "extra ops bindings";
  m.def("remove_xfix", &xir::remove_xfix);
  m.def("xdputil_query", &xdputil_query);
  m.def("xdputil_status", &xdputil_status);
  m.def("xmodel_to_txt", &xmodel_to_txt);
  m.def("xilinx_version", &xilinx_version);
  m.def("test_dpu_runner_mt", &test_dpu_runner_mt);
  m.def("test_op_run", &test_op_run);
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
  py::class_<xir_bytes_t>(m, "AttrBytes", py::buffer_protocol())
      .def_buffer([](xir_bytes_t bytes_value) -> py::buffer_info {
        char* data = bytes_value.data;
        size_t size = bytes_value.size;
        auto bit_width = 8;
        auto shape = std::vector<int>{(int)size};
        CHECK_EQ(shape.size(), 1u);
        auto format = py::format_descriptor<int8_t>::format();
        return py::buffer_info((void*)data,   /* Pointer to buffer */
                               bit_width / 8, /* Size of one scalar */
                               format,        /* Python struct-style
                                                 format descriptor */
                               shape.size(),  /* Number of dimensions */
                               shape,         /* Buffer dimensions */
                               calculate_strides(shape, bit_width / 8));
      });

  m.def("xir_get_attr_binary",
        [](const xir::Subgraph& sg, const std::string& name) {
          auto attrs = xir_subgraph_get_attrs((void*)&sg);
          auto attr_value = xir2_attrs_get(attrs, conv_to_xir_string(name));
          CHECK_EQ(attr_value.tag, XIR_ATTR_TYPE_TAG_bytes);
          return attr_value.u.bytes_value;
        });
  m.def("xir_get_attr_map_binary", [](const xir::Subgraph& sg,
                                      const std::string& name) {
    auto attrs = xir_subgraph_get_attrs((void*)&sg);
    auto attr_value = xir2_attrs_get(attrs, conv_to_xir_string(name));
    CHECK_EQ(attr_value.tag, XIR_ATTR_TYPE_TAG_MAP_bytes);
    auto ret = std::map<std::string, xir_bytes_t>();
    auto iter = attr_value.u.map_value;
    for (xir_attr_pair_t x = iter->next(iter->self);
         x.first.tag != XIR_ATTR_TYPE_TAG_NONE; x = iter->next(iter->self)) {
      auto key = x.first.u.string_value;
      CHECK_EQ(x.first.tag, xir_attr_value_tag_t::XIR_ATTR_TYPE_TAG_string);
      CHECK_EQ(x.second.tag, xir_attr_value_tag_t::XIR_ATTR_TYPE_TAG_bytes);
      ret.insert({conv_to_std_string(key), x.second.u.bytes_value});
    }
    iter->destroy(iter->self);
    delete iter;
    return ret;
  });
}
