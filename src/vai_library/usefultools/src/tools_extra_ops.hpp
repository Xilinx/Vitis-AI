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
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <xir/graph/subgraph.hpp>
namespace py = pybind11;
std::string xmodel_to_txt(std::string xmodel);
py::dict xdputil_query();
py::dict xdputil_status();
std::vector<uint32_t> read_register(void* handle, uint32_t ip_index,
                                    uint64_t cu_base_addr,
                                    const std::vector<uint32_t>& addrs);
std::vector<std::string> xilinx_version(std::vector<std::string> so_names);
std::vector<std::string> xilinx_version2(std::vector<std::string> so_names);
bool test_dpu_runner_mt(const xir::Subgraph* subgraph, uint32_t runner_num,
                        const std::vector<std::string>& input_filenames,
                        const std::vector<std::string>& output_filenames);

template <class T>
std::string to_string(T t, std::ios_base& (*f)(std::ios_base&),
                      std::string prefix = "0x") {
  std::ostringstream oss;
  oss << prefix << f << t;
  return oss.str();
}
std::map<std::string, std::string> get_reg_id_to_parameter(
    const xir::Subgraph* s);
bool test_op_run(const std::string& graph, const std::string& op,
                 const std::string& ref_dir, const std::string& dump_dir);
