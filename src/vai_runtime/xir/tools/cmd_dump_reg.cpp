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

#include "cmd_dump_reg.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

#include "xir/util/tool_function.hpp"
using namespace std;

CmdDumpReg::CmdDumpReg(const std::string& name) : Cmd(name) {}

static std::string indent(size_t n) { return std::string(n * 4u, ' '); }
static std::string get_device(const xir::Subgraph* s) {
  auto ret = std::string("");
  if (s->has_attr("device")) {
    ret = s->get_attr<std::string>("device");
  }
  return ret;
}

static std::string reg_info(const xir::Subgraph* s, int n) {
  std::ostringstream str;
  auto reg_id_to_context_type =
      s->get_attr<std::map<std::string, std::string>>("reg_id_to_context_type");
  str << indent(n + 1) << "reg_id_to_context_type:" << '\n';
  for (auto& r : reg_id_to_context_type) {
    str << indent(n + 1) << r.first << " => " << r.second << '\n';
  }
  auto reg_id_to_context_type_v2 =
      s->get_attr<std::map<std::string, std::string>>(
          "reg_id_to_context_type_v2");
  str << indent(n + 1) << "reg_id_to_context_type_v2:" << '\n';
  for (auto& r : reg_id_to_context_type_v2) {
    str << indent(n + 1) << r.first << " => " << r.second << '\n';
  }
  auto reg_id_to_hw_segment =
      s->get_attr<std::map<std::string, std::string>>("reg_id_to_hw_segment");
  str << indent(n + 1) << "reg_id_to_hw_segment:" << '\n';
  for (auto& r : reg_id_to_hw_segment) {
    str << indent(n + 1) << r.first << " => " << r.second << '\n';
  }
  auto reg_id_to_parameter_value =
      s->get_attr<std::map<std::string, std::vector<char>>>(
          "reg_id_to_parameter_value");
  str << indent(n + 1) << "reg_id_to_parameter_value:" << '\n';
  for (auto& r : reg_id_to_parameter_value) {
    str << indent(n + 1) << r.first << " => " << r.second.size()
        << " bytes md5sum= "
        << xir::get_md5_of_buffer(&r.second[0], r.second.size()) << '\n';
  }
  auto reg_id_to_size =
      s->get_attr<std::map<std::string, int32_t>>("reg_id_to_size");
  str << indent(n + 1) << "reg_id_to_size:" << '\n';
  for (auto& r : reg_id_to_size) {
    str << indent(n + 1) << r.first << " => " << r.second << '\n';
  }
  return str.str();
};

static void show_subgraph(const xir::Subgraph* s, int n) {
  auto device = get_device(s);
  if (device == "DPU") {
    cout << indent(n) << s->get_name() << '\n';
    cout << reg_info(s, n) << "\n";
  }
  auto children = s->children_topological_sort();
  for (auto c : children) {
    show_subgraph(c, n + 1);
  }
  return;
}

int CmdDumpReg::main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << help() << endl;
    return 1;
  }
  auto xmodel = std::string(argv[1]);
  auto graph = xir::Graph::deserialize(xmodel);
  auto root = graph->get_root_subgraph();
  show_subgraph(root, 0);
  return 0;
}

std::string CmdDumpReg::help() const {
  std::ostringstream str;
  str << "xir " << get_name() << " <xmodel>\n\t"
      << "show reg info for dpu suggraph.";
  return str.str();
}
