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

#include "./cmd_subgraph.hpp"

#include <glog/logging.h>

#include <iostream>
#include <sstream>
#include <xir/graph/graph.hpp>
using namespace std;
static std::string indent(size_t n) { return std::string(n * 4u, ' '); }
static std::string get_device(const xir::Subgraph* s) {
  auto ret = std::string("");
  if (s->has_attr("device")) {
    ret = s->get_attr<std::string>("device");
  }
  return ret;
}
static uint64_t get_fingerprint(const xir::Subgraph* c) {
  if (c->has_attr("dpu_fingerprint")) {
    return c->get_attr<std::uint64_t>("dpu_fingerprint");
  }
  return 0u;
}

static void show_subgraph(const xir::Subgraph* s, int n) {
  auto device = get_device(s);
  cout << indent(n) << s->get_name();
  if (!device.empty()) {
    cout << " [";
    cout << "device=" << device;
    if (device == "DPU") {
      cout << ",fingerprint="
           << "0x" << std::hex << get_fingerprint(s) << std::dec;
      cout << ",DPU=" << s->get_attr<std::string>("dpu_name");
      cout << ",I=" << s->get_input_tensors();
      cout << ",O=" << s->get_output_tensors();
    }
    cout << "]";
  }
  cout << "\n";
  auto children = s->children_topological_sort();
  for (auto c : children) {
    show_subgraph(c, n + 1);
  }
  return;
}

CmdSubgraph::CmdSubgraph(const std::string& name) : Cmd(name) {}

int CmdSubgraph::main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << help() << endl;
    return 1;
  }
  auto filename = argv[1];
  auto graph = xir::Graph::deserialize(filename);
  auto root = graph->get_root_subgraph();
  show_subgraph(root, 0);
  return 0;
}

std::string CmdSubgraph::help() const {
  std::ostringstream str;
  str << "xir " << get_name() << " <xmodel>\n\t"
      << "show subgraph tree" << endl;
  return str.str();
}
