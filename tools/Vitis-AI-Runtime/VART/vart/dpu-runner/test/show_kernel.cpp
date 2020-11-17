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

#include <glog/logging.h>

#include <iostream>
#include <xir/graph/graph.hpp>
using namespace std;
extern "C" const char* xilinx_xir_version();
template <typename T>
static inline std::ostream& operator<<(std::ostream& out,
                                       const std::vector<T>& v) {
  int c = 0;
  out << "[";
  for (const auto x : v) {
    if (c++ != 0) {
      out << ",";
    }
    out << x;
  }
  out << "]";
  return out;
}
static std::ostream& operator<<(std::ostream& out,
                                const xir::Tensor* xir_tensor) {
  out << "xir_tensor{";
  out << xir_tensor->get_name() << ":(";
  int fixpos = xir_tensor->template get_attr<int>("fix_point");
  auto dims = xir_tensor->get_shape();
  for (auto i = 0u; i < dims.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << dims[i];
  }
  out << "), fixpos=" << fixpos
      << " # of elements= " << xir_tensor->get_element_num();
  out << "}";
  return out;
}

int main(int argc, char* argv[]) {
  auto filename = argv[1];
  auto graph = xir::Graph::deserialize(filename);
  cout << "libxir.so: " << xilinx_xir_version() << endl;
  if (graph->has_attr("origin")) {
    cout << "graph name: '" << graph->get_name() << "' compiled from "
         << graph->get_attr<string>("origin") << " model." << endl;
  }
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  if (children.empty()) {
    cout << "no subgraph" << endl;
  }
  if (graph->has_attr("files_md5sum")) {
    for (auto md5 :
         graph->get_attr<std::map<std::string, std::string>>("files_md5sum")) {
      cout << "FILES MD5: " << md5.second << " " << md5.first << endl;
    }
  }
  if (graph->has_attr("tools_commit_id")) {
    for (auto commit_id :
         graph->get_attr<std::map<std::string, std::string>>("tools_commit_id")) {
      cout << "COMMIT IDS: " << commit_id.second << " " << commit_id.first
           << endl;
    }
  }
  for (auto c : children) {
    CHECK(c->has_attr("device"));
    auto device = c->get_attr<std::string>("device");
    cout << "device = " << device << " kernel: '" << c->get_name() << "' ";
    if (device == "DPU") {
      if (c->has_attr("dpu_fingerprint")) {
        auto fingerprint = c->get_attr<std::uint64_t>("dpu_fingerprint");
        std::cout << "dpu_fingerprint " << fingerprint << " "  //
            ;
      }

      if (c->has_attr("dpu_name")) {
        auto name = c->get_attr<std::string>("dpu_name");
        std::cout << "dpu_name " << name << " "  //
                  << std::endl;
        ;
      }
      auto inputs = c->get_input_tensors();
      for (auto input : inputs) {
        std::cout << "\tinput: " << input << std::endl;
      }

      auto outputs = c->get_output_tensors();
      for (auto output : outputs) {
        std::cout << "\toutputs: " << output << std::endl;
      }
    }
    cout << endl;
  }

  return 0;
}
