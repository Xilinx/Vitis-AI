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
 */
#include <iostream>

#include "vart/zero_copy_helper.hpp"
#include "xir/graph/graph.hpp"
using namespace std;

static std::string to_string(const std::vector<size_t>& v) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto& x : v) {
    if (c++ != 0) {
      str << ",";
    }
    str << x;
  }
  str << "]";
  return str.str();
}
int main(int argc, char* argv[]) {
  auto graph = xir::Graph::deserialize(argv[1]);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  for (auto c : root->get_children()) {
    if (c->get_attr<std::string>("device") == "DPU") {
      s = c;
      break;
    }
  }
  auto input_tensor_buffer_size = vart::get_input_buffer_size(s);
  auto input_offset = vart::get_input_offset(s);
  auto output_tensor_buffer_size = vart::get_output_buffer_size(s);
  auto output_offset = vart::get_output_offset(s);
  LOG(INFO) << "input_tensor_buffer_size " << input_tensor_buffer_size
            << " "                                                //
            << "input offset " << to_string(input_offset) << " "  //
            << "output_tensor_buffer_size " << output_tensor_buffer_size
            << " "                                                  //
            << "output offset " << to_string(output_offset) << " "  //
      ;
  return 0;
}
