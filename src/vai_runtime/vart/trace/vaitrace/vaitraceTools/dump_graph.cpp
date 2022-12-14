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

#include <iostream>
#include <string>
#include <xir/graph/graph.hpp>

using namespace std;

int main(int argc, char* argv[]) {
  auto filename = argv[1];

  auto graph = xir::Graph::deserialize(filename);

  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();

  cout << "[";
  for (auto c : children) {
    auto device = c->get_attr<std::string>("device");
    auto name = c->get_name();

    cout << "{" << endl;
    cout << "\"kernelName\": "
         << "\"" << name << "\"," << endl
         << "\"kernelDev\": "
         << "\"" << device << "\"," << endl;

    auto inputs = c->get_input_tensors();
    cout << "\"inputs\": [";
    for (auto input : inputs) {
      cout << "\"" << input->get_name() << "\""
           << ",";
    }
    cout << "\"\"], " << endl;

    auto outputs = c->get_output_tensors();
    cout << "\"outputs\": [";
    for (auto output : outputs) {
      cout << "\"" << output->get_name() << "\""
           << ",";
    }
    cout << "\"\"]," << endl;

    auto cc = c->get_children();
    cout << "\"subgraph\": [" << endl;
    for (auto csub : cc) {
      cout << "\t\"" << csub->get_name() << "\"," << endl;
    }
    cout << "\"\"]}," << endl;
    cout << endl;
  }

  // End of Kernel
  cout << "{}]" << endl;

  return 0;
}
