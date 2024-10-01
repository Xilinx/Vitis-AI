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

#include "util.hpp"
#include <fstream>
#include <string>
#include <vitis/ai/event.hpp>

// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
using std::cout;
using std::endl;
using std::ofstream;
using std::vector;
using std::string;
namespace vitis::ai::trace {
void dump_map(trace_entry_t& m) {
  for (const auto& i : m) {
    std::cout << i.first << ':' << i.second << "; ";
  }
  cout << endl;
};

void dump_to(vector<trace_entry_t>& trace_data, string file_path) {
  ofstream output_file;
  output_file.open(file_path, std::ios::out);

  if (output_file.is_open()) {
    for (const auto& entry : trace_data) {
      for (const auto& i : entry) {
        output_file << i.first << ':' << i.second << ";";
      }
      output_file << endl;
    }
    output_file.close();
  }
}
}  // namespace vitis::ai::trace
