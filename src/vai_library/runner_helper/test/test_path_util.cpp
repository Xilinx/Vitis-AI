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
#include <vector>

#include "xir/tensor/tensor.hpp"
using namespace std;

#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/path_util.hpp"
DEF_ENV_PARAM(DEBUG_RUNNER_HELPER, "0")
int main(int argc, char* argv[]) {
  vector<string> p1{"/", "abc/", "a/b/c", "a/b/c/"};
  for (auto p : p1) {
    cout << "p=" << p
         << " file_name_directory(p)=" << vitis::ai::file_name_directory(p)
         << endl;
  }
  vector<string> p2{".", ".."};
  for (auto p : p2) {
    cout << "p=" << p                                                      //
         << " file_name_realpath(p)=" << vitis::ai::file_name_realpath(p)  //
         << endl;
  }
  vector<string> p3{".", "..", "/tmp/a.txt"};
  for (auto p : p3) {
    cout << "p=" << p                                          //
         << " is_directory(p)=" << vitis::ai::is_directory(p)  //
         << " is_regular_file(p)="                             //
         << vitis::ai::is_regular_file(p)                      //
         << endl;
  }
  vector<string> p4{".", "..", "/tmp/a.txt", "./a.txt", "./abc"};
  for (auto p : p4) {
    cout << "p=" << p                                                      //
         << " file_name_basename(p)=" << vitis::ai::file_name_basename(p)  //
         << " file_name_basename_no_ext(p)="
         << vitis::ai::file_name_basename_no_ext(p)              //
         << " file_name_ext(p)=" << vitis::ai::file_name_ext(p)  //
         << endl;
  }
  vector<string> p5{".", "..", "./a/b/c", "./b/e/f/"};
  for (auto& p : p5) {
    vitis::ai::create_parent_path(p);
    cout << "p=" << p                                          //
         << " is_directory(p)=" << vitis::ai::is_directory(p)  //
         << endl;
  }
  auto tensor =
      xir::Tensor::create("hello()/abc/def(fixed)[]:hello", {2, 4},
                          xir::DataType{xir::DataType::Type::FLOAT, 32});
  auto tensor_buffer = vart::alloc_cpu_flat_tensor_buffer(tensor.get());
  ENV_PARAM(DEBUG_RUNNER_HELPER) = 1;
  vart::dump_tensor_buffer("a/b/c", tensor_buffer.get());

  for (std::string p : {"_/():[]{}\\?%*|\"'><;=", "hello", "goodname()_:?"}) {
    cout << "p=" << p
         << " to_valid_file_name(p)= " << vitis::ai::to_valid_file_name(p)
         << endl;
  }
  return 0;
}
