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
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "vitis/ai/dpu_task.hpp"
using namespace std;

int main(int argc, char* argv[]) {
  auto model_name = std::string(argv[1]);
  auto input_file = std::string(argv[2]);
  auto dpu_task = vitis::ai::DpuTask::create(model_name);
  if (!dpu_task) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }   
  auto inputs = dpu_task->getInputTensor(0u);
  CHECK_EQ(inputs.size(), 1u);
  CHECK(std::ifstream(input_file)
            .read(static_cast<char*>(inputs[0].get_data(0)), inputs[0].size)
            .good());
  dpu_task->run(0u);
  std::cout << model_name << " run success" << std::endl;
  return 0;
}
