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

#include "../src/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/dpu_runner.hpp"
#include "vitis/ai/tensor_buffer.hpp"

using namespace std;

void show_input_output_tensors(
    const std::vector<std::vector<std::unique_ptr<vitis::ai::DpuRunner>>>&
        runners) {
  if (!runners.empty()) {
    auto& r = runners[0];
    cout << "num of runners: " << r.size() << endl;
    if (!r.empty()) {
      auto& runner = r[0];
      auto input_tensors = runner->get_input_tensors();
      auto output_tensors = runner->get_output_tensors();
      int c = 0;
      c = 0;
      for (auto t : input_tensors) {
        LOG(INFO) << "input[" << c++ << "]:" << t->to_string() << endl;
      }
      c = 0;
      for (auto t : output_tensors) {
        LOG(INFO) << "output[" << c++ << "]:" << t->to_string() << endl;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    cout << "usage: test_runner <xmodel> <input> <num_of_runners> <count>"
         << endl;
    return 0;
  }
  auto dirname = std::string(argv[1]);
  auto input_file = std::string(argv[2]);
  auto runner_num = std::stoi(std::string(argv[3]));
  auto count = std::stoi(std::string(argv[4]));
  auto runners =
      std::vector<std::vector<std::unique_ptr<vitis::ai::DpuRunner>>>(
          runner_num);
  for (auto rr = 0; rr < runner_num; rr++) {
    runners[rr] = vitis::ai::DpuRunner::create_dpu_runner(dirname);
  }
  show_input_output_tensors(runners);
  for (auto rr = 0; rr < runner_num; rr++) {
    CHECK_EQ(runners[rr].size(), 1u) << "only single subgraph is supported";
    auto& runner = runners[rr][0];
    auto input =
        vitis::ai::alloc_cpu_flat_tensor_buffers(runner->get_input_tensors());
    CHECK_EQ(input.size(), 1u) << "only support single input yet.";
    auto output =
        vitis::ai::alloc_cpu_flat_tensor_buffers(runner->get_output_tensors());
    void* input_data = 0u;
    auto input_size = 0u;
    size_t batch_size = input[0]->get_tensor()->get_dims()[0];
    auto size_per_batch = input[0]->get_tensor()->get_element_num() *
                          size_of(input[0]->get_tensor()->get_data_type()) /
                          batch_size;
    for (auto i = 0u; i < batch_size; ++i) {
      std::tie(input_data, input_size) = input[0]->data({(int)i, 0, 0, 0});
      LOG(INFO) << "input[0]->get_tensor()->get_dims()[0] "
                << input[0]->get_tensor()->get_dims()[0] << " "  //
          ;
      CHECK(std::ifstream(input_file)
                .read((char*)input_data, size_per_batch)
                .good())
          << "fail to read! filename=" << input_file;
      if (0) {
        auto mode =
            std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
        CHECK(std::ofstream(std::string("input_") + std::to_string(i) +
                                std::string(".bin"),
                            mode)
                  .write((char*)input_data, size_per_batch)
                  .good())
            << " faild to write";
      }
    }

    for (auto i = 0; i < count; ++i) {
      LOG(INFO) << "count " << i << " ";  //
      auto job =
          runner->execute_async(vitis::ai::vector_unique_ptr_get(input),
                                vitis::ai::vector_unique_ptr_get(output));
      runner->wait((int)job.first, -1);
    }
  }
  LOG(INFO) << "BYEBYE";
  return 0;
}
