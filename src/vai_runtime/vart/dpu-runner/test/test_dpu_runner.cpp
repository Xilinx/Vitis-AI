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
#include <google/protobuf/message.h>

#include <fstream>
#include <iostream>
#include <xir/tensor/tensor.hpp>
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/runner_ext.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  auto filename = argv[1];
  auto kernel = argv[2];
  auto input_file = std::string(argv[3]);
  auto runner_num = std::stoi(std::string(argv[4]));
  auto count = std::stoi(std::string(argv[5]));
  auto runners = vector<std::unique_ptr<vart::Runner>>();
  for (auto rr = 0; rr < runner_num; rr++) {
    runners.emplace_back(
        vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, kernel));
  }
  for (auto rr = 0; rr < runner_num; rr++) {
    auto& runner = runners[rr];
    auto r = dynamic_cast<vart::RunnerExt*>(runner.get());
    auto input = r->get_inputs();
    CHECK_EQ(input.size(), 1u) << "only support single input yet.";
    auto output = r->get_outputs();
    uint64_t input_data = 0u;
    auto input_size = 0u;
    size_t batch_size = input[0]->get_tensor()->get_shape()[0];
    auto size_per_batch = input[0]->get_tensor()->get_data_size() / batch_size;
    for (auto i = 0u; i < batch_size; ++i) {
      std::tie(input_data, input_size) = input[0]->data({(int)i, 0, 0, 0});
      LOG(INFO) << "input_size " << input_size << " "  //
                << "input[0]: " << input[0]->to_string();
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
            << " faild to write to " << filename;
      }
    }

    for (auto i = 0; i < count; ++i) {
      for (auto in : input) {
        in->sync_for_write(0, in->get_tensor()->get_data_size() /
                                  in->get_tensor()->get_shape()[0]);
      }

      runner->execute_async(input, output);
      runner->wait(0, 0);
      for (auto out : output) {
        out->sync_for_read(0, out->get_tensor()->get_data_size() /
                                  out->get_tensor()->get_shape()[0]);
      }
    }
  }
  return 0;
}
