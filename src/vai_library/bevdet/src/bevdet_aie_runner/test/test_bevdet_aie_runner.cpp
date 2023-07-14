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
using namespace std;
#include <chrono>
#include <thread>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>

#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/profiling.hpp"
#include "vitis/ai/thread_pool.hpp"
#include "xir/tensor/tensor.hpp"

int main(int argc, char* argv[]) {
  // ifstream fin0("./data/data0.bin", ios::binary);
  // ifstream fin1("./data/data1.bin", ios::binary);
  // ifstream fin2("./data/data2.bin", ios::binary);
  // ifstream fin3("./data/data3.bin", ios::binary);
  // ofstream fout("./data/hwout.bin", ios::binary);
  vector<string> fin;
  if (argc > 6) {
    LOG(INFO) << "argc " << argc;
    fin = vector<string>{argv[2], argv[3], argv[4], argv[5], argv[6]};
  }
  // for (auto f : fin) f->seekg(0, ios::beg);

  auto graph = xir::Graph::deserialize(argv[1]);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  for (auto c : root->get_children()) {
    if (c->get_attr<std::string>("device") == "DPU") {
      s = c;
      break;
    }
  }
  auto attrs = xir::Attrs::create();
  attrs->set_attr("xclbin", "/run/media/mmcblk0p1/dpu.xclbin");
  attrs->set_attr("lib", std::map<std::string, std::string>{
                             {"DPU", "libbevdet_aie_runner.so"},
                             {"CPU", "libbevdet_aie_runner.so"}});

  auto runner = vart::Runner::create_runner_with_attrs(s, attrs.get());

  auto inputs =
      vart::alloc_cpu_flat_tensor_buffers(runner->get_input_tensors());
  auto outputs =
      vart::alloc_cpu_flat_tensor_buffers(runner->get_output_tensors());

  uint64_t data, tensor_size;
  __TIC__(aie)

  for (int i = 0; i < inputs.size(); i++) {
    std::tie(data, tensor_size) = inputs[i]->data({0, 0, 0, 0});
    //
    if (fin.size() == inputs.size()) {
      CHECK(std::ifstream(fin[i], std::ios::binary)
                .read((char*)data, tensor_size)
                .good())
          << "fail to read! filename=" << fin[i];
    } else {
      std::vector<char> tmp(tensor_size);
      LOG(INFO) << inputs[i]->get_tensor()->get_name() << " " << tensor_size;
      std::generate(tmp.begin(), tmp.end(),
                    [&] { return (char)(rand() % 255); });
      memcpy((char*)data, tmp.data(), tensor_size);
    }
  }

  auto job = runner->execute_async(vitis::ai::vector_unique_ptr_get(inputs),
                                   vitis::ai::vector_unique_ptr_get(outputs));

  if (job.second == 0) {
    auto value = runner->wait((int)job.first, -1);
    LOG_IF(INFO, false) << "runner return value=" << value;
  } else {
    LOG(ERROR) << " cannot create job "
               << " job_id=" << job.first << " status=" << job.second;
  }

  std::tie(data, tensor_size) = outputs[0]->data({0, 0, 0, 0});
  cout << "output size: " << tensor_size << endl;
  // fout.write((char*)data, tensor_size);
  __TOC__(aie)

  return 0;
}
