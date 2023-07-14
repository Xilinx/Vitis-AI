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

#include "../src/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/thread_pool.hpp"

DEF_ENV_PARAM(NUM_OF_THREADS, "1")
DEF_ENV_PARAM(NUM_OF_REQUESTS, "10")
DEF_ENV_PARAM(NUM_OF_RUNNERS, "10")

int main(int argc, char* argv[]) {
  LOG(INFO) << "HELLO , testing is started";
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
  attrs->set_attr("async", true);
  attrs->set_attr("num_of_dpu_runners", (size_t)ENV_PARAM(NUM_OF_RUNNERS));
  attrs->set_attr("lib", std::map<std::string, std::string>{
                             {"DPU", "libvart-dummy-runner.so"}});

  std::atomic<int> num_of_finished_tasks = 0;
  {
    auto pool2 = vitis::ai::ThreadPool::create(ENV_PARAM(NUM_OF_THREADS));
    auto runner = vart::Runner::create_runner_with_attrs(s, attrs.get());
    {
      cout << s->get_name() << endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));

      auto raw_pool2 = pool2.get();

      for (auto i = 0; i < ENV_PARAM(NUM_OF_REQUESTS); ++i) {
        auto inputs =
            vart::alloc_cpu_flat_tensor_buffers(runner->get_input_tensors());
        auto outputs =
            vart::alloc_cpu_flat_tensor_buffers(runner->get_output_tensors());
        auto job =
            runner->execute_async(vitis::ai::vector_unique_ptr_get(inputs),
                                  vitis::ai::vector_unique_ptr_get(outputs));
        if (job.second == 0) {
          // LOG(INFO) << "begin async wait";
          // TODO: MSVC does not support move lambda well
          /*Error
              C2280
             'std::unique_ptr<vart::TensorBuffer,std::default_delete<_Ty>>::unique_ptr(const
             std::unique_ptr<_Ty,std::default_delete<_Ty>> &)' : attempting to
             reference a deleted function test_dummy_runner C
              :\msvsn2017\VC\Tools\MSVC\14.14.26428\include \xmemory0 920
           */

         
		  raw_pool2->async([
            &runner, job, &num_of_finished_tasks, inputs2 = std::move(inputs),
            outputs2 = std::move(outputs)
          ]() mutable {
            LOG_IF(INFO, false)
                << "waiting for job "
                << " job_id=" << job.first << " status=" << job.second;
            auto value = runner->wait((int)job.first, -1);
            LOG_IF(INFO, false) << "runner return value=" << value;
            num_of_finished_tasks++;
          });
          LOG_IF(INFO, false) << "after async wait";
		  
        } else {
          LOG(ERROR) << " cannot create job "
                     << " job_id=" << job.first << " status=" << job.second;
        }
        LOG_IF(INFO, false)
            << "submission job "
            << " job_id=" << job.first << " status=" << job.second;
      }
    }
  }
  LOG(INFO) << "all tasks are submitted";
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  while (num_of_finished_tasks < ENV_PARAM(NUM_OF_REQUESTS)) {
    LOG(INFO) << "waiting for task finished. " << num_of_finished_tasks << "/"
              << ENV_PARAM(NUM_OF_REQUESTS);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  LOG(INFO) << "ZAI JIAN " << num_of_finished_tasks << "/"
            << ENV_PARAM(NUM_OF_REQUESTS);
  return 0;
}
