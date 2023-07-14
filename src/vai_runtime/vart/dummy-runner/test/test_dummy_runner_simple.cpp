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

#if _WIN32
#  pragma comment(linker, "/include:vart_dummy_runner_hook")
#endif
DEF_ENV_PARAM(NUM_OF_THREADS, "1")
DEF_ENV_PARAM(NUM_OF_REQUESTS, "10")
DEF_ENV_PARAM(NUM_OF_RUNNERS, "10")

int main(int argc, char* argv[]) {
  LOG(INFO) << "HELLO , testing is started ";
  //<< dummy_runner_hook;
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
  auto runner = vart::Runner::create_runner_with_attrs(s, attrs.get());
  return 0;
}
