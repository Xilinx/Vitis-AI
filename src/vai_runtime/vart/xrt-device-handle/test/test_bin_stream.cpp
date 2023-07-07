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

#include <iostream>
#include <memory>
// using namespace std;
// we cannot use namespace std if we include "xrt.h", because it indirectly
// include windows.h and results in many compilation errors
#include <vitis/ai/env_config.hpp>

#include "../src/xrt_bin_stream.hpp"
DEF_ENV_PARAM(BURN_BIT_STREAM, "0");
int main(int argc, char* argv[]) {
  auto filename = std::string(argv[1]);
  auto h = std::make_unique<xir::XrtBinStream>(filename);
  h->dump_layout();
  h->dump_mem_topology();
  if (ENV_PARAM(BURN_BIT_STREAM)) {
    LOG(INFO) << "start to upload xclbin." << filename;
    h->burn();
  }
  LOG(INFO) << "num of cu:" << h->get_num_of_cu();
  for (auto i = 0u; i < h->get_num_of_cu(); ++i) {
    LOG(INFO) << "cu[" << i << "] =" << h->get_cu(i);
  }
  return 0;
}
