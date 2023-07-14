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

#include <algorithm>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./sigmoid_table.hpp"
#include "./utils.hpp"

DEF_ENV_PARAM(DEBUG_SIGMOID_TABLE, "0")

namespace vitis { namespace ai {
namespace pointpillars_nus {

static std::vector<float> sigmoid_table(int fixpos) {
  std::vector<float> list(256);
  auto e2 = std::exp2((-1) * fixpos);
  for (auto i = 0u; i < 256; ++i) {
    list[i] = (-128 + (int)i) * e2; 
  }
  std::vector<float> table(256, 0);
  sigmoid(list.data(), table.data(), 256); 
  if (ENV_PARAM(DEBUG_SIGMOID_TABLE)) {
    for (auto i = 0u; i < 256; ++i) {
      LOG(INFO) << "table index:" << i
                << " value:" << table[i];
    }
  }
  return table;
}

// 
// sigmoid = 1 / (1 + e^(-x))
// x : [-128, 127] * 2^ (-1 * fixpos)
//

static void sigmoid_table_internal(const int8_t * input, int fixpos, unsigned int cls, 
                        unsigned int group, float *output) {
  auto size = cls * group;
__TIC__(SIGMOID_TABLE_INIT)
  auto table = sigmoid_table(fixpos);
__TOC__(SIGMOID_TABLE_INIT)
  for (auto i = 0u; i < size; ++i) {
    output[i] = table[input[i] + 128];
  }
}

void sigmoid_table(const int8_t * input, int fixpos, unsigned int cls, 
                        unsigned int group, float *output) {
  sigmoid_table_internal(input, fixpos, cls, group, output);
}

}}}
