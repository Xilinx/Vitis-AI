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
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

#include "vitis/ai/parse_value.hpp"
#include "vitis/ai/profiling.hpp"
#include "xir/sfm_controller.hpp"

using namespace std;
DEF_ENV_PARAM(DEBUG_TEST, "0");

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      float* output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp(input[i] * scale);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) output[i] /= sum;
}

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    input += cls;
    output += cls;
  }
}

static void compare(int cls, int group, signed char* input, float* output1,
                    float* output2) {
  for (auto g = 0; g < group; ++g) {
    for (auto i = 0; i < cls; ++i) {
      auto idx = g * cls + i;
      auto diff = output1[idx] - output2[idx];
      if (ENV_PARAM(DEBUG_TEST) || (diff != 0.0 && std::abs(diff) > 0.001)) {
        cout << " i=" << i               //
             << " g = " << g             //
             << " idx = " << idx << " "  //
             << (int)input[idx] << ": " << output1[idx] << " " << output2[idx]
             << " " << diff << endl;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  auto sfm = xir::SfmController::get_instance();
  if (argc < 5) {
    cout << "usage: " << argv[0] << "<file> <fixpos> <cls> <group>" << endl;
    return 0;
  }
  char* file = argv[1];
  int fixpos = stoi(argv[2]);
  int cls = stoi(argv[3]);
  int group = stoi(argv[4]);
  float scale = std::exp2f(-1.0f * (float)fixpos);
  vector<int8_t> input(cls * group);
  vector<float> output1(cls * group);
  vector<float> output2(cls * group);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << " fixpos=" << fixpos << " cls=" << cls << " group=" << group
      << " scale=" << scale;
  CHECK(std::ifstream(file)
            .read(reinterpret_cast<char*>(&input[0]), input.size())
            .good())
      << "cannot read file " << file << " "
      << "cls " << cls << " "      //
      << "group " << group << " "  //
      << "scale " << scale << " "  //
      ;

  __TIC__(sfmx);
  sfm->run(&input[0], scale, cls, group, &output1[0]);
  __TOC__(sfmx);
  softmax_c(&input[0], scale, cls, group, &output2[0]);
  compare(cls, group, &input[0], &output1[0], &output2[0]);
  return 0;
}
