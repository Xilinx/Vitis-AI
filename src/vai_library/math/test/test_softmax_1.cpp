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

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vitis/ai/math.hpp"
using namespace std;

extern int GLOBAL_ENABLE_C_SOFTMAX;
int main(int argc, char *argv[]) {
  char *file = argv[1];
  int fixpos = stoi(argv[2]);
  int cls = stoi(argv[3]);
  int group = stoi(argv[4]);
  float scale = std::exp2f(-1.0f * (float)fixpos);
  vector<int8_t> input(cls * group);
  vector<float> output1(cls * group);
  vector<float> output2(cls * group);
  CHECK(std::ifstream(file)
            .read(reinterpret_cast<char *>(&input[0]), input.size())
            .good())
      << "cannot read file " << file << " "
      << "cls " << cls << " "      //
      << "group " << group << " "  //
      << "scale " << scale << " "  //
      ;

  vitis::ai::softmax(&input[0], scale, cls, group, &output1[0]);
  GLOBAL_ENABLE_C_SOFTMAX = 2;
  vitis::ai::softmax(&input[0], scale, cls, group, &output2[0]);
  for (auto i = 0; i < cls; ++i) {
    cout << (int)input[i] << ": " << output1[i] << " " << output2[i] << " "
         << output1[i] - output2[i] << endl;
  }
  return 0;
}
