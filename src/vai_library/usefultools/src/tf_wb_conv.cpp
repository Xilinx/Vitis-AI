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
using namespace std;
int main(int argc, char* argv[]) {
  auto height = 1;  // stoi(string(argv[1]));
  auto width = stoi(string(argv[1]));
  auto channel_in = stoi(string(argv[2]));
  auto channel_out = stoi(string(argv[3]));
  auto input_file = string(argv[4]);
  auto sz = (size_t)(height * width * channel_in * channel_out);
  auto data_in = vector<char>(sz);
  auto data_out = vector<char>(sz);
  int c = 0;
  CHECK(std::ifstream(input_file).read(&data_in[0], sz).good());
  for (auto co = 0; co < channel_out; ++co) {
    for (auto h = 0; h < height; ++h) {
      for (auto w = 0; w < width; ++w) {
        for (auto ci = 0; ci < channel_in; ++ci) {
          auto value = data_in[h * width * channel_in * channel_out +  //
                               w * channel_in * channel_out +          //
                               ci * channel_out +                      //
                               co];
          // scale;
          // CHECK_LE(value, -128.0f);
          // value = std::max(-128.0f, value);
          // value = std::min(127.0f, value);
          // char c_value = (char)value;
          data_out[c++] = value;
        }
      }
    }
  }
  CHECK(cout.write(&data_out[0], sz).good());
  return 0;
}
