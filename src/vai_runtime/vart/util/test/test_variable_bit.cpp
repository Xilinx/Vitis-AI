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

#include <iostream>
#include <memory>
#include <vector>

#include "vitis/ai/variable_bit.hpp"

int main(int argc, char* argv[]) {
  auto input_data = std::vector<unsigned char>(500);
  for (auto i = 0u; i < 500u; i++) {
    input_data[i] = 0xa5;
  }
  auto bit_width = 10u;
  auto it = vitis::ai::VariableBitIterator(&input_data[0], bit_width, 0, 0);
  for (auto i = 0u; i < 50u; i++) {
    it.set(i + 1);
    it = it.next();
  }

  auto view = vitis::ai::VariableBitView(&input_data[0], bit_width, 10);

  for (auto x : view) {
    std::cout << "0x" << std::hex << x << std::endl;
  }
  /*for (auto b = view.begin(), e = view.end(); b != e; ++b) {
    auto x = *b;
    std::cout << "b = " << b.to_string() << std::endl;
    std::cout << "0x" << std::hex << x << std::endl;
    if (c++ > 120) {
      break;
    }
    }*/
  return 0;
}
