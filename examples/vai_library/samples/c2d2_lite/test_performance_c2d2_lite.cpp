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
#include <vitis/ai/c2d2_lite.hpp>

#include "./c2d2_benchmark.hpp"
int main(int argc, char* argv[]) {
  std::string model0 = argv[1];
  std::string model1 = argv[2];
  return vitis::ai::main_for_performance(argc, argv, [&] {
    { return vitis::ai::C2D2_lite::create(model0, model1); }
  });
}
