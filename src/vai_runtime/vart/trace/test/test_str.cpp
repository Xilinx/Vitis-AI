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
#include "str.hpp"

int main(void) {
  auto a = vitis::ai::trace::str("aaa");
  auto b = vitis::ai::trace::str("alsdfkajsdf");
  auto c = vitis::ai::trace::str("777");
  auto d = vitis::ai::trace::str("aaa");

  std::cout << "pool size: " << vitis::ai::trace::str_pool_size() << std::endl;
  std::cout << "size: " << sizeof(a) << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << a.to_string() << std::endl;
  std::cout << b.to_string() << std::endl;
  std::cout << c.to_string() << std::endl;
}
