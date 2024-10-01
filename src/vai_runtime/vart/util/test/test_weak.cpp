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
#include <memory>
#include "vitis/ai/weak.hpp"
using namespace std;

struct A {
  A(int x) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << " create A() with x=" << x << endl;
  }
};

struct B {
  static unique_ptr<B> create(int x) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << " create A() with x=" << x << endl;
    return std::make_unique<B>(x);
  }
  B(int x) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << " create B() with x=" << x << endl;
  }
};
int main(int argc, char* argv[]) {
  int x = 10;
  auto a0 = vitis::ai::WeakStore<int, A>::create(x, x);
  auto a1 = vitis::ai::WeakStore<int, A>::create(x, x);
  std::cerr << "a0.get() " << (void*)a0.get() << " "  //
            << "a1.get() " << (void*)a1.get() << " "  //
            << std::endl;
  auto b = vitis::ai::WeakStore<int, B>::create(x, x);
  return 0;
}
