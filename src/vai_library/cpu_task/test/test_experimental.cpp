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

#include "vart/op_imp.h"
struct X : public vart::experimental::OpImpBase {
  X(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {};
  int calculate(
      vart::simple_tensor_buffer_t<float> r,                   //
      vart::simple_tensor_buffer_t<float> a1,                  // required
      std::unique_ptr<vart::simple_tensor_buffer_t<float>> a2  // optional
  ) {
    return 0;
  }
};

int main(int argc, char* argv[]) {
  auto x = vart::experimental::make_vart_opt_imp<X>();
  std::cout << (void*)&x;
  return 0;
}
