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
#include <iostream>
#include <memory>
#include <vitis/ai/dim_calc.hpp>
using namespace std;
ostream& operator<<(ostream& s, const std::vector<size_t>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}

int main(int argc, char* argv[]) {
  auto dims = std::vector<size_t>{7, 5, 3, 2};
  auto strides = std::vector<size_t>{60, 6, 2, 1};
  auto dim_calc = std::make_unique<vitis::ai::DimCalc>(dims, strides);
  auto idx = std::vector<size_t>(dims.size(), 0u);
  auto next_idx = std::vector<size_t>(dims.size(), 0u);
  auto sz = 0u;
  for (tie(next_idx, sz) = dim_calc->next(idx); sz > 0;
       idx = next_idx, tie(next_idx, sz) = dim_calc->next(idx)) {
    cout << "idx: " << idx << " next_idx: " << next_idx << " sz: " << sz
         << " offset: " << dim_calc->offset(idx) << endl;
  }
  return 0;
}
