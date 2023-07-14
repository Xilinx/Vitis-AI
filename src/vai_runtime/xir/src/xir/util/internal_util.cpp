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

#include "internal_util.hpp"

#include <cfenv>
#include <cmath>

namespace xir {
namespace internal {

// op related
std::vector<const Op*> vec_input_ops(
    const std::map<std::string, std::vector<const Op*>>& input_ops) {
  std::vector<const Op*> ret;
  for (auto& it : input_ops) {
    auto ops = it.second;
    ret.insert(ret.end(), ops.begin(), ops.end());
  }
  return ret;
}

std::vector<Op*> vec_input_ops(
    const std::map<std::string, std::vector<Op*>>& input_ops) {
  std::vector<Op*> ret;
  for (auto& it : input_ops) {
    auto ops = it.second;
    ret.insert(ret.end(), ops.begin(), ops.end());
  }
  return ret;
}

// round related
float dpu_round_float(const float& input) {
  float ret = input;
  if (ret >= 0) {
    ret = std::round(ret);
  } else {
    float delta = ret - std::floor(ret);
    if (delta == 0.5f) {
      ret = std::ceil(ret);
    } else {
      ret = std::round(ret);
    }
  }
  return ret;
}

float py3_round_float(const float& input) {
  float ret;
#ifdef __microblaze__
  // warning: feget round is not implemented and will always fail
  ret = std::nearbyint(input);
#else
  auto default_round_mode = std::fegetround();
  if (!(FE_TONEAREST == default_round_mode)) {
    std::fesetround(FE_TONEAREST);
    ret = std::nearbyint(input);
    std::fesetround(default_round_mode);
  } else {
    ret = std::nearbyint(input);
  }
#endif
  return ret;
}

}  // namespace internal
}  // namespace xir
