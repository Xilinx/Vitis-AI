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

#include "vart/xir_helper.hpp"

namespace vart {
// op related
std::vector<const xir::Op*> vec_input_ops(
    const std::map<std::string, std::vector<const xir::Op*>>& input_ops) {
  std::vector<const xir::Op*> ret;
  for (auto& it : input_ops) {
    auto ops = it.second;
    ret.insert(ret.end(), ops.begin(), ops.end());
  }
  return ret;
}

std::vector<xir::Op*> vec_input_ops(
    const std::map<std::string, std::vector<xir::Op*>>& input_ops) {
  std::vector<xir::Op*> ret;
  for (auto& it : input_ops) {
    auto ops = it.second;
    ret.insert(ret.end(), ops.begin(), ops.end());
  }
  return ret;
}
}  // namespace vart
