/*
 * Copyright 2019 xilinx Inc.
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
#include "vitis/ai/bevdet.hpp"

#include <string>

#include "./bevdet_imp.hpp"

namespace vitis {
namespace ai {
BEVdet::BEVdet() {}
BEVdet::~BEVdet() {}

std::unique_ptr<BEVdet> BEVdet::create(const std::string& model_name0,
                                       const std::string& model_name1,
                                       const std::string& model_name2) {
  return std::unique_ptr<BEVdet>(
      new BEVdetImp(model_name0, model_name1, model_name2));
}
}  // namespace ai
}  // namespace vitis
