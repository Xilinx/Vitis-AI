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
#include "vitis/ai/polypsegmentation.hpp"

#include <string>

#include "./polypsegmentation_imp.hpp"

namespace vitis {
namespace ai {
PolypSegmentation::PolypSegmentation(const std::string& model_name,
                                     bool need_preprocess)
    : ConfigurableDpuTaskBase(model_name, need_preprocess) {}
PolypSegmentation::~PolypSegmentation() {}

std::unique_ptr<PolypSegmentation> PolypSegmentation::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<PolypSegmentation>(
      new PolypSegmentationImp(model_name, need_preprocess));
}
}  // namespace ai
}  // namespace vitis
