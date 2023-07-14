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

#include <memory>

#include "./clocs_imp.hpp"
#include "vitis/ai/clocs.hpp"

namespace vitis {
namespace ai {

Clocs::Clocs() {}
Clocs::~Clocs() {}

std::unique_ptr<Clocs> Clocs::create(const std::string& yolo,
                                     const std::string& pointpillars_0,
                                     const std::string& pointpillars_1,
                                     const std::string& fusionnet,
                                     bool need_preprocess) {
  return std::unique_ptr<Clocs>(new ClocsImp(
      yolo, pointpillars_0, pointpillars_1, fusionnet, need_preprocess));
}

}  // namespace ai
}  // namespace vitis

