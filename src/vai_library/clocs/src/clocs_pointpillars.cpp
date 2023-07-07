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

#include <memory>

#include "./clocs_pointpillars.hpp"
#include "./clocs_pointpillars_imp.hpp"

namespace vitis {
namespace ai {

ClocsPointPillars::ClocsPointPillars() {}
ClocsPointPillars::~ClocsPointPillars() {}

std::unique_ptr<ClocsPointPillars> ClocsPointPillars::create(
    const std::string& model_name_0, const std::string& model_name_1,
    bool need_preprocess) {
  return std::unique_ptr<ClocsPointPillars>(
      new ClocsPointPillarsImp(model_name_0, model_name_1, need_preprocess));
}

std::unique_ptr<ClocsPointPillars> ClocsPointPillars::create(
    const std::string& model_name_0, const std::string& model_name_1,
    xir::Attrs* attrs, bool need_preprocess) {
  return std::unique_ptr<ClocsPointPillars>(new ClocsPointPillarsImp(
      model_name_0, model_name_1, attrs, need_preprocess));
}
}  // namespace ai
}  // namespace vitis

