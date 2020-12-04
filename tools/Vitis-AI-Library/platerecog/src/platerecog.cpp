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
#include <vitis/ai/platerecog.hpp>

#include "./platerecog_imp.hpp"

namespace vitis {
namespace ai {
PlateRecog::PlateRecog() {}
PlateRecog::~PlateRecog() {}

std::unique_ptr<PlateRecog> PlateRecog::create(
    const std::string &platedetect_model, const std::string &platerecog_model,
    bool need_preprocess) {
  return std::unique_ptr<PlateRecog>(
      new PlateRecogImp(platedetect_model, platerecog_model, need_preprocess));
}
std::unique_ptr<PlateRecog> PlateRecog::create(
    const std::string &platedetect_model, const std::string &platerecog_model,
    xir::Attrs *attrs, bool need_preprocess) {
  return std::unique_ptr<PlateRecog>(
      new PlateRecogImp(platedetect_model, platerecog_model, attrs, need_preprocess));
}
}  // namespace ai
}  // namespace vitis
