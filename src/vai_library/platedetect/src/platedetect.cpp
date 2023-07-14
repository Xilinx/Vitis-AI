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
#include <vitis/ai/platedetect.hpp>

#include "./platedetect_imp.hpp"
namespace vitis {
namespace ai {

PlateDetect::PlateDetect() {}
PlateDetect::~PlateDetect() {}

std::unique_ptr<PlateDetect> PlateDetect::create(const std::string &model_name,
                                                 bool need_preprocess) {
  return std::unique_ptr<PlateDetect>(
      new PlateDetectImp(model_name, need_preprocess));
}
std::unique_ptr<PlateDetect> PlateDetect::create(const std::string &model_name,
                                                 xir::Attrs *attrs,
                                                 bool need_preprocess) {
  return std::unique_ptr<PlateDetect>(
      new PlateDetectImp(model_name, attrs, need_preprocess));
}

}  // namespace ai
}  // namespace vitis
