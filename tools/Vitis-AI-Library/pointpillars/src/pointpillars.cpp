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

#include "vitis/ai/pointpillars.hpp"
#include "./pointpillars_imp.hpp"

namespace vitis { namespace ai {

PointPillars::PointPillars(){}
PointPillars::~PointPillars(){}

std::unique_ptr<PointPillars> PointPillars::create(const std::string &model_name, const std::string&model_name1){
  return std::unique_ptr<PointPillars>(new PointPillarsImp(model_name, model_name1 ));
}

}}

