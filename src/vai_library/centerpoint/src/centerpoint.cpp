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

#include "vitis/ai/centerpoint.hpp"
#include "./centerpoint_imp.hpp"

namespace vitis { namespace ai {

CenterPoint::CenterPoint(){}
CenterPoint::~CenterPoint(){}

std::unique_ptr<CenterPoint>
CenterPoint::create(const std::string &model_name_0, const std::string &model_name_1){
  return std::unique_ptr<CenterPoint>(new CenterPointImp(model_name_0, model_name_1));
}

}}

