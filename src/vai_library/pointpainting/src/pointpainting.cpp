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

#include "vitis/ai/pointpainting.hpp"
#include "./pointpainting_imp.hpp"

namespace vitis { namespace ai {

PointPainting::PointPainting(){}
PointPainting::~PointPainting(){}

std::unique_ptr<PointPainting>
PointPainting::create(const std::string &seg_model_name,
                      const std::string &pp_model_name_0, 
                      const std::string &pp_model_name_1,
                      bool need_preprocess){
  return std::unique_ptr<PointPainting>(
             new PointPaintingImp(seg_model_name, pp_model_name_0, 
                                  pp_model_name_1, need_preprocess));
}

}}

