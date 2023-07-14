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
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <initializer_list>
#include "./preprocess.hpp"

namespace vitis { namespace ai {
namespace centerpoint{

void middle_process(const std::vector<int> &coors, const int8_t *input_data, float input_scale,  
                    int8_t *output_data, float output_scale, uint32_t in_channels = 64);
void middle_process(const DataContainer<int> &coors, const int8_t *input_data, 
                    int w, int h, int c, float input_scale, 
                    int8_t *output_data, float output_scale);
void middle_process(const DataContainer<int> &coors, const float *input_data, 
                    int w, int h, int c, float input_scale, 
                    float *output_data, float output_scale);

} // end of pointpillars_nuscenes
}}


