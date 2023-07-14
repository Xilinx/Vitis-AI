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
#include <vector>
#include <map>
#include <functional>
#include "./retinaface_detector.hpp"

using namespace std;

namespace vitis {
namespace ai {
namespace retinaface {

std::vector<std::vector<float>> 
generate_anchors(int input_width, int input_height, const std::vector<AnchorInfo> &params);

std::vector<std::vector<float>> 
generate_anchors(int input_width, int input_height, const std::map<int32_t, StrideLayers, std::greater<int32_t>> &params);

}}}
