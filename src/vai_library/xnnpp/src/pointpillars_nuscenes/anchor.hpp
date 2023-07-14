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
#include <functional>

using namespace std;

namespace vitis {
namespace ai {
namespace pointpillars_nus {

struct AnchorInfo {
  std::vector<float> featmap_size;
  std::vector<std::vector<float>> ranges; // anchor ranges
  std::vector<std::vector<float>> sizes;  // anchor sizes
  std::vector<float> rotations; 
  std::vector<float> custom_values;
  bool align_corner; // default false
  float scale;
};

using Anchors = std::vector<std::vector<float>>; 

Anchors generate_anchors(const AnchorInfo &params);

//std::vector<std::vector<std::vector<float>>>
//generate_anchors(const AnchorInfo &params);


}}}
