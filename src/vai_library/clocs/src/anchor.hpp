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
#include <functional>
#include <vector>

using namespace std;

namespace vitis {
namespace ai {
namespace clocs {

struct AnchorInfo {
  std::vector<float> featmap_size;  // z, y, x
  std::vector<float> sizes;         // anchor sizes
  std::vector<float> strides;
  std::vector<float> offsets;
  std::vector<float> rotations;
  float matched_threshold;
  float unmatched_threshold;
};

using Anchors = std::vector<std::vector<float>>;

Anchors generate_anchors_stride(const AnchorInfo& params);
Anchors get_anchors_bv(const Anchors& anchors);
vector<bool> get_anchor_mask(const vector<int>& coors, int nx, int ny,
                             const Anchors& anchor_bv, float anchor_area_thresh,
                             const vector<float>& voxel_size,
                             const vector<float>& pc_range,
                             const vector<int>& grid_size);

vector<size_t> get_valid_anchor_index(const vector<int>& coors, int nx, int ny,
                                      const Anchors& anchor_bv,
                                      float anchor_area_thresh,
                                      const vector<float>& voxel_size,
                                      const vector<float>& pc_range,
                                      const vector<int>& grid_size);
// std::vector<std::vector<std::vector<float>>>
// generate_anchors(const AnchorInfo &params);

}  // namespace clocs
}  // namespace ai
}  // namespace vitis
