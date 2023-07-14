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
#ifndef DEEPHI_SSDANCHOR_HPP_
#define DEEPHI_SSDANCHOR_HPP_

#include <memory>
#include <utility>
#include <vector>

#include "anchorbase.hpp"

namespace vitis {
namespace ai {
namespace dptfssd {

class SSDAnchor : public AnchorBase {
 public:
  SSDAnchor(int num_layers, bool reduce_boxes_in_lowest_layer, float min_scale,
            float max_scale, float interpolated_scale_aspect_ratio,
            const std::vector<int>& feature_map_list,
            const std::vector<float>& aspect_ratios, int image_width,
            int image_height);
};

}  // namespace dptfssd
}  // namespace ai
}  // namespace vitis

#endif
