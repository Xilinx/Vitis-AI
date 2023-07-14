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

#include <map>
#include <memory>
#include <vector>

namespace vitis {
namespace ai {
namespace efficientdet_d2 {

class Anchor {
 public:
  typedef struct {
    int min_level;
    int max_level;
    int num_scales;
    std::vector<float> anchor_scales;
    std::vector<float> aspect_ratios;
    int image_width;
    int image_height;
  } AnchorConfig;

  using LevelBoxes = std::vector<std::vector<float>>;

  Anchor(const AnchorConfig& config);
  void generate_boxes();
  std::shared_ptr<LevelBoxes> get_boxes(int level);

 private:
  std::vector<std::vector<float>> generate_boxes_(const int stride_w,
                                                  const int stride_h,
                                                  float octave_scale,
                                                  float aspect,
                                                  float anchor_scale);
  std::map<int, std::shared_ptr<LevelBoxes>> leveled_bboxes_;
  AnchorConfig config_;
};

}  // namespace efficientdet_d2
}  // namespace ai
}  // namespace vitis

