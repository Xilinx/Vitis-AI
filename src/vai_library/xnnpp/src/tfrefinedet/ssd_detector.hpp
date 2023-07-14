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

#include <array>
#include <map>
#include <vector>

#include "vitis/ai/nnpp/tfrefinedet.hpp"

namespace vitis {
namespace ai {
namespace tfrefinedet {
class SSDDetector {
 public:
  SSDDetector(int num_classes, const std::vector<std::vector<float>>& priors,
              float scale_xy, float scale_wh, float score_threshold,
              int keep_topk, int topk, float criteria);

  void detect(const int8_t* arm_loc_addr, const int8_t* odm_loc_addr,
              const int8_t* arm_conf_addr, const int8_t* odm_conf_addr,
              float arm_loc_scale, float odm_loc_scale, float arm_conf_scale,
              float odm_conf_scale, RefineDetResult& result,
              bool sort_by_class = true);

 private:
  int num_classes_;
  std::vector<std::vector<float>> priors_;
  float scale_xy_;
  float scale_wh_;
  float score_threshold_;
  const int keep_topk_;
  const int topk_;
  const float criteria_;

 private:
  std::map<int, std::vector<float>> decoded_bboxes_;
};

}  // namespace tfrefinedet
}  // namespace ai
}  // namespace vitis
