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

#ifndef _FTD_STRUCTURE_HPP
#define _FTD_STRUCTURE_HPP

#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vitis/ai/reid.hpp>
#include "ftd_hungarian.hpp"
#include "ftd_trajectory.hpp"
typedef pair<int, Mat> imagePair;
class paircomp {
 public:
  bool operator()(const imagePair& n1, const imagePair& n2) const {
    return n1.first > n2.first;
  }
};

namespace vitis {
namespace ai {

class FTD_Structure {
 public:
  FTD_Structure(const SpecifiedCfg& specified_cfg);
  ~FTD_Structure();
  void clear();

  std::vector<OutputCharact> Update(uint64_t frame_id, bool detect_flag,
                                    int mode,
                                    std::vector<InputCharact>& input_characts);
  std::vector<int> GetRemoveID();

  int max_age = 60;
  int min_hits = 3;
  int frame_count = 0;

 private:
  std::vector<uint64_t> id_record;
  uint64_t track_id;
  float iou_threshold;
  float feat_distance_low;
  float feat_distance_high;
  float score_threshold;
  cv::Rect_<float> roi_range;
  std::vector<std::shared_ptr<FTD_Trajectory>> tracks;

  void GetOut(std::vector<OutputCharact>& output_characts);
  std::vector<int> remove_id_this_frame;
  SpecifiedCfg specified_cfg_;
};

}  // namespace ai
}  // namespace vitis
#endif
