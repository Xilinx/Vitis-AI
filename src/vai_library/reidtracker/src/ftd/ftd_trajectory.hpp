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

#ifndef _FTD_TRAJECTORY_HPP_
#define _FTD_TRAJECTORY_HPP_

#include <tuple>
#include <vitis/ai/env_config.hpp>
#include "ftd_filter_linear.hpp"
DEF_ENV_PARAM(DEBUG_REID_TRACKER, "0")

using namespace cv;

namespace vitis {
namespace ai {

// InputCharact: roi, score, label, local_id
typedef std::tuple<cv::Mat, cv::Rect_<float>, float, int, int> InputCharact;
// OutputCharact: gid, roi, score, label, local_id
typedef std::tuple<uint64_t, cv::Rect_<float>, float, int, int> OutputCharact;

class FTD_Trajectory {
 public:
  FTD_Trajectory(SpecifiedCfg& specified_cfg);
  ~FTD_Trajectory(){};
  void Predict();
  int GetId();
  void SetId(uint64_t& update_id);
  InputCharact& GetCharact();
  void Init(const InputCharact& input_charact, std::vector<uint64_t>& id_record,
            int mode);
  void UpdateTrack();
  void UpdateDetect(const InputCharact& input_charact);
  void UpdateWithoutDetect();
  void UpdateFeature(const cv::Mat& feat);
  int GetStatus();
  bool GetShown();
  OutputCharact GetOut();
  std::vector<cv::Mat> GetFeatures();
  int B2G=0;
  int G2B=0;
  int B2D=0;
  int leap=0;
  int hit_streak = 0;
  int age = 0;
  int time_since_update = 0;

 private:
  int id=0;
  InputCharact charact;
  // FTD_Filter filter;
  // FTD_Filter_Light filter;
  FTD_Filter_Linear filter;
  // FTD_Filter_Run filter;
  SpecifiedCfg specified_cfg_;
  std::vector<cv::Mat> features;
  cv::Mat feature;
  int status=0;
  bool have_been_shown=false;
};

}  // namespace ai
}  // namespace vitis
#endif
