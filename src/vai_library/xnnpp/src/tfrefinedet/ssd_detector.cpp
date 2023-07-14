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
#include "ssd_detector.hpp"

#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/nnpp/apply_nms.hpp>
#include <vitis/ai/profiling.hpp>

DEF_ENV_PARAM(ENABLE_REFINE_DET_DEBUG, "0")
using namespace std;

namespace vitis {
namespace ai {
namespace tfrefinedet {

SSDDetector::SSDDetector(int num_classes, const vector<vector<float>>& priors,
                         float scale_xy, float scale_wh, float score_threshold,
                         int keep_topk, int topk, float criteria)
    : num_classes_(num_classes),
      priors_(priors),
      scale_xy_(scale_xy),
      scale_wh_(scale_wh),
      score_threshold_(score_threshold),
      keep_topk_(keep_topk),
      topk_(topk),
      criteria_(criteria) {}

void SSDDetector::detect(const int8_t* arm_loc_addr, const int8_t* odm_loc_addr,
                         const int8_t* arm_conf_addr,
                         const int8_t* odm_conf_addr, float arm_loc_scale,
                         float odm_loc_scale, float arm_conf_scale,
                         float odm_conf_scale, RefineDetResult& result,
                         bool sort_by_class) {
  // var in python: arm_scores_in, arm_bboxes_in, odm_scores_in, odm_bboxes_in
  //                arm_conf_addr  arm_loc_addr   odm_conf_addr  odm_loc_addr

  vector<float> arm_conf_softmax(priors_.size() * 2);
  vector<float> odm_conf_softmax(priors_.size() * num_classes_);

  softmax((int8_t*)arm_conf_addr, arm_conf_scale, 2, priors_.size(),
          arm_conf_softmax.data());
  softmax((int8_t*)odm_conf_addr, odm_conf_scale, num_classes_, priors_.size(),
          odm_conf_softmax.data());

  for (auto i = 0u; i < priors_.size(); i++) {
    if (arm_conf_softmax[i * 2 + 1] <= 0.01) {
      std::fill_n(odm_conf_softmax.begin() + i * num_classes_, num_classes_, 0);
    }
  }

  auto DecodeBBox = [&](int j) {
    vector<float> arm_bboxes_in(4);
    vector<float> odm_bboxes_in(4);
    arm_bboxes_in[0] =
        arm_loc_addr[j * 4 + 0] * arm_loc_scale * scale_xy_ * priors_[j][2] +
        priors_[j][0];
    arm_bboxes_in[1] =
        arm_loc_addr[j * 4 + 1] * arm_loc_scale * scale_xy_ * priors_[j][3] +
        priors_[j][1];
    arm_bboxes_in[2] =
        std::exp(arm_loc_addr[j * 4 + 2] * arm_loc_scale * scale_wh_) *
        priors_[j][2];
    arm_bboxes_in[3] =
        std::exp(arm_loc_addr[j * 4 + 3] * arm_loc_scale * scale_wh_) *
        priors_[j][3];
    odm_bboxes_in[0] =
        odm_loc_addr[j * 4 + 0] * odm_loc_scale * scale_xy_ * arm_bboxes_in[2] +
        arm_bboxes_in[0];
    odm_bboxes_in[1] =
        odm_loc_addr[j * 4 + 1] * odm_loc_scale * scale_xy_ * arm_bboxes_in[3] +
        arm_bboxes_in[1];
    odm_bboxes_in[2] =
        std::exp(odm_loc_addr[j * 4 + 2] * odm_loc_scale * scale_wh_) *
        arm_bboxes_in[2];
    odm_bboxes_in[3] =
        std::exp(odm_loc_addr[j * 4 + 3] * odm_loc_scale * scale_wh_) *
        arm_bboxes_in[3];

    return odm_bboxes_in;
  };

  // decode_single logic
  vector<vector<vector<float>>> bboxes(num_classes_);
  vector<vector<float>> score(num_classes_);
  vector<vector<size_t>> candidates(num_classes_);
  int num_det = 0;

  decoded_bboxes_.clear();
  for (auto i = 1; i < num_classes_; i++) {  // start from 1 to skip background
    for (auto j = 0u; j < priors_.size(); j++) {
      if (odm_conf_softmax[j * num_classes_ + i] > 0.005) {
        if (decoded_bboxes_.find(j) == decoded_bboxes_.end()) {
          decoded_bboxes_[j] = DecodeBBox(j);
        }
        bboxes[i].emplace_back(decoded_bboxes_[j]);
        score[i].emplace_back(odm_conf_softmax[j * num_classes_ + i]);
      }
    }
    if (score[i].empty()) {
      continue;
    }
    applyNMS(bboxes[i], score[i], criteria_, score_threshold_, candidates[i]);
    num_det += candidates[i].size();
  }  // end of long loop i

  vector<tuple<float, int, size_t>> sort_tuple;  // score,label,pos
  for (int i = 1; i < num_classes_; i++) {
    for (auto c : candidates[i]) {
      sort_tuple.emplace_back(score[i][c], i, c);
    }
  }
  // keep top K
  sort(sort_tuple.begin(), sort_tuple.end(),
       [](const tuple<float, int, size_t>& lhs,
          const tuple<float, int, size_t>& rhs) {
         return get<0>(lhs) > get<0>(rhs);
       });

  if (num_det > keep_topk_) {
    sort_tuple.resize(keep_topk_);
    candidates.clear();
    candidates.resize(num_classes_);

    for (auto& s : sort_tuple) {
      candidates[get<1>(s)].emplace_back(get<2>(s));
    }
  }

  RefineDetResult::BoundingBox res;
  if (sort_by_class) {
    for (int i = 1; i < num_classes_; i++) {
      for (auto idx : candidates[i]) {
        res.label = i;
        res.score = score[i][idx];
        res.x = bboxes[i][idx][0] - 0.5f * bboxes[i][idx][2];
        res.y = bboxes[i][idx][1] - 0.5f * bboxes[i][idx][3];
        res.width = bboxes[i][idx][2];
        res.height = bboxes[i][idx][3];
        result.bboxes.emplace_back(res);
      }
    }
  } else {
    for (auto& item : sort_tuple) {
      int i = get<1>(item);
      int idx = get<2>(item);
      auto scorex = get<0>(item);
      res.label = i;
      res.score = scorex;
      res.x = bboxes[i][idx][0] - 0.5f * bboxes[i][idx][2];
      res.y = bboxes[i][idx][1] - 0.5f * bboxes[i][idx][3];
      res.width = bboxes[i][idx][2];
      res.height = bboxes[i][idx][3];
      result.bboxes.emplace_back(res);
    }
  }
  return;
}

}  // namespace tfrefinedet
}  // namespace ai
}  // namespace vitis
