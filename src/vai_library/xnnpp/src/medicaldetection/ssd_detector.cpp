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
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/nnpp/apply_nms.hpp"

using namespace std;

namespace vitis {
namespace ai {
namespace medicaldetection {

SSDDetector::SSDDetector(int num_classes,
                         const std::vector<std::vector<float>>& priors,
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
                         const int8_t* odm_conf_addr,
                         float arm_loc_scale,   // 0.0625
                         float odm_loc_scale,   // 0.03125
                         float arm_conf_scale,  // 0.125
                         float odm_conf_scale,  // 0.125
                         MedicalDetectionResult& result) {
  // var in python: arm_scores_in, arm_bboxes_in, odm_scores_in, odm_bboxes_in
  //                arm_conf_addr  arm_loc_addr   odm_conf_addr  odm_loc_addr

  std::vector<float> arm_conf_softmax(priors_.size() * 2);
  std::vector<float> odm_conf_softmax(priors_.size() * num_classes_);

  vitis::ai::softmax((int8_t*)arm_conf_addr, arm_conf_scale, 2, priors_.size(),
                     arm_conf_softmax.data());
  vitis::ai::softmax((int8_t*)odm_conf_addr, odm_conf_scale, num_classes_,
                     priors_.size(), odm_conf_softmax.data());

  for (auto i = 0u; i < priors_.size(); i++) {
    if (arm_conf_softmax[i * 2 + 1] <= 0.01) {
      std::fill_n(odm_conf_softmax.begin() + i * num_classes_, num_classes_, 0);
    }
  }

  auto DecodeBBox = [&](int j) {
    std::vector<float> arm_bboxes_in(4);
    std::vector<float> odm_bboxes_in(4);
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
  std::vector<std::vector<std::vector<float>>> bboxes(num_classes_);
  std::vector<std::vector<float>> score(num_classes_);
  std::vector<std::vector<size_t>> candidates(num_classes_);
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

  std::vector<std::tuple<float, int, size_t>>
      score_index_tuples;  // score, label, pos
  for (int i = 1; i < num_classes_; i++) {
    for (auto j = 0u; j < candidates[i].size(); j++) {
      auto scorex = score[i][candidates[i][j]];
      score_index_tuples.emplace_back(scorex, i, candidates[i][j]);
    }
  }
  // keep top K
  std::sort(score_index_tuples.begin(), score_index_tuples.end(),
            [](const std::tuple<float, int, size_t>& lhs,
               const std::tuple<float, int, size_t>& rhs) {
              return std::get<0>(lhs) > std::get<0>(rhs);
            });

  if (num_det > keep_topk_) {
    score_index_tuples.resize(keep_topk_);
  }

  MedicalDetectionResult::BoundingBox res;
#if 1
  for (auto& item : score_index_tuples) {
    int i = std::get<1>(item);
    int idx = std::get<2>(item);
    auto scorex = std::get<0>(item);
    res.label = i;
    res.score = scorex;
    res.x = bboxes[i][idx][0] - 0.5f * bboxes[i][idx][2];
    res.y = bboxes[i][idx][1] - 0.5f * bboxes[i][idx][3];
    res.width = bboxes[i][idx][2];
    res.height = bboxes[i][idx][3];
    result.bboxes.emplace_back(res);
  }

#else

  // this branch sort result by classes

  candidates.clear();
  candidates.resize(num_classes_);
  for (auto& item : score_index_tuples) {
    candidates[std::get<1>(item)].emplace_back(std::get<2>(item));
  }
  // now prepare the result
  MedicalDetectionResult::BoundingBox res;
  for (int i = 1; i < num_classes_; i++) {
    for (auto idx : candidates[i]) {
      auto scorex = score[i][idx];
      if (scorex < score_threshold_) {
        continue;
      }
      float bbox_0 = std::max(std::min(bboxes[i][idx][0], 1.f), 0.f);
      float bbox_1 = std::max(std::min(bboxes[i][idx][1], 1.f), 0.f);
      float bbox_2 = std::max(std::min(bboxes[i][idx][2], 1.f), 0.f);
      float bbox_3 = std::max(std::min(bboxes[i][idx][3], 1.f), 0.f);
      if (bbox_2 <= bbox_0 || bbox_3 <= bbox_1) {
        continue;
      }
      auto box_rect = cv::Rect_<float>(cv::Point2f(bbox_0, bbox_1),
                                       cv::Point2f(bbox_2, bbox_3));

      res.label = i;
      res.score = scorex;
      res.x = box_rect.x;
      res.y = box_rect.y;
      res.width = box_rect.width;
      res.height = box_rect.height;
      result.bboxes.emplace_back(res);
    }
  }
#endif

  return;
}

}  // namespace medicaldetection
}  // namespace ai
}  // namespace vitis
