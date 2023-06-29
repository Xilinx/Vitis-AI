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
#ifndef DEEPHI_SSD_DETECTOR_HPP_
#define DEEPHI_SSD_DETECTOR_HPP_

#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "vitis/ai/nnpp/multitaskv3.hpp"
namespace vitis {
namespace ai {

namespace multitaskv3 {

struct SSDOutputInfo {
  uint32_t order;        // order of bbox layer or conf layer
  int8_t* base_ptr;      // original ptr
  int8_t* ptr;           // ptr for batch
  uint32_t index_begin;  // index for prior boxes
  uint32_t index_size;
  float scale;
  uint32_t size;
  uint32_t bbox_single_size;  // usualy 4, but sometimes 6 and last 2 number not
                              // valid
};

class SSDdetector {
 public:
  enum CodeType { CORNER, CENTER_SIZE, CORNER_SIZE };

  SSDdetector(unsigned int num_classes,  // int background_label_id,
              CodeType code_type, bool variance_encoded_in_target,
              unsigned int keep_top_k,
              const std::vector<float>& confidence_threshold,
              unsigned int nms_top_k, float nms_threshold, float eta,
              const std::vector<std::shared_ptr<std::vector<float>>>& priors,
              float scale = 1.f, bool clip = false);

  void Detect(const std::map<uint32_t, SSDOutputInfo>& loc_infos,
              const float* conf_data, std::vector<Vehiclev3Result>& result);

  unsigned int num_classes() const { return num_classes_; }
  unsigned int num_priors() const { return priors_.size(); }

 protected:
  void ApplyOneClassNMS(
      const std::map<uint32_t, SSDOutputInfo>& bbox_layer_infos,
      const float* conf_data, int label,
      const std::vector<std::pair<float, int>>& score_index_vec,
      std::vector<int>* indices);

  void GetOneClassMaxScoreIndex(
      const float* conf_data, int label,
      std::vector<std::pair<float, int>>* score_index_vec);

  void GetMultiClassMaxScoreIndex(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int>>>* score_index_vec);

  void GetMultiClassMaxScoreIndexMT(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int>>>* score_index_vec,
      int threads = 2);
  void DecodeBBox(const int8_t* bbox_ptr, int idx, float scale,
                  bool normalized);

  std::map<int, std::vector<float>> decoded_bboxes_;

  const unsigned int num_classes_;
  // int background_label_id_;
  CodeType code_type_;
  bool variance_encoded_in_target_;
  unsigned int keep_top_k_;
  std::vector<float> confidence_threshold_;
  float nms_confidence_;
  unsigned int nms_top_k_;
  float nms_threshold_;
  float eta_;

  const std::vector<std::shared_ptr<std::vector<float>>> priors_;
  float scale_;

  bool clip_;

  int num_priors_;
};

}  // namespace multitask
}  // namespace ai
}  // namespace vitis
#endif
