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
#include <utility>
#include <vector>

namespace vitis {
namespace ai {

namespace dpssd {

struct SSDOutputInfo {
  int8_t output_tensor_index=0;  // output tensor index
  int8_t type=0;                 // conf = 1; bbox = 2;
  uint32_t order=0;              // order of bbox layer or conf layer
  float* base_ptr=NULL;            // original ptr
  float* ptr=NULL;                 // ptr for batch
  uint32_t index_begin=0;        // index for prior boxes
  uint32_t index_size=0;
  float scale=0.0;
  uint32_t size=0;
  uint32_t bbox_single_size=0;  // usualy 4, but sometimes 6 and last 2 number not
                              // valid
};


class SSDdetector {
 public:
  enum CodeType { CORNER, CENTER_SIZE, CORNER_SIZE };

  SSDdetector(unsigned int num_classes,  // int background_label_id,
              CodeType code_type, bool variance_encoded_in_target,
              unsigned int keep_top_k,
              const std::vector<float>& confidence_threshold,
              unsigned int nms_top_k, float nms_threshold,  // float eta,
              const std::vector<std::shared_ptr<std::vector<float>>>& priors,
              bool if_tfmodel, bool is_mlperf, float scale = 1.f,
              bool clip = false);

  void detect(const std::map<uint32_t, SSDOutputInfo>& loc_infos,
              const float* conf_data, std::vector<std::vector<float>>* result);

  unsigned int num_classes() const { return num_classes_; }
  unsigned int num_priors() const { return priors_.size(); }

 protected:
  void apply_one_class_nms(
      const std::map<uint32_t, SSDOutputInfo>& bbox_layer_infos,
      const float* conf_data, int label,
      const std::vector<std::pair<float, int>>& score_index_vec,
      std::vector<int>* indices);

  void get_one_class_max_score_index(
      const float* conf_data, int label,
      std::vector<std::pair<float, int>>* score_index_vec);

  void get_multi_class_max_score_index(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int>>>* score_index_vec);

  void get_multi_class_max_score_index_mt(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int>>>* score_index_vec,
      int threads = 1);

  float jaccard_overlap(int idx, int kept_idx, bool normalized = true);

  void decode_bbox(const float* bbox_ptr, int idx, float scale,
                   bool normalized);

  std::map<int, std::vector<float>> decoded_bboxes_;

  const unsigned int num_classes_=0;
  // int background_label_id_;
  CodeType code_type_;
  bool variance_encoded_in_target_=false;
  unsigned int keep_top_k_=0;
  std::vector<float> confidence_threshold_;
  float nms_confidence_=0.0;
  unsigned int nms_top_k_=0;
  float nms_threshold_=0.0;
  float eta_=0.0;

  const std::vector<std::shared_ptr<std::vector<float>>>& priors_;
  bool if_tfmodel_=false;
  bool is_mlperf_=false;
  float scale_=0.0;

  bool clip_=false;

  int num_priors_=0;
};

std::unique_ptr<SSDdetector> CreateSSDUniform(
    const std::vector<std::shared_ptr<std::vector<float>>>& priors, const int num_classes, 
    const float nms_threshold, const std::vector<float>& confidence_threshold, 
    const int keep_top_k, const int nms_top_k, const bool if_tfmodel, const bool is_mlperf);

}  // namespace dpssd
}  // namespace ai
}  // namespace vitis
