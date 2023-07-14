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

#include "vitis/ai/nnpp/ssd.hpp"

namespace vitis {
namespace ai {
using ai::SSDResult;

namespace dpssd {

struct SSDOutputInfo {
  int8_t output_tensor_index;  // output tensor index
  int8_t type;                 // conf = 1; bbox = 2;
  uint32_t order;              // order of bbox layer or conf layer
  int8_t* base_ptr;            // original ptr
  int8_t* ptr;                 // ptr for batch
  uint32_t index_begin;        // index for prior boxes
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
              unsigned int nms_top_k, float nms_threshold,  // float eta,
              const std::vector<std::shared_ptr<std::vector<float>>>& priors,
              bool if_tfmodel, bool is_mlperf, float scale = 1.f,
              bool clip = false);

  void detect(const std::map<uint32_t, SSDOutputInfo>& loc_infos,
              const float* conf_data, SSDResult* result);
  // template <typename T>
  // void detect(const T* loc_data, const float* conf_data,
  //            SSDResult* result);

  unsigned int num_classes() const { return num_classes_; }
  unsigned int num_priors() const { return priors_.size(); }

 protected:
  void apply_one_class_nms(
      const std::map<uint32_t, SSDOutputInfo>& bbox_layer_infos,
      const float* conf_data, int label,
      const std::vector<std::pair<float, int>>& score_index_vec,
      std::vector<int>* indices);
  // template <typename T>
  // void apply_one_class_nms(
  //    const T (*bboxes)[4], const float* conf_data, int label,
  //    const std::vector<std::pair<float, int> >& score_index_vec,
  //    std::vector<int>* indices);

  void get_multi_class_max_score_index(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int>>>* score_index_vec);

  void get_multi_class_max_score_index_mt(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int>>>* score_index_vec,
      int threads = 1);

  float jaccard_overlap(int idx, int kept_idx, bool normalized = true);
  // template <typename T>
  // float JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
  //                     bool normalized = true);

  void decode_bbox(const int8_t* bbox_ptr, int idx, float scale,
                   bool normalized);
  // template <typename T>
  // void DecodeBBox(const T (*bboxes)[4], int idx, bool normalized);

  std::map<int, std::vector<float>> decoded_bboxes_;

  const unsigned int num_classes_=0;
  // int background_label_id_;
  CodeType code_type_=CORNER;
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
    const std::vector<std::shared_ptr<std::vector<float>>>& priors,
    const vitis::ai::proto::DpuModelParam& config);

}  // namespace dpssd
}  // namespace ai
}  // namespace vitis
