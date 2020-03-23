/*
 * Copyright 2019 Xilinx Inc.
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

#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "vitis/ai/nnpp/tfssd.hpp"

namespace vitis {
namespace ai {

using ai::TFSSDResult;
namespace dptfssd {

enum SCORE_CONVERTER { SOFTMAX=0, SIGMOID=1 };

class TFSSDdetector {

 public:
  enum CodeType {
    CORNER,
    CENTER_SIZE,
    CORNER_SIZE
  };

  TFSSDdetector(unsigned int num_classes, 
              CodeType code_type, bool variance_encoded_in_target,
              unsigned int keep_top_k,
              const std::vector<float>& confidence_threshold,
              unsigned int nms_top_k, float nms_threshold, float eta,
              const std::vector<std::shared_ptr<std::vector<float> > >& priors,
              float y_scale,
              float x_scale,
              float height_scale,
              float width_scale,
              SCORE_CONVERTER score_converter,
              float scale_score = 1.f, 
              float scale_loc = 1.f, 
              bool clip = false);

  template <typename T>
  void Detect(const T* loc_data, const float* conf_data, 
              TFSSDResult* result);

  unsigned int num_classes() const { return num_classes_; }
  unsigned int num_priors() const { return priors_.size(); }

 protected:
  template <typename T>
  void ApplyOneClassNMS(
      const T (*bboxes)[4], const float* conf_data, int label,
      const std::vector<std::pair<float, int> >& score_index_vec,
      std::vector<int>* indices);

  void GetOneClassMaxScoreIndex(
      const float* conf_data, int label,
      std::vector<std::pair<float, int> >* score_index_vec);

  void GetMultiClassMaxScoreIndex(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int> > >* score_index_vec);

  void GetMultiClassMaxScoreIndexMT(
      const float* conf_data, int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int> > >* score_index_vec,
      int threads = 1);

  template <typename T>
  float JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
                       bool normalized = true);

  template <typename T>
  void DecodeBBox(const T (*bboxes)[4], int idx, bool normalized);

  std::map<int, std::vector<float> > decoded_bboxes_;

  const unsigned int num_classes_;
  CodeType code_type_;
  bool variance_encoded_in_target_;
  unsigned int keep_top_k_;
  std::vector<float> confidence_threshold_;
  float nms_confidence_;
  unsigned int nms_top_k_;
  float nms_threshold_;
  float eta_;

  const std::vector<std::shared_ptr<std::vector<float> > >& priors_;
  float y_scale_;
  float x_scale_;
  float height_scale_;
  float width_scale_;


  SCORE_CONVERTER score_converter_;
  float scale_score_;
  float scale_loc_;

  bool clip_;
  int num_priors_;
};
}  // namespace dpssd
}  // namespace ai
}  // namespace vitis
