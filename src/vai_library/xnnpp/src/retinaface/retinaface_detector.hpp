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

#include "vitis/ai/nnpp/retinaface.hpp"

namespace vitis {
namespace ai {
namespace retinaface {

struct RetinaFaceOutputInfo {
  std::string layer_name;
  int8_t output_tensor_index;
  int8_t type; // conf=1; bbox=2; landmark=3
  int8_t anchor_type; // bg=0; fg=1; 
  int8_t anchor_index;
  int32_t stride;
  int8_t* base_ptr;            // original ptr
  int8_t* ptr;                 // ptr for batch
  //float *load_data; // only for conf
  float scale;
  uint32_t size;
};

struct AnchorInfo {
  int stride;
  int base_size;
  std::vector<float> ratios;
  std::vector<int> scales;
};

struct StrideLayers {
  AnchorInfo anchor_info;
  int32_t conf_data_size;
  RetinaFaceOutputInfo landmark_layer;
  RetinaFaceOutputInfo bbox_layer;
  std::vector<float> copyed_bbox_data; // debug
  std::vector<float> copyed_conf_data;
  //float *softmax_data;
};

typedef std::map<int32_t, StrideLayers, std::greater<int32_t>>  StrideLayersMap;

class RetinaFaceDetector {
 public:
  RetinaFaceDetector(unsigned int num_classes,  unsigned int label, //class label of face
              unsigned int keep_top_k,
              //const std::vector<float>& confidence_threshold, // need only one
              float confidence_threshold, 
              unsigned int nms_top_k, float nms_threshold,  // float eta,
              const std::vector<std::vector<float>>& priors);

  void detect(const StrideLayersMap &layers_map,
              const float* conf_data, RetinaFaceResult* result);

  unsigned int num_classes() const { return num_classes_; }
  unsigned int num_priors() const { return priors_.size(); }

 protected:
  void apply_one_class_nms(
      //const std::map<uint32_t, RetinaFaceOutputInfo>& bbox_layer_infos,
      const StrideLayersMap &layers_map,
      const float* conf_data, int label,
      const std::vector<std::pair<float, int>>& score_index_vec,
      std::vector<int>* indices);

  void get_one_class_max_score_index(
      const float* conf_data, int label,
      std::vector<std::pair<float, int>>* score_index_vec);

  void decode_bbox_landmark(const int8_t* bbox_ptr, const int8_t* landmark_ptr, 
                   int idx, float bbox_scale, float landmark_scale,
                   bool normalized);

  std::map<int, std::vector<float>> decoded_bboxes_; // key = index
  std::map<int, std::array<std::pair<float, float>, 5>> decoded_landmarks_; // key = index

  const unsigned int num_classes_;
  const unsigned int label_; // label of face
  // int background_label_id_;
  unsigned int keep_top_k_;
  //std::vector<float> confidence_threshold_;
  float confidence_threshold_;
  unsigned int nms_top_k_;
  float nms_threshold_;

  const std::vector<std::vector<float>>& priors_;
  //float scale_;

  int num_priors_;
};

std::unique_ptr<RetinaFaceDetector> create_retinaface_detector(
    const std::vector<std::vector<float>>& priors,
    const vitis::ai::proto::DpuModelParam& config);

}  // namespace retinaface
}  // namespace ai
}  // namespace vitis
