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

#include <memory>
#include <vector>

namespace vitis {
namespace ai {
namespace object_detection_base {

struct SelectedOutput {
  int level;
  int index;
  const int8_t* pscore;
  float score_scale;
  const int8_t* pbox;
  float box_scale;
  int box_size;
  int cls;
  friend inline bool operator<(const SelectedOutput& lhs,
                               const SelectedOutput& rhs) {
    return (*(lhs.pscore) * lhs.score_scale) <
           (*(rhs.pscore) * rhs.score_scale);
  }

  friend inline bool operator>(const SelectedOutput& lhs,
                               const SelectedOutput& rhs) {
    return (*(lhs.pscore) * lhs.score_scale) >
           (*(rhs.pscore) * rhs.score_scale);
  }
};

struct DecodedOutput {
  int cls;
  std::vector<float> bbox;  // yxyx
  float score;
};

// single output
std::vector<SelectedOutput> select(int num_classes, const int8_t* box_output,
                                   float box_output_scale, int box_length,
                                   const int8_t* cls_output,
                                   float cls_output_scale, int size,
                                   int8_t score_int8);
std::vector<std::vector<SelectedOutput>> select_all_classes(
    int num_classes, const int8_t* box_output, float box_output_scale,
    int box_length, const int8_t* cls_output, float cls_output_scale, int size,
    int8_t score_int8);

// multi output layers with different fix points
std::vector<SelectedOutput> select(int level, int num_classes,
                                   const int8_t* box_output,
                                   float box_output_scale, int box_length,
                                   const int8_t* cls_output,
                                   float cls_output_scale, int size,
                                   int8_t score_int8);

std::vector<std::vector<SelectedOutput>> select_all_classes(
    int level, int num_classes, const int8_t* box_output,
    float box_output_scale, int box_length, const int8_t* cls_output,
    float cls_output_scale, int size, int8_t score_int8);

std::vector<SelectedOutput> topK(const std::vector<SelectedOutput>& input,
                                 int k);

DecodedOutput decode(const SelectedOutput& selected,
                     const std::vector<float>& anchor);

std::vector<DecodedOutput> per_class_nms(
    const std::vector<DecodedOutput>& candidates, int num_classes,
    float nms_thresh, float score_thresh, int max_output_num,
    bool need_sort = false);

}  // namespace object_detection_base
}  // namespace ai
}  // namespace vitis

