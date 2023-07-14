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

#include "./object_detection_base.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <queue>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using std::vector;

namespace vitis {
namespace ai {
namespace object_detection_base {

DEF_ENV_PARAM(DEBUG_EFFICIENTDET_D2_DECODE, "0")
DEF_ENV_PARAM(DEBUG_TOPK, "0")

static float sigmoid(float input) { return 1.0 / (1.0 + std::exp(-input)); }

static float overlap(float x1, float w1, float x2, float w2) {
  float left = std::max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = std::min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

float cal_iou_xywh(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

float cal_iou_xyxy(vector<float> box, vector<float> truth) {
  float box_w = box[2] - box[0];
  float box_h = box[3] - box[1];
  float truth_w = truth[2] - truth[0];
  float truth_h = truth[3] - truth[1];
  float w = overlap(box[0], box_w, truth[0], truth_w);
  float h = overlap(box[1], box_h, truth[1], truth_h);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box_w * box_h + truth_w * truth_h - inter_area;
  return inter_area * 1.0 / union_area;
}

float cal_iou_yxyx(vector<float> box, vector<float> truth) {
  float box_h = box[2] - box[0];
  float box_w = box[3] - box[1];
  float truth_h = truth[2] - truth[0];
  float truth_w = truth[3] - truth[1];
  float h = overlap(box[0], box_h, truth[0], truth_h);
  float w = overlap(box[1], box_w, truth[1], truth_w);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box_w * box_h + truth_w * truth_h - inter_area;
  return inter_area * 1.0 / union_area;
}

vector<SelectedOutput> select(int level, int num_classes,
                              const int8_t* box_output, float box_output_scale,
                              int box_length, const int8_t* cls_output,
                              float cls_output_scale, int size,
                              int8_t score_int8_thresh) {
  // assert(num_classes > 0);
  vector<SelectedOutput> result;
  for (auto i = 0; i < size; ++i) {
    if (*(cls_output + i) >= score_int8_thresh) {
      auto cls = i % num_classes;
      auto index = i / num_classes;
      // auto score = sigmoid(*(cls_output + i) * cls_output_scale);
      result.emplace_back(SelectedOutput{
          level, index, cls_output + i, cls_output_scale,
          box_output + index * box_length, box_output_scale, box_length, cls});
    }
  }
  return result;
}

vector<vector<SelectedOutput>> select_all_classes(
    int level, int num_classes, const int8_t* box_output,
    float box_output_scale, int box_length, const int8_t* cls_output,
    float cls_output_scale, int size, int8_t score_int8_thresh) {
  // assert(num_classes > 0);
  vector<vector<SelectedOutput>> result(num_classes);
  for (auto i = 0; i < size; ++i) {
    if (*(cls_output + i) < score_int8_thresh) {
      auto cls = i % num_classes;
      result[cls].emplace_back(SelectedOutput{
          level, i, cls_output + i, cls_output_scale,
          box_output + i * box_length, box_output_scale, box_length, cls});
    }
  }
  return result;
}

vector<SelectedOutput> select(int num_classes, const int8_t* box_output,
                              float box_output_scale, int box_length,
                              const int8_t* cls_output, float cls_output_scale,
                              int size, int8_t score_int8_thresh) {
  return select(0, num_classes, box_output, box_output_scale, box_length,
                cls_output, cls_output_scale, size, score_int8_thresh);
}

vector<vector<SelectedOutput>> select_all_classes(
    int num_classes, const int8_t* box_output, float box_output_scale,
    int box_length, const int8_t* cls_output, float cls_output_scale, int size,
    int8_t score_int8_thresh) {
  return select_all_classes(0, num_classes, box_output, box_output_scale,
                            box_length, cls_output, cls_output_scale, size,
                            score_int8_thresh);
}

vector<SelectedOutput> topK(const vector<SelectedOutput>& input, int k) {
  // assert(k >= 0);
  int size = input.size();
  int num = std::min(size, k);
  std::vector<SelectedOutput> result(input.begin(), input.begin() + num);
  if (ENV_PARAM(DEBUG_TOPK)) {
    for (auto i = 0u; i < input.size(); ++i) {
      auto& s = input[i];
      LOG(INFO) << "topk input:" << i << ", level:" << s.level
                << ", index:" << s.index
                << ", score:" << (*(s.pscore)) * s.score_scale
                << ", ori score:" << (int)(*(s.pscore)) << "*" << s.score_scale;
    }
  }
  std::make_heap(result.begin(), result.begin() + num, std::greater<>());
  for (auto i = num; i < size; ++i) {
    if (input[i] > result[0]) {
      std::pop_heap(result.begin(), result.end(), std::greater<>());
      result[num - 1] = input[i];
    }
  }

  for (auto i = 0; i < num; ++i) {
    std::pop_heap(result.begin(), result.begin() + num - i, std::greater<>());
  }
  // std::stable_sort(result.begin(), result.end(), compare);
  if (ENV_PARAM(DEBUG_TOPK)) {
    for (auto i = 0u; i < result.size(); ++i) {
      LOG(INFO) << "topk:" << i << ", level:" << result[i].level
                << ", index:" << result[i].index
                << ", score:" << (*(result[i].pscore)) * result[i].score_scale
                << ", ori score:" << (int)(*(result[i].pscore)) << "*"
                << result[i].score_scale;
    }
  }
  return result;
}

DecodedOutput decode(const SelectedOutput& selected,
                     const std::vector<float>& anchor_boxes) {
  float ycenter_a = (anchor_boxes[0] + anchor_boxes[2]) / 2.0;
  float xcenter_a = (anchor_boxes[1] + anchor_boxes[3]) / 2.0;
  auto ha = anchor_boxes[2] - anchor_boxes[0];
  auto wa = anchor_boxes[3] - anchor_boxes[1];
  auto ty = selected.pbox[0] * selected.box_scale;
  auto tx = selected.pbox[1] * selected.box_scale;
  auto th = selected.pbox[2] * selected.box_scale;
  auto tw = selected.pbox[3] * selected.box_scale;
  auto w = std::exp(tw) * wa;
  auto h = std::exp(th) * ha;
  LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2_DECODE))
      << "ty:" << ty << ", tx:" << tx << ", th:" << th << ", tw:" << tw
      << ", w:" << w << ", h:" << h
      << ", score ori:" << (int)(*selected.pscore);
  auto ycenter = ty * ha + ycenter_a;
  auto xcenter = tx * wa + xcenter_a;
  auto ymin = ycenter - h / 2;
  auto xmin = xcenter - w / 2;
  auto ymax = ycenter + h / 2;
  auto xmax = xcenter + w / 2;
  auto bbox = vector<float>{ymin, xmin, ymax, xmax};
  auto score = sigmoid(*selected.pscore * selected.score_scale);
  LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2_DECODE))
      << "ymin:" << ymin << ", xmin:" << xmin << ", ymax:" << ymax
      << ", xmax:" << xmax << ", score:" << score;
  DecodedOutput output{selected.cls, bbox, score};
  return output;
}

vector<DecodedOutput> per_class_nms(const vector<DecodedOutput>& candidates,
                                    int num_classes, float nms_thresh,
                                    float score_thresh, int max_output_num,
                                    bool need_sort) {
  vector<DecodedOutput> result;
  vector<vector<DecodedOutput>> candidate_classes(num_classes);
  auto compare = [](const DecodedOutput& l, const DecodedOutput& r) {
    return l.score >= r.score;
  };

  for (auto c = 0; c < num_classes; ++c) {
    for (auto i = 0u; i < candidates.size(); ++i) {
      if (candidates[i].cls == c) {
        candidate_classes[c].emplace_back(candidates[i]);
      }
    }
    // Todo: sort
    if (need_sort) {
      std::stable_sort(candidate_classes[c].begin(), candidate_classes[c].end(),
                       compare);
    }
  }

  // single class nms;
  for (auto c = 0; c < num_classes; ++c) {
    auto size = candidate_classes[c].size();
    vector<bool> exist_box(size, true);
    for (size_t i = 0; i < size; ++i) {
      if (!exist_box[i]) {
        continue;
      }
      if (candidate_classes[c][i].score < score_thresh) {
        exist_box[i] = false;
        continue;
      }
      result.push_back(candidate_classes[c][i]);
      for (size_t j = i + 1; j < size; ++j) {
        if (!exist_box[j]) {
          continue;
        }
        if (candidate_classes[c][j].score < score_thresh) {
          exist_box[j] = false;
          continue;
        }
        float overlap = 0.0;
        overlap = cal_iou_yxyx(candidate_classes[c][i].bbox,
                               candidate_classes[c][j].bbox);
        if (overlap >= nms_thresh) {
          exist_box[j] = false;
        }
      }
    }
  }

  // Todo: change stable sort to top-K
  std::stable_sort(result.begin(), result.end(), compare);
  if (result.size() > (unsigned int)max_output_num) {
    result.resize(max_output_num);
  }
  return result;
}

// int pre_nms_num = 5000;
// int nms_output_num = 100;
//// float score_thresh = 0.001; // acc
// float score_thresh = 0.3;
// float iou_thresh = 1.0;
// float sigma = 0.5;

}  // namespace object_detection_base
}  // namespace ai
}  // namespace vitis

