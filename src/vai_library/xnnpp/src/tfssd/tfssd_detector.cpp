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
#include "./tfssd_detector.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <thread>
#include <tuple>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/nnpp/apply_nms.hpp"

using namespace cv;
using namespace std;
using vitis::ai::TFSSDResult;

namespace vitis {
namespace ai {
namespace dptfssd {

TFSSDdetector::TFSSDdetector(unsigned int num_classes, CodeType code_type,
                             bool variance_encoded_in_target,
                             unsigned int keep_top_k,
                             const vector<float>& confidence_threshold,
                             unsigned int nms_top_k, float nms_threshold,
                             float eta,
                             const vector<shared_ptr<vector<float>>>& priors,
                             float y_scale, float x_scale, float height_scale,
                             float width_scale, SCORE_CONVERTER score_converter,
                             float scale_score, float scale_loc, bool clip)
    : num_classes_(num_classes),
      code_type_(code_type),
      variance_encoded_in_target_(false),
      keep_top_k_(keep_top_k),
      confidence_threshold_(confidence_threshold),
      nms_top_k_(nms_top_k),
      nms_threshold_(nms_threshold),
      eta_(eta),
      priors_(priors),
      y_scale_(y_scale),
      x_scale_(x_scale),
      height_scale_(height_scale),
      width_scale_(width_scale),
      score_converter_(score_converter),
      scale_score_(scale_score),
      scale_loc_(scale_loc),
      clip_(clip) {
  num_priors_ = priors_.size();
  nms_confidence_ = *std::min_element(confidence_threshold_.begin() + 1,
                                      confidence_threshold_.end());
  if (score_converter_ == SIGMOID) {
    // 1/(1+exp(-x*scale))==y;  -->  x=-ln(1/y-1)/scale
    // std::cout << "nms_confidence_:" << nms_confidence_ ;
    nms_confidence_ = (-log(1.0 / nms_confidence_ - 1.0)) / scale_score_;
    // std::cout << "   new nms_confidence_:" << nms_confidence_ << std::endl;
    // also need fix confidence_threshold_
    for (auto i = 1u; i < confidence_threshold_.size(); i++) {
      confidence_threshold_[i] =
          (-log(1.0 / confidence_threshold_[i] - 1.0)) / scale_score_;
    }
  }
}

template <typename T>
void TFSSDdetector::Detect(const T* loc_data, const float* conf_data,
                           TFSSDResult* result) {
  decoded_bboxes_.clear();
  const T(*bboxes)[4] = (const T(*)[4])loc_data;

  unsigned int num_det = 0;
  vector<vector<int>> indices(num_classes_);
  vector<vector<pair<float, int>>> score_index_vec(num_classes_);

  // Get top_k scores (with corresponding indices).
  GetMultiClassMaxScoreIndex(conf_data, 1, num_classes_ - 1, &score_index_vec);
  for (unsigned int c = 1; c < num_classes_; ++c) {
    // Perform NMS for one class
    ApplyOneClassNMS(bboxes, c, score_index_vec[c], &(indices[c]));
    num_det += indices[c].size();
  }

  if (keep_top_k_ > 0 && num_det > keep_top_k_) {
    vector<tuple<float, int, int>> score_index_tuples;
    for (auto label = 0u; label < num_classes_; ++label) {
      const vector<int>& label_indices = indices[label];
      for (auto j = 0u; j < label_indices.size(); ++j) {
        auto idx = label_indices[j];
        auto score = conf_data[idx * num_classes_ + label];
        score_index_tuples.emplace_back(score, label, idx);
      }
    }

    // Keep top k results per image.
    std::sort(score_index_tuples.begin(), score_index_tuples.end(),
              [](const tuple<float, int, int>& lhs,
                 const tuple<float, int, int>& rhs) {
                return get<0>(lhs) > get<0>(rhs);
              });
    score_index_tuples.resize(keep_top_k_);

    indices.clear();
    indices.resize(num_classes_);
    for (auto& item : score_index_tuples) {
      indices[get<1>(item)].push_back(get<2>(item));
    }
  }

  for (auto label = 1u; label < indices.size(); ++label) {
    for (auto idx : indices[label]) {
      auto score = conf_data[idx * num_classes_ + label];

      auto& bbox = decoded_bboxes_[idx];
      TFSSDResult::BoundingBox res;
      res.label = label;
      // res.score = score;
      //  =  1.0/(1.0+exp(-1.0*input[i]*scale ));
      res.score = (score_converter_ == SIGMOID)
                      ? 1.0 / (1.0 + exp(-1.0 * score * scale_score_))
                      : score;

      res.x = bbox[0] - bbox[2] / 2.0;
      res.y = bbox[1] - bbox[3] / 2.0;

      res.width = bbox[2];
      res.height = bbox[3];
      // std::cout <<"Detect(): Rect: bbox: " << bbox[0] << " " << bbox[1] << "
      // " << bbox[2] << " " << bbox[3]
      //          << "----- res.x:" <<  res.x << " res.y" << res.y << " width: "
      //          << res.width << " height:" << res.height<< std::endl;
      result->bboxes.emplace_back(res);
    }
  }
}

void TFSSDdetector::Detect(const int8_t* loc_data, const int8_t* conf_data,
                           TFSSDResult* result) {
  decoded_bboxes_.clear();
  const int8_t(*bboxes)[4] = (const int8_t(*)[4])loc_data;

  unsigned int num_det = 0;
  vector<vector<int>> indices(num_classes_);
  vector<vector<pair<float, int>>> score_index_vec(num_classes_);

  // Get top_k scores (with corresponding indices).
  GetMultiClassMaxScoreIndex(conf_data, 1, num_classes_ - 1, &score_index_vec);

  for (unsigned int c = 1; c < num_classes_; ++c) {
    // Perform NMS for one class
    ApplyOneClassNMS(bboxes, c, score_index_vec[c], &(indices[c]));
    num_det += indices[c].size();
  }

  if (keep_top_k_ > 0 && num_det > keep_top_k_) {
    vector<tuple<float, int, int>> score_index_tuples;
    for (auto label = 0u; label < num_classes_; ++label) {
      const vector<int>& label_indices = indices[label];
      for (auto j = 0u; j < label_indices.size(); ++j) {
        auto idx = label_indices[j];
        auto score = conf_data[idx * num_classes_ + label];
        score_index_tuples.emplace_back(score, label, idx);
      }
    }

    // Keep top k results per image.
    std::sort(score_index_tuples.begin(), score_index_tuples.end(),
              [](const tuple<float, int, int>& lhs,
                 const tuple<float, int, int>& rhs) {
                return get<0>(lhs) > get<0>(rhs);
              });
    score_index_tuples.resize(keep_top_k_);

    indices.clear();
    indices.resize(num_classes_);
    for (auto& item : score_index_tuples) {
      indices[get<1>(item)].push_back(get<2>(item));
    }
  }
  for (auto label = 1u; label < indices.size(); ++label) {
    for (auto idx : indices[label]) {
      auto score = conf_data[idx * num_classes_ + label];

      auto& bbox = decoded_bboxes_[idx];
      TFSSDResult::BoundingBox res;
      res.label = label;
      // res.score = score;
      //  =  1.0/(1.0+exp(-1.0*input[i]*scale ));
      res.score = (score_converter_ == SIGMOID)
                      ? 1.0 / (1.0 + exp(-1.0 * score * scale_score_))
                      : score;

      res.x = bbox[0] - bbox[2] / 2.0;
      res.y = bbox[1] - bbox[3] / 2.0;

      res.width = bbox[2];
      res.height = bbox[3];
      // std::cout <<"Detect(): Rect: bbox: " << bbox[0] << " " << bbox[1] << "
      // " << bbox[2] << " " << bbox[3]
      //          << "----- res.x:" <<  res.x << " res.y" << res.y << " width: "
      //          << res.width << " height:" << res.height<< std::endl;
      result->bboxes.emplace_back(res);
    }
  }
}

template void TFSSDdetector::Detect(const int* loc_data, const float* conf_data,
                                    TFSSDResult* result);
template void TFSSDdetector::Detect(const int8_t* loc_data,
                                    const float* conf_data,
                                    TFSSDResult* result);
// remove sigmoid_c
void TFSSDdetector::Detect(const int8_t* loc_data, const int8_t* conf_data,
                           TFSSDResult* result);
template <typename T>
void TFSSDdetector::ApplyOneClassNMS(
    const T (*bboxes)[4], int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices) {
  vector<size_t> results;
  vector<vector<float>> boxes;
  vector<float> scores;
  map<size_t, int> resultmap;
  int i = 0;
  for (auto sc : score_index_vec) {
    const int idx = sc.second;
    if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
      DecodeBBox(bboxes, idx, true);
    }
    boxes.push_back(decoded_bboxes_[idx]);
    scores.push_back(sc.first);
    resultmap[i++] = idx;
  }
  applyNMS(boxes, scores, nms_threshold_, confidence_threshold_[label],
           results);
  for (auto& r : results) {
    indices->push_back(resultmap[r]);
  }
}

template void TFSSDdetector::ApplyOneClassNMS(
    const int (*bboxes)[4], int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices);
template void TFSSDdetector::ApplyOneClassNMS(
    const int8_t (*bboxes)[4], int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices);

void TFSSDdetector::GetMultiClassMaxScoreIndex(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec) {
  for (int i = 0; i < num_priors_; ++i) {
    conf_data += start_label;
    for (auto j = start_label; j < start_label + num_classes; ++j) {
      auto score = *conf_data;
      if (score >= nms_confidence_) {
        (*score_index_vec)[j].emplace_back(score, i);
      }
      conf_data += 1;
    }
  }
  for (auto i = start_label; i < start_label + num_classes; ++i) {
    std::stable_sort(
        (*score_index_vec)[i].begin(), (*score_index_vec)[i].end(),
        [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
          return lhs.first > rhs.first;
        });

    if (nms_top_k_ < (*score_index_vec)[i].size()) {
      (*score_index_vec)[i].resize(nms_top_k_);
    }
  }
}

void TFSSDdetector::GetMultiClassMaxScoreIndex(
    const int8_t* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec) {
  int8_t int8_nms = (int8_t)nms_confidence_;
  for (int i = 0; i < num_priors_; ++i) {
    conf_data += start_label;
    for (auto j = start_label; j < start_label + num_classes; ++j) {
      auto score = *conf_data;

      if (score >= int8_nms) {
        (*score_index_vec)[j].emplace_back(score * 1.0f, i);
      }
      conf_data += 1;
    }
  }
  for (auto i = start_label; i < start_label + num_classes; ++i) {
    std::stable_sort(
        (*score_index_vec)[i].begin(), (*score_index_vec)[i].end(),
        [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
          return lhs.first > rhs.first;
        });

    if (nms_top_k_ < (*score_index_vec)[i].size()) {
      (*score_index_vec)[i].resize(nms_top_k_);
    }
  }
}

template <typename T>
void TFSSDdetector::DecodeBBox(const T (*bboxes)[4], int idx, bool normalized) {
  // in tfconcat_decode, we get the center and size directly in prior_bbox, so
  // we don't need decode center/size here. corresponding logic is in
  // box_coders/faster_rcnn_box_coder.py: _decode() scale bboxes

  auto& prior_bbox = *priors_[idx];
  auto y_center_a = prior_bbox[0];
  auto x_center_a = prior_bbox[1];
  auto ha = prior_bbox[2];
  auto wa = prior_bbox[3];

  auto ty = bboxes[idx][0] * scale_loc_ / y_scale_;
  auto tx = bboxes[idx][1] * scale_loc_ / x_scale_;
  auto th = bboxes[idx][2] * scale_loc_ / height_scale_;
  auto tw = bboxes[idx][3] * scale_loc_ / width_scale_;

  auto w = exp(tw) * wa;
  auto h = exp(th) * ha;
  auto ycenter = ty * ha + y_center_a;
  auto xcenter = tx * wa + x_center_a;

#if 0
  std::cout <<"DecodeBox:  bboxes[idx] " << float( bboxes[idx][0]) << " " << float( bboxes[idx][1] ) << " " 
            <<  float (bboxes[idx][2]) << " " << float( bboxes[idx][3] ) << std::endl;

  std::cout << "DecodeBox: ha wa:" << ha << " " << wa 
            << " ty tx th tw:" << ty << " " << tx << " " << th << " " << tw 
            << "    yxcenter:" << ycenter << " " << xcenter << std::endl;
#endif

  vector<float> bbox(4, 0);

  // seems x, y changed ? Yes. test and found.

  // bbox x,y,w,h
  bbox[0] = xcenter;
  bbox[1] = ycenter;
  bbox[2] = w;
  bbox[3] = h;

  decoded_bboxes_.emplace(idx, std::move(bbox));
}

template void TFSSDdetector::DecodeBBox(const int (*bboxes)[4], int idx,
                                        bool normalized);
template void TFSSDdetector::DecodeBBox(const int8_t (*bboxes)[4], int idx,
                                        bool normalized);
}  // namespace dptfssd
}  // namespace ai
}  // namespace vitis

