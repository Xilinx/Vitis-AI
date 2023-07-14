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

#include "./retinaface_detector.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <thread>
#include <tuple>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "vitis/ai/nnpp/apply_nms.hpp"

DEF_ENV_PARAM(DEBUG_XNNPP_RETINAFACE, "0")
using namespace cv;
using namespace std;
using vitis::ai::RetinaFaceResult;

namespace vitis {
namespace ai {
namespace retinaface {

RetinaFaceDetector::RetinaFaceDetector(unsigned int num_classes,  unsigned int label,
                         unsigned int keep_top_k,
                         //const vector<float>& confidence_threshold,
                         float confidence_threshold,
                         unsigned int nms_top_k, float nms_threshold,
                         const vector<vector<float>>& priors)
    : num_classes_(num_classes),
      label_(label), //anchor1
      keep_top_k_(keep_top_k),
      confidence_threshold_(confidence_threshold),
      nms_top_k_(nms_top_k),
      nms_threshold_(nms_threshold),
      priors_(priors) {
  num_priors_ = priors_.size();
}


//void RetinaFaceDetector::detect(const std::map<uint32_t, RetinaFaceOutputInfo>& loc_infos,
void RetinaFaceDetector::detect(const StrideLayersMap &layers_map,
                         const float* conf_data, RetinaFaceResult* result) {
  decoded_bboxes_.clear();
  decoded_landmarks_.clear();
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "decoded_bboxes_ size:" << decoded_bboxes_.size();

  // 1. select scores >= conf_thresh and mark the index of priors 
  unsigned int num_det = 0;
  // only need class of face
  //vector<vector<int>> indices(num_classes_);
  vector<int> indices;
  //vector<vector<pair<float, int>>> score_index_vec(num_classes_);
  vector<pair<float, int>> score_index_vec;

  // Get top_k scores (with corresponding indices).
  //get_multi_class_max_score_index_mt(conf_data, 1, num_classes_ - 1,
  //                                   &score_index_vec);

  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "get face top_K scores ...";
  get_one_class_max_score_index(conf_data, label_, &score_index_vec);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "score_index_vec.size :" << score_index_vec.size();
  //__TOC__(Sort)
  //__TIC__(NMS)
  //for (unsigned int c = 1; c < num_classes_; ++c) {
  //  // Perform NMS for one class
  //  apply_one_class_nms(loc_infos, conf_data, c, score_index_vec[c],
  //                      &(indices[c]));

  //  num_det += indices[c].size();
  //}
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "nms...";
  apply_one_class_nms(layers_map, conf_data, label_, score_index_vec,
                      &(indices));
  num_det += indices.size();

  if (keep_top_k_ > 0 && num_det > keep_top_k_) {
    vector<tuple<float, int, int>> score_index_tuples;
    for (auto j = 0u; j < indices.size(); ++j) {
      auto idx = indices[j];
      auto score = conf_data[idx * num_classes_ + label_];
      score_index_tuples.emplace_back(score, label_, idx);
    }
    

    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "select ...";
    // Keep top k results per image.
    std::sort(score_index_tuples.begin(), score_index_tuples.end(),
              [](const tuple<float, int, int>& lhs,
                 const tuple<float, int, int>& rhs) {
                return get<0>(lhs) > get<0>(rhs);
              });
    score_index_tuples.resize(keep_top_k_);

    indices.clear();
    for (auto& item : score_index_tuples) {
      indices.push_back(get<2>(item));
    }

    // num_det = keep_top_k_;
  }

  //for (auto label = 1u; label < indices.size(); ++label) {
    for (auto idx : indices) {
      auto score = conf_data[idx];
      auto& bbox = decoded_bboxes_[idx];
      auto& landmark = decoded_landmarks_[idx];
      RetinaFaceResult::BoundingBox res;
      //res.label = label;
      res.score = score;
      res.x = bbox[0];
      res.y = bbox[1];
      res.width = bbox[2];
      res.height = bbox[3];
      // res.box = box_rect;
      result->bboxes.emplace_back(res);
      result->landmarks.emplace_back(landmark);
    }
  //}
}

void RetinaFaceDetector::apply_one_class_nms(
//    const std::map<uint32_t, RetinaFaceOutputInfo>& bbox_layer_infos,
    const StrideLayersMap &layers_map,
    const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices) {
  vector<size_t> results;
  vector<vector<float>> boxes;
  vector<float> scores;
  map<size_t, int> resultmap;

  // float adaptive_threshold = nms_threshold_;
  indices->clear();
  unsigned int i = 0;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "score_index_vec.size() " << score_index_vec.size();

  while (i < score_index_vec.size()) {
    const uint32_t idx = score_index_vec[i].second;
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
          << "decode idx : " << idx << ", i = " << i;
    if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
      //for (auto it = bbox_layer_infos.begin(); it != bbox_layer_infos.end();
      auto index_begin = 0u;
      auto index_end = 0u;
      for (auto it = layers_map.begin(); it != layers_map.end();
           ++it) {
        index_end = index_begin + it->second.bbox_layer.size / 4; 
        //LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        //       << "stride = " << it->first
        //       << "bbox layer name: " << it->second.bbox_layer.layer_name
        //       << "size : " << it->second.bbox_layer.size;

        //LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        //        << "idx: " << idx << ", index_begin :" << index_begin
        //        << ", index_end :" << index_end;
        if (idx >= index_begin && idx < index_end){
          //bool debug_load = false;
          //if (!debug_load) {
            decode_bbox_landmark(it->second.bbox_layer.ptr + (idx - index_begin) * 4,
                                 it->second.landmark_layer.ptr + (idx - index_begin) * 10,
                        idx, it->second.bbox_layer.scale, 
                        it->second.landmark_layer.scale, true);
          //} else {
          //  // debug
          //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
          //        << "idx: " << idx << ", index_begin :" << index_begin
          //        << ", index_end :" << index_begin + it->second.bbox_layer.size/4; 
          //  decode_bbox(it->second.copyed_bbox_data.data() + (idx - index_begin) * 4,
          //              idx, it->second.bbox_layer.scale, true);
          //}
          break;
        }
        index_begin += it->second.bbox_layer.size / 4;
      }
    }

    boxes.push_back(decoded_bboxes_[idx]);
    scores.push_back(score_index_vec[i].first);
    resultmap[i] = idx;
    ++i;
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "bboxes size: " << boxes.size()
        << ", scores size: " << scores.size();
  //for (auto i = 0u; i < boxes.size(); ++i) {
  //  LOG(INFO) << "bbox[" << i << "] :" << boxes[i][0] << ", "
  //            << boxes[i][1] << ", "
  //            << boxes[i][2] << ", "
  //            << boxes[i][3] << ", "
  //            << " score: " << scores[i];

  //} 
  applyNMS(boxes, scores, nms_threshold_, confidence_threshold_,
           results);
  for (auto& r : results) {
    indices->push_back(resultmap[r]);
  }
}

void RetinaFaceDetector::get_one_class_max_score_index(
    const float* conf_data, int label,
    vector<pair<float, int>>* score_index_vec) {
  // __TIC__(PUSH2)
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
        << "num_priors_ size:" << num_priors_;
  //for (int i = 0; i < 10; ++i) {
  //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //       << "conf_data[" << i << "] : " << conf_data[i];
  //}

  //for (int i = 480; i < 480+10; ++i) {
  //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //       << "conf_data[" << i << "] : " << conf_data[i];
  //}

  //for (int i = 480 + 1920; i < 480+1920+10; ++i) {
  //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //       << "conf_data[" << i << "] : " << conf_data[i];
  //}

  for (int i = 0; i < num_priors_; ++i) {
      auto score = *(conf_data + i);
      //if (i >=480 && i < 480 + 1920) {
      //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
      //        << "stride16, index:" << i - 480 << ", score:" << score;
      //}
      //if (i >=480 + 1920 && i < 480 + 1920 + 7680) {
      //  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
      //        << "stride8, index:" << i - 480 - 1920 << ", score:" << score;
      //}
      if (score >= confidence_threshold_) {
        score_index_vec->emplace_back(score, i);
        LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
              << "find score: " << score << ", index:" << i; 
      }
  }
  // __TOC__(PUSH2)
  // __TIC__(SORT2)
  std::stable_sort(
      score_index_vec->begin(), score_index_vec->end(),
      [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
        return lhs.first > rhs.first;
      });
  // __TOC__(SORT2)
  if (nms_top_k_ < score_index_vec->size()) {
    score_index_vec->resize(nms_top_k_);
  }
}


void RetinaFaceDetector::decode_bbox_landmark(const int8_t* bbox_ptr,
                              const int8_t* landmark_ptr,
                              int idx, float bbox_scale, float landmark_scale,
                              bool normalized) {
  //LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE)) 
  //      << "bbox decoding ... idx = " << idx;
  vector<float> bbox(4, 0);
  std::array<std::pair<float, float>, 5> landmark;
  // scale bboxes
  //transform(bbox_ptr, bbox_ptr + 4, bbox.begin(),
  //          std::bind2nd(multiplies<float>(), scale));
  for (auto i = 0; i < 4; ++i) {
    bbox[i] = *(bbox_ptr + i) * bbox_scale; 
  }

  for (auto i = 0; i < 5; ++i) {
    landmark[i].first = *(landmark_ptr + 2 * i) * landmark_scale; 
    landmark[i].second = *(landmark_ptr + 2 * i + 1) * landmark_scale; 
  }
  //if (idx == 1337 + 480) {
  //  LOG(INFO) << "bbox scale : " << bbox_scale 
  //            << ", bbox[" << idx << "] :" 
  //          << bbox[0] << ", "
  //          << bbox[1] << ", "
  //          << bbox[2] << ", "
  //          << bbox[3] ;
  //  LOG(INFO) << "landmark scale : " << landmark_scale 
  //            << ", landmark[" << idx << "] :"
  //            << landmark[0].first << ", "
  //            << landmark[0].second << ", "
  //            << landmark[1].first << ", "
  //            << landmark[1].second << ", "
  //            << landmark[2].first << ", "
  //            << landmark[2].second << ", "
  //            << landmark[3].first << ", "
  //            << landmark[3].second << ", "
  //            << landmark[4].first << ", "
  //            << landmark[4].second;
  //}

  //transform(bbox_ptr, bbox_ptr + 4, bbox.begin(),
  //          std::bind2nd(multiplies<float>(), 1.0));

  // if (idx == 13149) {
  //  std::cout << "hi, id 13149 bbox "
  //            << bbox[0] << ","
  //            << bbox[1] << ","
  //            << bbox[2] << ","
  //            << bbox[3] << ","
  //            << std::endl;
  // }
  auto& prior_bbox = priors_[idx];
  //if (idx == 1337 + 480) {
  //  LOG(INFO) << "prior_bbox: " 
  //          << prior_bbox[0] << ", "
  //          << prior_bbox[1] << ", "
  //          << prior_bbox[2] << ", "
  //          << prior_bbox[3] ;
  //}

  auto w = prior_bbox[2] - prior_bbox[0] + 1.0;
  auto h = prior_bbox[3] - prior_bbox[1] + 1.0;
  auto ctr_x = prior_bbox[0] + 0.5 * (w - 1.0);
  auto ctr_y = prior_bbox[1] + 0.5 * (h - 1.0);
  //if (idx == 1337 + 480) {
  //  LOG(INFO) << "w: " << w << ", "
  //          << "h: " << h << ", "
  //          << "ctr_x: " << ctr_x << ", "
  //          << "ctr_y: " << ctr_y;
  //}
  auto pred_ctr_x = bbox[0] * w + ctr_x;
  auto pred_ctr_y = bbox[1] * h + ctr_y;
  auto pred_w = std::exp(bbox[2]) * w;
  auto pred_h = std::exp(bbox[3]) * h;
  //if (idx == 1337 + 480) {
  //  LOG(INFO) << "pred_w: " << pred_w << ", "
  //          << "pred_h: " << pred_h << ", "
  //          << "pred_ctr_x: " << pred_ctr_x << ", "
  //          << "pred_ctr_y: " << pred_ctr_y ;
  //}
  float x1 = pred_ctr_x - 0.5 * (pred_w - 1);
  float y1 = pred_ctr_y - 0.5 * (pred_h - 1);
  float x2 = pred_ctr_x + 0.5 * (pred_w - 1);
  float y2 = pred_ctr_y + 0.5 * (pred_h - 1);

  // bbox x,y,w,h
  bbox[0] = x1;
  bbox[1] = y1;
  bbox[2] = x2 - x1; 
  bbox[3] = y2 - y1;
  //if (idx == 1337 + 480) {
  //  LOG(INFO) << "bbox : " 
  //          << bbox[0] << ", "
  //          << bbox[1] << ", "
  //          << bbox[2] << ", "
  //          << bbox[3] ;
  //}
  bbox[0] /= 640;
  bbox[1] /= 384;
  bbox[2] /= 640;
  bbox[3] /= 384;
  //bbox[2] -= bbox[0];
  //bbox[3] -= bbox[1];
  for (auto i = 0u; i < 5; ++i) {
    landmark[i].first = landmark[i].first * w + ctr_x;
    landmark[i].second = landmark[i].second * h + ctr_y;
    landmark[i].first /= 640; 
    landmark[i].second /= 384; 
  }
  decoded_bboxes_.emplace(idx, std::move(bbox));
  decoded_landmarks_.emplace(idx, std::move(landmark));
}

std::unique_ptr<RetinaFaceDetector> create_retinaface_detector(
    const vector<vector<float>>& priors,
    const vitis::ai::proto::DpuModelParam& config) {
  const int num_classes = 2;
  const float NMS_THRESHOLD = config.retinaface_param().nms_threshold();
  const float conf_thresh = config.retinaface_param().det_threshold();
  //const int KEEP_TOP_K = config.ssd_param().keep_top_k();
  const int KEEP_TOP_K = 2000;
  //const int TOP_K = config.ssd_param().top_k();
  const int TOP_K = 2000;

  // if(ENV_PARAM(ENABLE_RetinaFace_DEBUG) == 1)
  //   DLOG(INFO) << " scale " << scale                              //
  //             << " num_classes " << num_classes                   //
  //             << " KEEP_TOP_K " << KEEP_TOP_K                     //
  //             << " th_conf " << th_conf[0] << ", " << th_conf[1]  //
  //             << " TOP_K " << TOP_K                               //
  //             << " NMS_THRESHOLD " << NMS_THRESHOLD               //
  //             << " priors.size() " << priors.size();              //
  const int label = 1;
  return std::unique_ptr<RetinaFaceDetector>(new RetinaFaceDetector(
      num_classes, label, KEEP_TOP_K,
      conf_thresh, TOP_K, NMS_THRESHOLD, priors));
}

}  // namespace dpssd
}  // namespace ai
}  // namespace vitis
