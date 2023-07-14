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

#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
#include "./pointpillars_nuscenes_post.hpp"
#include <algorithm>
#include "./anchor.hpp"
#include "./utils.hpp"
#include "./sigmoid_table.hpp"

DEF_ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS, "0")
DEF_ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS_ANCHOR, "0")
DEF_ENV_PARAM(USE_DEBUG_SCORE, "0")
DEF_ENV_PARAM(USE_DEBUG_SCORE2, "0")
DEF_ENV_PARAM(DEBUG_SIGMOID_TABLE, "0")
DEF_ENV_PARAM(DEBUG_SIMPLE_API, "0")
DEF_ENV_PARAM(DEBUG_OLD_API, "0")
DEF_ENV_PARAM(DEBUG_NMS, "0")

using namespace std;
using namespace vitis::ai::pointpillars_nus;

namespace vitis {
namespace ai {

static inline float limit_period(float val, float offset, float period) {
  return val - std::floor(val / period + offset) * period;
}

//void build_anchor_info(AnchorInfo &anchor_info) {
//  anchor_info.featmap_size = std::vector<float>{200, 200};
//  anchor_info.ranges.resize(7);
//  anchor_info.ranges[0] = std::vector<float>{-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795};
//  anchor_info.ranges[1] = std::vector<float>{-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365};
//  anchor_info.ranges[2] = std::vector<float>{-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504};
//  anchor_info.ranges[3] = std::vector<float>{-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111};
//  anchor_info.ranges[4] = std::vector<float>{-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072};
//  anchor_info.ranges[5] = std::vector<float>{-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986};
//  anchor_info.ranges[6] = std::vector<float>{-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965};
//  anchor_info.sizes.resize(7);
//  anchor_info.sizes[0] = std::vector<float>{1.95017717, 4.60718145, 1.72270761}; // car
//  anchor_info.sizes[1] = std::vector<float>{2.4560939, 6.73778078, 2.73004906}; // truck
//  anchor_info.sizes[2] = std::vector<float>{2.87427237, 12.01320693, 3.81509561}; // tailer
//  anchor_info.sizes[3] = std::vector<float>{0.60058911, 1.68452161, 1.27192197}; // bicycle
//  anchor_info.sizes[4] = std::vector<float>{0.66344886, 0.7256437, 1.75748069}; // pedestrian
//  anchor_info.sizes[5] = std::vector<float>{0.39694519, 0.40359262, 1.06232151};  // traffic_cone
//  anchor_info.sizes[6] = std::vector<float>{2.49008838, 0.48578221, 0.98297065};  // barrier
//  anchor_info.rotations = std::vector<float>{0, 1.57};
//  anchor_info.custom_values = std::vector<float>{0, 0};
//  anchor_info.align_corner = false;
//  anchor_info.scale = 1.0;
//
//}

static void build_anchor_info(AnchorInfo &anchor_info, const vitis::ai::proto::DpuModelParam& config) {
  auto &featmap_size = config.pointpillars_nus_param().featmap_size();
  auto &anchor_config = config.pointpillars_nus_param().anchor_info();
  std::copy(featmap_size.begin(), 
            featmap_size.end(),
            std::back_inserter(anchor_info.featmap_size));
  auto anchor_ranges_size = anchor_config.ranges_size();
  anchor_info.ranges.resize(anchor_ranges_size);
  for (auto i = 0; i < anchor_ranges_size; ++i) {
    std::copy(anchor_config.ranges(i).single_range().begin(), 
              anchor_config.ranges(i).single_range().end(),
              std::back_inserter(anchor_info.ranges[i]));
  }
  auto sizes_ranges_size = anchor_config.sizes_size();
  anchor_info.sizes.resize(sizes_ranges_size);
  for (auto i = 0; i < sizes_ranges_size; ++i) {
    std::copy(anchor_config.sizes(i).single_size().begin(), 
              anchor_config.sizes(i).single_size().end(),
              std::back_inserter(anchor_info.sizes[i]));
  }
  std::copy(anchor_config.rotations().begin(), 
            anchor_config.rotations().end(),
            std::back_inserter(anchor_info.rotations));
  std::copy(anchor_config.custom_value().begin(), 
            anchor_config.custom_value().end(),
            std::back_inserter(anchor_info.custom_values));
  anchor_info.align_corner = anchor_config.align_corner();
  anchor_info.scale = anchor_config.scale();
}

PointPillarsNuscenesPost::~PointPillarsNuscenesPost(){};
PointPillarsNuscenesPost::PointPillarsNuscenesPost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : input_tensors_(input_tensors),
      output_tensors_(output_tensors),
      num_classes_(config.pointpillars_nus_param().num_classes()),
      nms_pre_(config.pointpillars_nus_param().nms_pre()),
      nms_thresh_(config.pointpillars_nus_param().nms_thresh()),
      max_num_(config.pointpillars_nus_param().max_num()),
      score_thresh_(config.pointpillars_nus_param().score_thresh()) {
      //input_width_(input_tensors[0].width), // read from input tensor
      //input_height_(input_tensors[0].height){ // read from input tensor
  AnchorInfo anchor_info; //read from config 
  build_anchor_info(anchor_info, config);
  anchors_ = generate_anchors(anchor_info);
  bbox_code_size_ = 7 + anchor_info.custom_values.size();
  // config.pointpillars_nus_param().output_info()
  // score : [200, 200, 140]
  // bbox: [200, 200, 126]
  // dir: [200, 200, 28]
  for (auto i = 0u; i < output_tensors_.size(); ++i) {
    std::string name = config.pointpillars_nus_param().score_layer_name();
    //if (output_tensors_[i].name.find("conv_cls") != std::string::npos) {
    if (output_tensors_[i].name.find(name.c_str()) != std::string::npos) {
      output_score_index_ = i; // [200, 200, 140]
    }

    name = config.pointpillars_nus_param().bbox_layer_name();
    //if (output_tensors_[i].name.find("conv_reg") != std::string::npos) {
    if (output_tensors_[i].name.find(name) != std::string::npos) {
      output_bbox_index_ = i; // [200, 200, 126]
    }

    name = config.pointpillars_nus_param().dir_layer_name();
    //if (output_tensors_[i].name.find("conv_dir_cls") != std::string::npos) {
    if (output_tensors_[i].name.find(name) != std::string::npos) {
      output_dir_index_ = i; // [200, 200, 28]
    }
  }

}

PointPillarsNuscenesResult PointPillarsNuscenesPost::postprocess_internal_simple(unsigned int idx) {
__TIC__(PP_NUS_POST)
  // suppose no need to transform
  for (auto i = 0u; i < output_tensors_.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
          << "output tensors: " << i
          << " name: " << output_tensors_[i].name
          << " size: " << output_tensors_[i].size
          << " scale:" << vitis::ai::library::tensor_scale(output_tensors_[i]);
  }
  
  auto &score_layer = output_tensors_[output_score_index_];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "score_layer name:" << score_layer.name
        << " size:" << score_layer.size;
  auto &bbox_layer = output_tensors_[output_bbox_index_];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "bbox_layer name:" << bbox_layer.name
        << " size:" << bbox_layer.size;
  auto &dir_layer = output_tensors_[output_dir_index_];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "dir_layer name:" << dir_layer.name
        << " size:" << dir_layer.size;
  //int num_classes = 10;   // read from config
  auto batch_size = output_tensors_[0].batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "batch size:" << batch_size;

  // anchors_num: number of anchors
  // score_layer: [anchors_num, num_classes]
  // bbox_layer: [anchors_num, 7 + customer_data_size]
  // dir_layer : [anchors_num, 2]

__TIC__(PP_NUS_DIR_SELECT)
  //constexpr int DIR_LAST_DIM = 2;
  // 1. dir cls find max index
  // skip now
  auto dir_layer_ptr = (int8_t *)(dir_layer.get_data(idx));
__TOC__(PP_NUS_DIR_SELECT)

  // 2. scores sigmoid
  // skip now
__TIC__(PP_NUS_SIGMOID)
__TOC__(PP_NUS_SIGMOID)
__TIC__(PP_NUS_SCORE_SELECT_MAX)
  // 3. score find max (for select top k) 
  // skip now
__TOC__(PP_NUS_SCORE_SELECT_MAX)

  // 4. find top K result;
  // skip now
__TIC__(PP_NUS_TOP_K)
__TOC__(PP_NUS_TOP_K)

  // new step
__TIC__(TEST_SELECT_SCORE)
  //auto nms_pre_ = 1000; // read from config;
  auto score_length = score_layer.size / batch_size; // should be equal to anchors_num * num_classes
  auto score_layer_scale = vitis::ai::library::tensor_scale(score_layer);
  auto score_layer_ptr = (int8_t *)score_layer.get_data(idx);
  auto scores_group = score_length / num_classes_; // anchors num
  if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS)) {
    LOG(INFO) << "idx: " << idx
              << ", score length: " << score_length
              << ", score layer ptr: " << (void *)score_layer_ptr
              << ", num_classes_: " << num_classes_
              << ", score group: " << scores_group;
  }

  //float score_thresh = 0.f;
  if (ENV_PARAM(USE_DEBUG_SCORE)) {
    score_thresh_ = 0.05; // read from config
  } else if (ENV_PARAM(USE_DEBUG_SCORE2)) {
    score_thresh_ = 0.1; // read from config
  } else {
    //score_thresh_ = 0.3; // read from config
  }
  // y = ln(y/(1-y)) / scale
  float score_inter =std::log(score_thresh_/(1 - score_thresh_)) / score_layer_scale;
  int8_t score_int_thresh = std::floor(score_inter);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
         << "score_thresh : " << score_thresh_
         << " score_int_thresh : " << (int)score_int_thresh;

__TIC__(TEST_SELECT_SCORE_1)
  std::vector<std::vector<ScoreIndex>> score_indices(num_classes_);

  for (auto i = 0u; i < scores_group; ++i) {
    for (auto j = 0; j < num_classes_; ++j) {
      auto index  = i * num_classes_ +j;
      if (*(score_layer_ptr + index) > score_int_thresh) {
         score_indices[j].push_back(ScoreIndex{i, *(score_layer_ptr + index)});
         if (ENV_PARAM(DEBUG_NMS)) {
           LOG(INFO) << "find score: " << (int)(*(score_layer_ptr + index)) << " i: " << i << ", j: " << j;
         }
      } 
    }
  }
__TOC__(TEST_SELECT_SCORE_1)
__TIC__(TEST_SCORE_SORT)
  for (auto j = 0; j < num_classes_; ++j) {
    if (ENV_PARAM(DEBUG_NMS)) {
      LOG(INFO) << "score_incices[" << j <<"] size:" << score_indices[j].size();
    }
    std::stable_sort(score_indices[j].begin(), score_indices[j].end(),
                     [](const ScoreIndex &l, const ScoreIndex &r){return l.score >= r.score;});
    if (score_indices.size() > (uint32_t)nms_pre_) {
      score_indices.resize(nms_pre_);
    }
  }
__TOC__(TEST_SCORE_SORT)
__TOC__(TEST_SELECT_SCORE)
__TIC__(PP_NUS_DECODE)
  // 5. bbox decode
  // skip now
  //auto bbox_num = bbox_layer.size / batch_size / bbox_code_size; // should be equal to anchors_num
  //auto bbox_code_size = 9u; // read from config;
  auto bbox_layer_scale = vitis::ai::library::tensor_scale(bbox_layer); 
  auto bbox_layer_ptr = (int8_t*)(bbox_layer.get_data(idx));
  if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS)) {
    LOG(INFO) << "idx: " << idx 
              << ", bbox_code_size: " << bbox_code_size_
              << ", bbox_layer_ptr: " << (void *)bbox_layer_ptr;
  }
__TOC__(PP_NUS_DECODE)

__TIC__(PP_NUS_GET_BEV)
  // 6. get bev bboxes
  // skip now
__TOC__(PP_NUS_GET_BEV)

  // 7. 3d nms
__TIC__(PP_NUS_NMS)
  //auto max_num = 500; // read from config
  //auto nms_thresh = 0.2; // read from config

  // key: anchor_index, value: decoded bbox and bev (size = bbox_ndim + 5)
  std::map<int, std::vector<float>> bbox_decoded; 
  //auto nms_indexes = nms_3d_multiclasses(bboxes_bev, scores_topk, num_classes, score_thresh, nms_thresh, max_num);
  //
__TIC__(PP_NUS_NMS_API)
  // anchor index, label
  auto nms_indexes = nms_multiclasses_int8(bbox_layer_ptr, bbox_code_size_, bbox_layer_scale,
                                           anchors_, score_indices, bbox_decoded, num_classes_, 
                                           score_int_thresh, nms_thresh_, max_num_);
__TOC__(PP_NUS_NMS_API)
__TOC__(PP_NUS_NMS)

__TIC__(PP_NUS_MAKE_RESULT)
  // 8. make result
  PointPillarsNuscenesResult result;
  //result.width = input_width_;
  //result.height = input_height_;
  auto dir_offset = 0.7854; // read from config
  auto dir_limit_offset = 0.0; // read from config
  float pi = 3.1415926;
  for (auto i = 0u; i != nms_indexes.size(); ++i) {
    auto label = nms_indexes[i].second;
    auto index = score_indices[label][nms_indexes[i].first].index;
    auto dir_rot = limit_period(bbox_decoded[index][6] - dir_offset, dir_limit_offset, pi);
    //bbox_decoded[index][6] = dir_rot + dir_offset + pi * dir_max_index_topk[index]; 
    auto dir_value = *(dir_layer_ptr + index * 2) > *(dir_layer_ptr + index * 2 + 1) ? 0: 1; 
    bbox_decoded[index][6] = dir_rot + dir_offset + pi * dir_value; 
    auto score_ori = (int)(*(score_layer_ptr + index * num_classes_ + label)) * score_layer_scale; 
    float score = 1.0 / (1.0 + std::exp(-score_ori));
    result.bboxes.emplace_back(PPBbox{score, 
                                      std::vector<float>(bbox_decoded[index].begin(), 
                                                         bbox_decoded[index].begin() + bbox_code_size_), 
                                      label});
  } 
__TOC__(PP_NUS_MAKE_RESULT)
__TOC__(PP_NUS_POST)
  return result;
}
 

PointPillarsNuscenesResult PointPillarsNuscenesPost::postprocess_internal(unsigned int idx) {
__TIC__(PP_NUS_POST)
  // suppose no need to transform
  for (auto i = 0u; i < output_tensors_.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
          << "output tensors: " << i
          << " name: " << output_tensors_[i].name
          << " size: " << output_tensors_[i].size;
  }
  
  auto &score_layer = output_tensors_[output_score_index_];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "score_layer name:" << score_layer.name
        << " size:" << score_layer.size;
  auto &bbox_layer = output_tensors_[output_bbox_index_];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "bbox_layer name:" << bbox_layer.name
        << " size:" << bbox_layer.size;
  auto &dir_layer = output_tensors_[output_dir_index_];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "dir_layer name:" << dir_layer.name
        << " size:" << dir_layer.size;
  auto batch_size = output_tensors_[0].batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "batch size:" << batch_size;

  // output size of every batch 
  // anchors_num: number of anchors
  // score_layer: [anchors_num, num_classes]
  // bbox_layer: [anchors_num, 7 + customer_data_size]
  // dir_layer : [anchors_num, 2]

  // 1. dir cls find max index
__TIC__(PP_NUS_DIR_SELECT)
  constexpr int DIR_LAST_DIM = 2;
  std::vector<uint32_t>  dir_max_index(dir_layer.size / batch_size / DIR_LAST_DIM);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "dir_max_index size:" << dir_max_index.size();
  auto dir_layer_ptr = (int8_t *)(dir_layer.get_data(idx));
  for (auto i = 0u; i < dir_max_index.size(); ++i) {
    //dir_max_index[i] = std::distance(dir_layer_ptr + i * DIR_LAST_DIM, 
    //                                 std::max_element(dir_layer_ptr + i * DIR_LAST_DIM, 
    //                                                  dir_layer_ptr + (i+1) * DIR_LAST_DIM));
    dir_max_index[i] = *(dir_layer_ptr + i * 2) > *(dir_layer_ptr + i * 2 + 1) ? 0 : 1;
  }
__TOC__(PP_NUS_DIR_SELECT)

  // 2. scores sigmoid
__TIC__(PP_NUS_SIGMOID)
  auto score_length = score_layer.size / batch_size; // should be equal to anchors_num * num_classes
  auto score_layer_fix_pos = score_layer.fixpos;
  auto score_layer_scale = vitis::ai::library::tensor_scale(score_layer);
  auto score_layer_ptr = (int8_t *)score_layer.get_data(idx);
  auto scores_group = score_length / num_classes_; // anchors num
  if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS)) {
    LOG(INFO) << "idx: " << idx
              << ", score length: " << score_length
              << ", score layer ptr: " << (void *)score_layer_ptr
              << ", score layer fixpos: " << score_layer_fix_pos
              << ", num_classes_: " << num_classes_
              << ", score group: " << scores_group;
  }

__TIC__(PP_NUS_SIGMOID_BUFFER_INIT)
  std::vector<float> scores_sigmoid(score_length);
__TOC__(PP_NUS_SIGMOID_BUFFER_INIT)
__TIC__(PP_NUS_SIGMOID_TABLE)
  sigmoid_table(score_layer_ptr, score_layer_fix_pos, num_classes_, scores_group, scores_sigmoid.data()); 
__TOC__(PP_NUS_SIGMOID_TABLE)
__TOC__(PP_NUS_SIGMOID)

__TIC__(PP_NUS_SELECT_MAX)
  // 3. score find max (for select top k) 
  std::vector<float> score_max(score_length / num_classes_); // should be equal to anchors_num
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "score_max size:" << score_max.size();
  for (auto i = 0u; i < score_max.size(); ++i) {
    score_max[i] = *(std::max_element(scores_sigmoid.begin() + i * num_classes_, 
				    scores_sigmoid.begin() + (i+1) * num_classes_));
  }
__TOC__(PP_NUS_SELECT_MAX)

  // 4. find top K result;
__TIC__(PP_NUS_TOP_K)
  //auto nms_pre_ = 1000; // read from config;
  //float score_thresh_ = 0.f;
  if (ENV_PARAM(USE_DEBUG_SCORE)) {
    score_thresh_ = 0.05; // read from config
  } else if (ENV_PARAM(USE_DEBUG_SCORE2)) {
    score_thresh_ = 0.1; // read from config
  } else {
    //score_thresh_ = 0.3; // read from config
  }
  // y = ln(y/(1-y)) / scale
  float score_inter =std::log(score_thresh_/(1 - score_thresh_)) / score_layer_scale;
  int8_t score_int_thresh = std::floor(score_inter);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
         << "score thresh : " << score_thresh_
         << "score inter: " << score_inter
         << " score_int_thresh : " << (int)score_int_thresh;


  // top k indexes should be sorted
  auto topk_idxes = topK_indexes(score_max, nms_pre_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS)) // check and some items has same value but order different 
        << "topk indexes size:" << topk_idxes.size();
__TOC__(PP_NUS_TOP_K)
__TIC__(PP_NUS_DECODE)
  // 5. bbox decode
  //auto bbox_code_size = 9u; // read from config;
  if (ENV_PARAM(DEBUG_NMS)) {
    LOG(INFO) << "bbox_code_size: " << bbox_code_size_;
  }
  //auto bbox_num = bbox_layer.size / batch_size / bbox_code_size; // should be equal to anchors_num
  auto bbox_layer_scale = vitis::ai::library::tensor_scale(bbox_layer); 
  auto bbox_layer_ptr = (int8_t*)(bbox_layer.get_data(idx));
  //auto bboxes = bbox_decode(valid_indexes, anchors_, (int8_t*)(bbox_layer.get_data(idx)), bbox_code_size, bbox_layer_scale); 
  auto bboxes = bbox_decode_test(topk_idxes, anchors_, bbox_layer_ptr, bbox_code_size_, bbox_layer_scale);  
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS))
        << "decode bboxes size:" << bboxes.size();
__TOC__(PP_NUS_DECODE)

__TIC__(PP_NUS_GET_BEV)
  // 6. get bev bboxes
  //auto bboxes_bev = get_bboxes_for_nms(bboxes, bbox_code_size);
  auto bboxes_bev = get_bboxes_for_nms_test(bboxes, bbox_code_size_);
__TOC__(PP_NUS_GET_BEV)

  // 7. 3d nms
__TIC__(PP_NUS_NMS)
  //auto max_num = 500; // read from config
  //auto nms_thresh = 0.2; // read from config
  std::vector<std::vector<float>> scores_topk(num_classes_);
  for (auto i = 0; i < num_classes_; ++i) {
    scores_topk[i].resize(topk_idxes.size());
    for (auto j = 0u; j < topk_idxes.size(); ++j) {
      scores_topk[i][j] = scores_sigmoid[topk_idxes[j] * num_classes_ + i];
    }
  }
  std::vector<uint32_t> dir_max_index_topk(topk_idxes.size());
  for (auto i = 0u; i < topk_idxes.size(); ++i) {
    dir_max_index_topk[i] = dir_max_index[topk_idxes[i]];
  }

__TIC__(PP_NUS_NMS_API)
  auto nms_indexes = nms_3d_multiclasses(bboxes_bev, scores_topk, num_classes_, score_thresh_, nms_thresh_, max_num_);
__TOC__(PP_NUS_NMS_API)
__TOC__(PP_NUS_NMS)

__TIC__(PP_NUS_MAKE_RESULT)
  // 8. make result
  PointPillarsNuscenesResult result;
  //result.width = input_width_;
  //result.width = input_height_;
  auto dir_offset = 0.7854; // read from config
  auto dir_limit_offset = 0.0; // read from config
  float pi = 3.1415926;
  for (auto i = 0u; i != nms_indexes.size(); ++i) {
    auto index = nms_indexes[i].first;
    auto label = nms_indexes[i].second;
    auto dir_rot = limit_period(bboxes[index][6] - dir_offset, dir_limit_offset, pi);
    bboxes[index][6] = dir_rot + dir_offset + pi * dir_max_index_topk[index]; 
    result.bboxes.emplace_back(PPBbox{scores_topk[label][index], 
                               bboxes[index], label});
  } 
__TOC__(PP_NUS_MAKE_RESULT)
__TOC__(PP_NUS_POST)
  return result;
}
 
std::vector<PointPillarsNuscenesResult> PointPillarsNuscenesPost::postprocess(size_t batch_size) {
  __TIC__(PP_NUS_total_batch)
  auto ret = std::vector<PointPillarsNuscenesResult>{};
  for (auto i = 0u; i < batch_size; ++i) {
    //if (ENV_PARAM(DEBUG_SIMPLE_API)) {
    if (ENV_PARAM(DEBUG_OLD_API)) {
      ret.emplace_back(postprocess_internal(i));
    } else {
      ret.emplace_back(postprocess_internal_simple(i));
    }
  }
  __TOC__(PP_NUS_total_batch)
  return ret;
}
 
}}
