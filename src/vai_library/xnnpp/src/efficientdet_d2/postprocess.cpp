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

#include "./postprocess.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "object_detection_base.hpp"

using namespace vitis::ai::object_detection_base;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_EFFICIENTDET_D2, "0")
DEF_ENV_PARAM(DEBUG_EFFICIENTDET_D2_DECODE, "0")

EfficientDetD2Post::~EfficientDetD2Post(){};

EfficientDetD2Post::EfficientDetD2Post(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : num_classes_(config.efficientdet_d2_param().num_classes()),
      min_level_(config.efficientdet_d2_param().anchor_info().min_level()),
      max_level_(config.efficientdet_d2_param().anchor_info().max_level()),
      score_thresh_(config.efficientdet_d2_param().conf_threshold()),
      nms_thresh_(config.efficientdet_d2_param().nms_threshold()),
      pre_nms_num_(config.efficientdet_d2_param().pre_nms_num()),
      max_output_num_(config.efficientdet_d2_param().nms_output_num()),
      input_tensors_(input_tensors),
      output_tensors_(output_tensors) {
  vitis::ai::efficientdet_d2::Anchor::AnchorConfig anchor_config;
  auto& anchor_info = config.efficientdet_d2_param().anchor_info();
  anchor_config.min_level = min_level_;
  anchor_config.max_level = max_level_;
  anchor_config.num_scales = anchor_info.num_scales();
  anchor_config.anchor_scales = std::vector<float>(
      anchor_info.anchor_scales().begin(), anchor_info.anchor_scales().end());
  anchor_config.aspect_ratios = std::vector<float>(
      anchor_info.aspect_ratio().begin(), anchor_info.aspect_ratio().end());
  anchor_config.image_width = input_tensors[0].width;
  anchor_config.image_height = input_tensors[0].height;

  anchor_ = std::make_shared<efficientdet_d2::Anchor>(anchor_config);

  anchor_->generate_boxes();
  for (auto l = min_level_; l <= max_level_; l++) {
    const auto anchor_boxes = *(anchor_->get_boxes(l));
    LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
        << "l:" << l << " shape:" << anchor_boxes.size();
    // if (ENV_PARAM(ENABLE_EFFICIENTDET_D2_DEBUG)) {
    //  for (auto i = 0u; i < anchor_boxes.size(); ++i) {
    //    LOG(INFO) << "l:" << l << ", i:" << i << ", boxes: ["
    //              << anchor_boxes[i][0] << "," << anchor_boxes[i][1] << ","
    //              << anchor_boxes[i][2] << "," << anchor_boxes[i][3] << "]";
    //  }
    //}
  }
  for (auto i = 0u; i < output_tensors_.size(); ++i) {
    std::string name = output_tensors[i].name;
    // LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2)) << "name:" << name;
    bool find = false;
    for (auto& info : config.efficientdet_d2_param().output_info()) {
      // LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
      //    << "info.name:" << info.name() << ", level:" << info.level();
      if (name.find(info.name()) != std::string::npos) {
        if (info.type() == 1) {  // conf
          cls_output_layers_[info.level()] = output_tensors_[i];
          find = true;
        } else if (info.type() == 2) {  // bbox
          bbox_output_layers_[info.level()] = output_tensors_[i];
          find = true;
        }
      }
      if (find) {
        break;
      }
    }
  }
  if (ENV_PARAM(DEBUG_EFFICIENTDET_D2)) {
    for (auto it = bbox_output_layers_.begin(); it != bbox_output_layers_.end();
         ++it) {
      LOG(INFO) << "bbox output layer level:" << it->first
                << ", tensor info:" << it->second.name;
    }
    for (auto it = cls_output_layers_.begin(); it != cls_output_layers_.end();
         ++it) {
      LOG(INFO) << "cls output layer level:" << it->first
                << ", tensor info:" << it->second.name;
    }
  }
}

std::vector<vitis::ai::EfficientDetD2Result> EfficientDetD2Post::postprocess(
    size_t batch_size, const std::vector<int>& swidths,
    const std::vector<int>& sheights, const std::vector<float>& image_scales) {
  __TIC__(EfficientDetD2_POST_BATCH)
  auto ret = std::vector<vitis::ai::EfficientDetD2Result>{};

  for (auto i = 0u; i < batch_size; ++i) {
    ret.emplace_back(
        postprocess_kernel(i, swidths[i], sheights[i], image_scales[i]));
  }
  __TOC__(EfficientDetD2_POST_BATCH)
  return ret;
}

vitis::ai::EfficientDetD2Result EfficientDetD2Post::postprocess_kernel(
    size_t batch_idx, int swidth, int sheight, float image_scale) {
  __TIC__(EfficientDetD2_post)
  //  auto batch_size = input_tensors_[0].batch;
  int num_classes = num_classes_;
  int min_level = min_level_;
  int max_level = max_level_;
  float score_thresh = score_thresh_;
  float nms_thresh = nms_thresh_;
  int pre_nms_num = pre_nms_num_;
  int max_output_num = max_output_num_;

  // 1. pre nms
  __TIC__(SELECT)
  std::vector<SelectedOutput> all_selected;
  std::vector<float> box_scales = {0.015625, 0.0078125, 0.0078125, 0.0078125,
                                   0.0078125};
  std::vector<float> cls_scales = {0.125, 0.125, 0.125, 0.125, 0.125};
  for (auto i = min_level; i <= max_level; ++i) {
    int8_t* box_output_ptr =
        (int8_t*)bbox_output_layers_[i].get_data(batch_idx);
    auto box_output_scale =
        vitis::ai::library::tensor_scale(bbox_output_layers_[i]);
    int8_t* cls_output_ptr = (int8_t*)cls_output_layers_[i].get_data(batch_idx);
    auto cls_output_scale =
        vitis::ai::library::tensor_scale(cls_output_layers_[i]);
    auto box_length = 4;
    auto size = cls_output_layers_[i].size / cls_output_layers_[i].batch;
    // vitis::ai::library::tensor_scale(bbox_output_layers_[i]);
    // vitis::ai::library::tensor_scale(cls_output_layers_[i]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
        << "level:" << i << ", size: " << cls_output_layers_[i].size
        << ", batch: " << cls_output_layers_[i].batch;

    // x = ln(y/(1-y))
    float score_inter = std::log(score_thresh_ / (1 - score_thresh_));
    int8_t score_thresh_int8 = std::floor(score_inter / cls_output_scale);
    // for (auto ii = 0u; ii < size; ++ii) {
    //  if (*(cls_output_ptr + ii) >= score_thresh_int8) {
    //    LOG(INFO) << "level:" << i << ", i: " << ii
    //              << ", score:" << (int)*(cls_output_ptr + ii);
    //  }
    //}
    LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
        << "level:" << i << ", box scale: " << box_output_scale
        << ", cls scale:" << cls_output_scale
        << " score thresh int8:" << (int)score_thresh_int8;
    auto selected =
        select(i, num_classes, box_output_ptr, box_output_scale, box_length,
               cls_output_ptr, cls_output_scale, size, score_thresh_int8);
    LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
        << "selected size:" << selected.size();
    std::copy(selected.begin(), selected.end(),
              std::back_inserter(all_selected));
  }
  __TOC__(SELECT)
  LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
      << "all selected size:" << all_selected.size();
  __TIC__(TOP_K)
  auto top_selected = topK(all_selected, pre_nms_num);
  // auto top_selected = topK2(all_selected, pre_nms_num);
  __TOC__(TOP_K)
  LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
      << "top k size:" << top_selected.size();

  __TIC__(DECODE)
  std::vector<DecodedOutput> top_decoded(top_selected.size());
  for (auto i = 0u; i < top_selected.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2_DECODE))
        << "top i:" << i << ", level:" << top_selected[i].level
        << ", index:" << top_selected[i].index;
    auto anchor_boxes = anchor_->get_boxes(top_selected[i].level);
    if (anchor_boxes) {
      auto& anchor_box = (*anchor_boxes)[top_selected[i].index];
      LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2_DECODE))
          << "anchor_box: [" << anchor_box[0] << ", " << anchor_box[1] << ", "
          << anchor_box[2] << ", " << anchor_box[3] << ", selected_box: ["
          << (int)top_selected[i].pbox[0] << ", "
          << (int)top_selected[i].pbox[1] << ", "
          << (int)top_selected[i].pbox[2] << ", "
          << (int)top_selected[i].pbox[3];
      top_decoded[i] = decode(top_selected[i], anchor_box);
    }
  }
  __TOC__(DECODE)

  // 2. pre class nms
  if (ENV_PARAM(DEBUG_EFFICIENTDET_D2_DECODE)) {
    for (auto i = 0u; i < top_decoded.size(); ++i) {
      LOG(INFO) << "decoded:" << i << " cls:" << top_decoded[i].cls
                << " box:" << top_decoded[i].bbox[0] << " "
                << top_decoded[i].bbox[1] << " " << top_decoded[i].bbox[2]
                << " " << top_decoded[i].bbox[3]
                << " score:" << top_decoded[i].score;
    }
  }
  __TIC__(NMS)
  auto ori_result = per_class_nms(top_decoded, num_classes, nms_thresh,
                                  score_thresh, max_output_num);
  __TOC__(NMS)
  LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
      << "ori result size:" << ori_result.size();

  // make result
  EfficientDetD2Result result;
  result.width = swidth;
  result.height = sheight;
  // auto tensor_width = input_tensors_[0].width;
  // auto tensor_height = input_tensors_[0].height;
  result.bboxes.resize(ori_result.size());
  for (auto i = 0u; i < ori_result.size(); ++i) {
    result.bboxes[i].label = ori_result[i].cls;
    result.bboxes[i].score = ori_result[i].score;
    result.bboxes[i].x = ori_result[i].bbox[1] / image_scale / result.width;
    result.bboxes[i].y = ori_result[i].bbox[0] / image_scale / result.height;
    result.bboxes[i].width = (ori_result[i].bbox[3] - ori_result[i].bbox[1]) /
                             image_scale / result.width;
    result.bboxes[i].height = (ori_result[i].bbox[2] - ori_result[i].bbox[0]) /
                              image_scale / result.height;
  }

  __TOC__(EfficientDetD2_post)
  return result;
}

}  // namespace ai
}  // namespace vitis
