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

#include "./anchor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

namespace vitis {
namespace ai {
namespace efficientdet_d2 {

DEF_ENV_PARAM(DEBUG_ANCHOR, "0");

Anchor::Anchor(const AnchorConfig& config) : config_(config){};

void Anchor::generate_boxes() {
  int level_num = config_.max_level - config_.min_level + 1;
  LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR)) << "level_num:" << level_num;
  //assert(config_.aspect_ratios.size() > 0);
  //assert(config_.anchor_scales.size() == level_num);

  int feat_w = config_.image_width;
  int feat_h = config_.image_height;

  for (auto l = config_.min_level; l <= config_.max_level; ++l) {
    auto level = l;
    feat_w = config_.image_width / std::round(std::exp2(level));
    feat_h = config_.image_height / std::round(std::exp2(level));
    LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR))
        << "feat_w:" << feat_w << " , feat_h:" << feat_h;
    int stride_w = config_.image_width / feat_w;
    int stride_h = config_.image_height / feat_h;
    int mesh_width = (config_.image_width + 0.5 * stride_w) / stride_w;
    int mesh_height = (config_.image_height + 0.5 * stride_h) / stride_h;
    LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR))
        << "mesh_width:" << mesh_width << " , mesh_height:" << mesh_height;
    int num = config_.num_scales * config_.aspect_ratios.size();
    // LevelBoxes boxes(mesh_width * mesh_height * num, std::vector<float>(4));
    leveled_bboxes_[level] = std::make_shared<LevelBoxes>(
        mesh_width * mesh_height * num, std::vector<float>(4));
    auto& boxes = *leveled_bboxes_[level];
    for (auto scale_octave = 0; scale_octave < config_.num_scales;
         ++scale_octave) {
      for (auto ii = 0u; ii < config_.aspect_ratios.size(); ++ii) {
        // for (auto& aspect : config_.aspect_ratios) {
        int level_idx = scale_octave * config_.aspect_ratios.size() + ii;
        LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR))
            << "num:" << num << ", level_idx:" << level_idx;
        float octave_scale = ((float)scale_octave) / config_.num_scales;
        float anchor_scale = config_.anchor_scales[level - config_.min_level];
        float aspect = config_.aspect_ratios[ii];
        LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR))
            << "stride_w:" << stride_w << " stride_h:" << stride_h
            << " octave_scale:" << octave_scale << " aspect:" << aspect
            << " anchor_scale:" << anchor_scale;

        auto base_anchor_size_x =
            anchor_scale * stride_w * std::exp2(octave_scale);
        auto base_anchor_size_y =
            anchor_scale * stride_h * std::exp2(octave_scale);
        auto aspect_x = std::sqrt(aspect);
        auto aspect_y = 1.f / aspect_x;
        auto anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0;
        auto anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0;

        for (auto y = stride_h / 2; y < config_.image_height;
             y = y + stride_h) {
          for (auto x = stride_w / 2; x < config_.image_width;
               x = x + stride_w) {
            int index =
                (y / stride_h * mesh_width + x / stride_w) * num + level_idx;
            // LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR))
            //    << "(y / stride_h):" << y / stride_h
            //    << ", mesh_width:" << mesh_width
            //    << ", (x / stide_w):" << x / stride_w;
            boxes[index][0] = y - anchor_size_y_2;
            boxes[index][1] = x - anchor_size_x_2;
            boxes[index][2] = y + anchor_size_y_2;
            boxes[index][3] = x + anchor_size_x_2;

            LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR))
                << "y:" << y << ", x:" << x << ", index:" << index
                << ", boxes: [" << boxes[index][0] << "," << boxes[index][1]
                << "," << boxes[index][2] << "," << boxes[index][3] << "]";
          }
        }
      }
    }
  }
}

std::vector<std::vector<float>> Anchor::generate_boxes_(const int stride_w,
                                                        const int stride_h,
                                                        float octave_scale,
                                                        float aspect,
                                                        float anchor_scale) {
  int mesh_width = (config_.image_width + stride_w) / stride_w;
  int mesh_height = (config_.image_height + stride_h) / stride_h;

  LevelBoxes boxes(mesh_width * mesh_height, std::vector<float>(4));

  auto base_anchor_size_x = anchor_scale * stride_w * std::exp2(octave_scale);
  auto base_anchor_size_y = anchor_scale * stride_h * std::exp2(octave_scale);
  auto aspect_x = std::sqrt(aspect);
  auto aspect_y = 1.f / aspect_x;
  auto anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0;
  auto anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0;
  for (auto y = stride_h / 2; y < config_.image_height; y = y + stride_h) {
    for (auto x = stride_w / 2; x < config_.image_width; x = x + stride_w) {
      int index = y / stride_h * mesh_width + x / stride_w;
      boxes[index][0] = y - anchor_size_y_2;
      boxes[index][1] = x - anchor_size_x_2;
      boxes[index][2] = y + anchor_size_y_2;
      boxes[index][3] = x + anchor_size_x_2;
      LOG_IF(INFO, ENV_PARAM(DEBUG_ANCHOR))
          << "y:" << y << ", x:" << x << ", boxes: [" << boxes[index][0] << ","
          << boxes[index][1] << "," << boxes[index][2] << "," << boxes[index][3]
          << "]";
    }
  }
  return boxes;
}

std::shared_ptr<Anchor::LevelBoxes> Anchor::get_boxes(int level) {
  if (leveled_bboxes_.count(level) == 0) {
    return nullptr;
  } else {
    return leveled_bboxes_[level];
  }
}

}  // namespace efficientdet_d2
}  // namespace ai
}  // namespace vitis
