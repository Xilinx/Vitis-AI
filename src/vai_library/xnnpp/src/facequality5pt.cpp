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
#include "vitis/ai/nnpp/facequality5pt.hpp"

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(DEBUG_XNNPP_FACEQUALITY5PT, "0")
using namespace std;
namespace vitis {
namespace ai {

static float mapped_quality_day(float original_score) {
  return 1.0f / (1.0f + std::exp(-((3.0f * original_score - 600.0f) / 150.0f)));
}

static float mapped_quality_night(float original_score) {
  return 1.0f / (1.0f + std::exp(-((3.0f * original_score - 400.0f) / 150.0f)));
}

FaceQuality5ptResult face_quality5pt_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx, bool day) {
  const int input_width = input_tensors[0][0].width;
  const int input_height = input_tensors[0][0].height;

  vitis::ai::library::OutputTensor point_layer = output_tensors[0][0];
  vitis::ai::library::OutputTensor quality_layer = output_tensors[0][1];
  if (!config.face_quality5pt_param().point_layer_name().empty()) {
    auto key = config.face_quality5pt_param().point_layer_name();
    for (auto i = 0u; i < output_tensors[0].size(); ++i) {
      if (output_tensors[0][i].name.find(key) != std::string::npos) {
        point_layer = output_tensors[0][i];
        LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_FACEQUALITY5PT))
            << "find point layer : " << point_layer.name << ", index :" << i;
        break;
      }
    }
  }

  if (!config.face_quality5pt_param().quality_layer_name().empty()) {
    auto key = config.face_quality5pt_param().quality_layer_name();
    for (auto i = 0u; i < output_tensors[0].size(); ++i) {
      if (output_tensors[0][i].name.find(key) != std::string::npos) {
        quality_layer = output_tensors[0][i];
        LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_FACEQUALITY5PT))
            << "find quality layer : " << quality_layer.name
            << ", index :" << i;
        break;
      }
    }
  }

  // 5 points
  auto points = std::unique_ptr<std::array<std::pair<float, float>, 5>>(
      new std::array<std::pair<float, float>, 5>());
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_FACEQUALITY5PT))
      << " point layer name " << point_layer.name
      << ", scale : " << vitis::ai::library::tensor_scale(point_layer);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_FACEQUALITY5PT))
      << " quality layer name " << quality_layer.name
      << ", scale : " << vitis::ai::library::tensor_scale(quality_layer);

  for (auto i = 0u; i < points->size(); i++) {
    auto x = (float)(((int8_t*)point_layer.get_data(batch_idx))[i]) *
             vitis::ai::library::tensor_scale(point_layer) / input_width;
    auto y = (float)(((int8_t*)point_layer.get_data(batch_idx))[i + 5]) *
             vitis::ai::library::tensor_scale(point_layer) / input_height;
    (*points)[i] = std::make_pair(x, y);
  }

  // quality output
  float score_original = ((int8_t*)quality_layer.get_data(batch_idx))[0] *
                         vitis::ai::library::tensor_scale(quality_layer);
  float score = score_original;

  // if set original_quality = true, return original score,
  // else return mapped score;
  if (!config.face_quality5pt_param().original_quality()) {
    if (day) {
      score = mapped_quality_day(score_original);
    } else {
      score = mapped_quality_night(score_original);
    }
  }
  return FaceQuality5ptResult{input_width, input_height, score, *points};
}

// FaceQuality5ptResult face_quality5pt_post_process_original(
//    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
//        input_tensors,
//    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
//        output_tensors,
//    const vitis::ai::proto::DpuModelParam& config,
//    size_t batch_idx) {
//  const int input_width = input_tensors[0][0].width;
//  const int input_height = input_tensors[0][0].height;
//  // 5 points
//  auto points = std::unique_ptr<std::array<std::pair<float, float>, 5>>(
//      new std::array<std::pair<float, float>, 5>());
//  for (auto i = 0u; i < points->size(); i++) {
//    auto x = (float)(((int8_t*)output_tensors[0][0].get_data(batch_idx))[i]) *
//             vitis::ai::library::tensor_scale(output_tensors[0][0]) /
//             input_width;
//    auto y = (float)(((int8_t*)output_tensors[0][0].get_data(batch_idx))[i +
//    5]) *
//             vitis::ai::library::tensor_scale(output_tensors[0][0]) /
//             input_height;
//    (*points)[i] = std::make_pair(x, y);
//  }
//
//  // quality output
//  float score_original =
//  ((int8_t*)output_tensors[0][1].get_data(batch_idx))[0] *
//                         vitis::ai::library::tensor_scale(output_tensors[0][1]);
//  return FaceQuality5ptResult{input_width, input_height, score_original,
//                              *points};
//}

std::vector<FaceQuality5ptResult> face_quality5pt_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, bool day) {
  auto batch_size = input_tensors[0][0].batch;
  auto ret = std::vector<FaceQuality5ptResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    auto r = face_quality5pt_post_process(input_tensors, output_tensors, config,
                                          i, day);
    ret.emplace_back(r);
  }
  return ret;
}

// std::vector<FaceQuality5ptResult> face_quality5pt_post_process_original(
//    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
//        input_tensors,
//    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
//        output_tensors,
//    const vitis::ai::proto::DpuModelParam& config){
//    auto batch_size = input_tensors[0][0].batch;
//    auto ret = std::vector<FaceQuality5ptResult>{};
//    ret.reserve(batch_size);
//    for (auto i = 0u; i < batch_size; i++) {
//      auto r = face_quality5pt_post_process_original(input_tensors,
//      output_tensors,
//                                          config, i);
//      ret.emplace_back(r);
//    }
//    return ret;
//}

}  // namespace ai
}  // namespace vitis
