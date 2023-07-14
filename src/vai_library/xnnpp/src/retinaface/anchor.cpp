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

#include <iostream>
#include <string>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./anchor.hpp"

using namespace std;

DEF_ENV_PARAM(DEBUG_XNNPP_RETINAFACE, "0")
DEF_ENV_PARAM(DEBUG_XNNPP_RETINAFACE_ANCHOR, "0")

namespace vitis {
namespace ai {
namespace retinaface {

static std::vector<std::vector<float>> generate_anchor_fpn(const AnchorInfo anchor_param) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE_ANCHOR)) << "generate_anchors_fpn";
  // topx, topy, bottomx, bottomy
  std::array<int, 4> base_anchor{0, 0, anchor_param.base_size - 1, anchor_param.base_size - 1};
  float w = base_anchor[2] - base_anchor[0] + 1;
  float h = base_anchor[3] - base_anchor[1] + 1;
  float center_x = base_anchor[0] + 0.5 * (w - 1);
  float center_y = base_anchor[1] + 0.5 * (h - 1);
  auto size = w * h;

  std::vector<std::vector<float>> ratio_anchors(anchor_param.ratios.size());
  for (auto i = 0u; i < anchor_param.ratios.size(); ++i) {
    auto size_ratio = size / anchor_param.ratios[i];
    auto ws = std::round(std::sqrt(size_ratio));
    auto hs = ws * anchor_param.ratios[i];
    ratio_anchors[i] = std::vector<float>{float(center_x - 0.5 * (ws - 1)),
                                          float(center_y - 0.5 * (hs - 1)),
                                          float(center_x + 0.5 * (hs - 1)),
                                          float(center_y + 0.5 * (hs - 1))};

  }
  for (auto i = 0u; i < ratio_anchors.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XNNPP_RETINAFACE_ANCHOR))
              << "ratio anchors[" << i << "] "
              << ratio_anchors[i][0] << ", "
              << ratio_anchors[i][1] << ", "
              << ratio_anchors[i][2] << ", "
              << ratio_anchors[i][3];
  }
  //std::vector<std::vector<int>> anchors(ratio_anchors.size() * anchor_param.scales.size());
  std::vector<std::vector<float>> anchors;
  for (auto i = 0u; i < ratio_anchors.size(); ++i) {
    auto w = ratio_anchors[i][2] - ratio_anchors[i][0] + 1; 
    auto h = ratio_anchors[i][3] - ratio_anchors[i][1] + 1;
    auto center_x = ratio_anchors[i][0] + 0.5 * (w -1);
    auto center_y = ratio_anchors[i][0] + 0.5 * (h -1);
    for (auto j = 0u; j < anchor_param.scales.size(); ++j) {
      auto ws = w * anchor_param.scales[j];
      auto hs = h * anchor_param.scales[j];
      anchors.emplace_back(std::vector<float>{(float)(center_x - 0.5 * (ws - 1)),
                                              (float)(center_y - 0.5 * (hs - 1)),
                                              (float)(center_x + 0.5 * (hs - 1)),
                                              (float)(center_y + 0.5 * (hs - 1))});
    }
  }
  return anchors; 
}

static std::vector<std::vector<float>> 
anchor_plane(int h, int w, int stride, const std::vector<std::vector<float>> anchor_fpn) {
  auto anchor_fpn_size = anchor_fpn.size();
  auto size = h * w * anchor_fpn_size;
  std::vector<std::vector<float>> anchors(size);
  for (auto ww = 0; ww < w; ww++) {
    for (auto hh = 0; hh < h; hh++) {
      for (auto i = 0u; i < anchor_fpn_size; i++) {
        auto index = hh * w * anchor_fpn_size + ww * anchor_fpn_size + i;
        anchors[index] = std::vector<float>{anchor_fpn[i][0] + ww * stride,
                                          anchor_fpn[i][1] + hh * stride,
                                          anchor_fpn[i][2] + ww * stride,
                                          anchor_fpn[i][3] + hh * stride};
      }
    }
  }
  return anchors;
}


std::vector<std::vector<float>> 
generate_anchors(int input_width, int input_height, const std::vector<AnchorInfo> &params) {
//static std::vector<std::vector<int>> generate_anchors() {
  //std::vector<AnchorInfo> params = std::vector<AnchorInfo>{AnchorInfo{32, 16, {1.0}, {32, 16}}};
  //std::vector<AnchorInfo> params = std::vector<AnchorInfo>{AnchorInfo{32, 16, {1.0}, {32, 16}},
  //                                                           AnchorInfo{16, 16, {1.0}, {8, 4}},
  //                                                           AnchorInfo{8, 16, {1.0}, {2, 1}} };

  std::vector<std::vector<float>> anchors_fpns;
  std::vector<std::vector<float>> anchors;
  for (auto i = 0u; i < params.size(); ++i) {
    auto anchor_fpn = generate_anchor_fpn(params[i]);
    auto anchors_by_stride = anchor_plane(input_height / params[i].stride, input_width / params[i].stride, params[i].stride, anchor_fpn);
    //anchors_fpns.insert(anchors_fpns.end(), anchor_fpn.begin(), anchor_fpn.end());
    anchors.insert(anchors.end(), anchors_by_stride.begin(), anchors_by_stride.end());
  }
  return anchors;
}

std::vector<std::vector<float>> 
generate_anchors(int input_width, int input_height, const std::map<int32_t, StrideLayers, std::greater<int32_t>> &params) {
//static std::vector<std::vector<int>> generate_anchors() {
  //std::vector<AnchorInfo> params = std::vector<AnchorInfo>{AnchorInfo{32, 16, {1.0}, {32, 16}}};
  //std::vector<AnchorInfo> params = std::vector<AnchorInfo>{AnchorInfo{32, 16, {1.0}, {32, 16}},
  //                                                           AnchorInfo{16, 16, {1.0}, {8, 4}},
  //                                                           AnchorInfo{8, 16, {1.0}, {2, 1}} };

  if (ENV_PARAM(DEBUG_XNNPP_RETINAFACE) || ENV_PARAM(DEBUG_XNNPP_RETINAFACE_ANCHOR)) {
    LOG(INFO) << "generate anchors"; 
  }
  std::vector<std::vector<float>> anchors_fpns;
  std::vector<std::vector<float>> anchors;
  //for (auto i = 0u; i < params.size(); ++i) {
  for (auto it = params.begin(); it != params.end(); ++it) {
    auto anchor_fpn = generate_anchor_fpn(it->second.anchor_info);
    auto anchors_by_stride = anchor_plane(input_height / it->first,
                                          input_width / it->first, 
                                          it->first, anchor_fpn);
    //anchors_fpns.insert(anchors_fpns.end(), anchor_fpn.begin(), anchor_fpn.end());
    anchors.insert(anchors.end(), anchors_by_stride.begin(), anchors_by_stride.end());
  }
  return anchors;
}

}}}
