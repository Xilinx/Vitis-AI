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
#include <array>
#include <vector>
#include <cassert>
#include <utility>
#include <opencv2/core.hpp>
#include <vitis/ai/pointpillars_nuscenes.hpp>

namespace vitis { namespace ai {
namespace pointpillars_nus {

PointsInfo remove_useless_dim(const PointsInfo &points_info, 
                              int invalid_channel);

std::shared_ptr<std::vector<float>> 
points_filter(const std::shared_ptr<std::vector<float>> &points, 
              int dim, const std::vector<float> &range);
std::vector<float> points_filter(const std::vector<float> &points, int dim, const std::vector<float> &range);

PointsInfo 
points_filter(const PointsInfo &points_info, 
              const std::vector<float> &range);

std::vector<float> multi_frame_fusion(const PointsInfo &frame_info, 
                                      const std::vector<SweepInfo> &sweeps_infos);

}}}
