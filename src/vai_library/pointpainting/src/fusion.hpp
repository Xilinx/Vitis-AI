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
#include <opencv2/core.hpp>
#include <vitis/ai/pointpillars_nuscenes.hpp> // for CamInfo and PointsInfo

using namespace vitis::ai::pointpillars_nus;

namespace vitis { namespace ai {
namespace pointpainting {

//struct CamInfo{
//  uint64_t timestamp;
//  std::array<float, 3> s2l_t; // sensor to lidar translation
//  std::array<float, 9> s2l_r; // sensor to lidar rotation 
//  std::array<float, 9> cam_intr; // camera intrinsic
//  
//};

//cv::Mat build_cam2lidar(const CamInfo &cam_info);

std::vector<float> fusion(const std::vector<CamInfo> &cam_infos, 
                          const std::vector<float> &points,
                          int dim,
                          const std::vector<cv::Mat> &images,
                          int num_classes);

}}}
