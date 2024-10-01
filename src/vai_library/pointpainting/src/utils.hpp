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

#include <vitis/ai/pointpillars_nuscenes.hpp>

using namespace vitis::ai::pointpillars_nus;
namespace vitis {
namespace ai {
namespace pointpainting{

void read_inno_file_pointpainting(const std::string &file_name, PointsInfo &points_info, int points_dim, 
                                  std::vector<SweepInfo> &sweeps, int sweeps_points_dim, 
                                  std::vector<cv::Mat> &images);
}}}
