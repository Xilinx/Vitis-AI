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

#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./scatter.hpp"

DEF_ENV_PARAM(DEBUG_POINTPILLARS_NUS, "0");

DEF_ENV_PARAM(DEBUG_SCATTER, "0");

namespace vitis { namespace ai {
namespace pointpillars_nus{

// input: 40000*1*64
// output: 400*400*64
//
void scatter(const std::vector<int> &coors, int coors_dim, const int8_t *input_data, float input_scale, 
             int8_t *output_data, float output_scale, uint32_t in_channels, int nx, int ny) {
  //auto size = w * h * c;
  //auto coors_shape = coors.shape; // [40000, 4] or [num, 4]   
  if (ENV_PARAM(DEBUG_SCATTER)) {
    LOG(INFO) << "coors size:" << coors.size();
    LOG(INFO) << "coors dim:" << coors_dim;
    LOG(INFO) << "input cnannels:" << in_channels; 
    LOG(INFO) << "input scale:" << input_scale; 
    LOG(INFO) << "output scale:" << output_scale; 
  } 
  auto coors_num = coors.size() / coors_dim;
  //auto in_channels = 64; // read from config
  //auto nx = 400; // read from config
  //auto ny = 400; // read from config

  bool copy_directly = (std::abs(input_scale * output_scale -1) < 0.0001);  

  //for (auto i = 0u; i < coors.shape[0]; ++i) {
  for (auto i = 0u; i < coors_num; ++i) {
    //auto index = coors.at({(int)i, 2}) * nx + coors.at({(int)i, 3});
    auto index = coors[i * coors_dim + 2] * nx + coors[i * coors_dim + 3];
    auto ibegin = input_data + i * in_channels;
    auto iend = ibegin + in_channels;
    auto obegin = output_data + index * in_channels;
    if (copy_directly) {
      std::memcpy(obegin, ibegin, in_channels);
    } else {
      std::transform(ibegin, iend, obegin, [=](int8_t in)->int8_t {return (int)(in * input_scale * output_scale);});
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_SCATTER)) 
          << "i: " << i << " coor: " << coors[i * coors_dim + 2] <<  ", "<< coors[i * coors_dim + 3]
          << " scatter index:" << index
          << ", input data:" << (int)*(ibegin) 
          << ", output data:" << (int)*(obegin); 
    if (ENV_PARAM(DEBUG_SCATTER)) {
      std::cout << "input data:";
      for (auto j = 0u; j < in_channels; ++j) {
        std::cout << (int)*(ibegin +j) << " ";
      }
      std::cout << std::endl;
    }

  }
  // maybe bug
  //uint32_t max_coors_num = 40000;
  //if (coors_num < max_coors_num) {
  //  auto last = max_coors_num -1;
  //  auto index = 0;
  //  auto ibegin = input_data + last * in_channels;
  //  auto iend = ibegin + in_channels;
  //  auto obegin = output_data + index * in_channels;
  //  if (copy_directly) {
  //    std::memcpy(obegin, ibegin, in_channels);
  //  } else {
  //    std::transform(ibegin, iend, obegin, [=](int8_t in)->int8_t {return (int)(in * input_scale * output_scale);});
  //  }
  //} 
}

} // end of pointpillars_nuscenes
}}


