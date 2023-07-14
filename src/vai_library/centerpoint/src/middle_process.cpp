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
#include "./middle_process.hpp"
#include <iostream>

DEF_ENV_PARAM(DEBUG_CENTERPOINT, "0");

DEF_ENV_PARAM(DEBUG_SCATTER, "0");

namespace vitis { namespace ai {
namespace centerpoint{

// input: 40000*1*64
// output: 400*400*64
//

void middle_process(const std::vector<int> &coors, const int8_t *input_data, float input_scale, 
                    int8_t *output_data, float output_scale, uint32_t in_channels) {
  //auto size = w * h * c;
  //auto coors_shape = coors.shape; // [40000, 4] or [num, 4]   
  if (ENV_PARAM(DEBUG_SCATTER)) {
    LOG(INFO) << "coors size:" << coors.size();
    LOG(INFO) << "input cnannels:" << in_channels; 
    LOG(INFO) << "input scale:" << input_scale; 
    LOG(INFO) << "output scale:" << output_scale; 
  } 
  auto coors_num = coors.size() / 4;
  //auto in_channels = 64; // read from config
  auto nx = 400; // read from config
  //auto ny = 400; // read from config

  bool copy_directly = (std::abs(input_scale * output_scale -1) < 0.0001);  

  //for (auto i = 0u; i < coors.shape[0]; ++i) {
  for (auto i = 0u; i < coors_num; ++i) {
    //auto index = coors.at({(int)i, 2}) * nx + coors.at({(int)i, 3});
    auto index = coors[i * 4 + 2] * nx + coors[i * 4 + 3];
    auto ibegin = input_data + i * in_channels;
    auto iend = ibegin + in_channels;
    auto obegin = output_data + index * in_channels;
    if (copy_directly) {
      std::memcpy(obegin, ibegin, in_channels);
    } else {
      std::transform(ibegin, iend, obegin, [=](int8_t in)->int8_t {return (int)(in * input_scale * output_scale);});
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_SCATTER)) 
          << "i: " << i << " coor: " << coors[i * 4 + 2] <<  ", "<< coors[i * 4 + 3]
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
 
}

void middle_process(const DataContainer<int> &coors, const int8_t *input_data, 
                    int w, int h, int c, float input_scale, 
                    int8_t *output_data, float output_scale) {
  //auto size = w * h * c;
  auto coors_shape = coors.shape; // [40000, 4] or [num, 4]   
  auto in_channels = 64; // read from config
  auto nx = 400; // read from config
  //auto ny = 400; // read from config

  bool copy_directly = (std::abs(input_scale * output_scale -1) < 0.0001);  

  for (auto i = 0u; i < coors.shape[0]; ++i) {
    auto index = coors.at({(int)i, 2}) * nx + coors.at({(int)i, 3});
    auto ibegin = input_data + i * in_channels;
    auto iend = ibegin + in_channels;
    auto obegin = output_data + index * in_channels;
    if (copy_directly) {
      std::memcpy(obegin, ibegin, in_channels);
    } else {
      std::transform(ibegin, iend, obegin, [=](int8_t in)->int8_t {return (int)(in * input_scale * output_scale);});
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_SCATTER)) 
          << "i: " << i << " coor: " << coors.at({int(i), 2}) <<  ", "<< coors.at({int(i), 3})
          << " scatter index:" << index
          << ", input data:" << (int)*(ibegin) 
          << ", output data:" << (int)*(obegin); 
    if (ENV_PARAM(DEBUG_SCATTER)) {
      std::cout << "input data:";
      for (auto j = 0; j < in_channels; ++j) {
        std::cout << (int)*(ibegin +j) << " ";
      }
      //std::cout << std::endl;
    }

  }
 
}

void middle_process(const DataContainer<int> &coors, const float *input_data, 
                    int w, int h, int c, float input_scale, 
                    float *output_data, float output_scale) {
  //auto size = w * h * c;
  auto coors_shape = coors.shape; // [40000, 4] or [num, 4]   
  auto in_channels = 64; // read from config
  auto nx = 400; // read from config
  //auto ny = 400; // read from config

  bool copy_directly = (std::abs(input_scale * output_scale -1) < 0.0001);  

  for (auto i = 0u; i < coors.shape[0]; ++i) {
    auto index = coors.at({(int)i, 2}) * nx + coors.at({(int)i, 3});
    auto ibegin = input_data + i * in_channels;
    auto iend = ibegin + in_channels;
    auto obegin = output_data + index * in_channels;
    if (copy_directly) {
      std::memcpy(obegin, ibegin, in_channels);
    } else {
      std::transform(ibegin, iend, obegin, [=](int8_t in)->int8_t {return (int)(in * input_scale * output_scale);});
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_SCATTER)) 
          << "i: " << i << " coor: " << coors.at({int(i), 2}) <<  ", "<< coors.at({int(i), 3})
          << " scatter index:" << index
          << ", input data:" << *(ibegin) 
          << ", output data:" << *(obegin); 
    if (ENV_PARAM(DEBUG_SCATTER)) {
      std::cout << "input data:";
      for (auto j = 0; j < in_channels; ++j) {
        std::cout << *(ibegin +j) << " ";
      }
      std::cout << std::endl;
    }

  }
 
}

} // end of pointpillars_nuscenes
}}


