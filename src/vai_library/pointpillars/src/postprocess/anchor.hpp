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

#include <iostream>
#include "./helper.hpp"

namespace vitis { namespace ai {

class anchor_stride{
public:
  static int create_all_anchors();

  anchor_stride(const std::vector<float>& sizes,
                const std::vector<float>& strides,
                const std::vector<float>& offsets,
                const std::vector<float>& rotations,
                float matched_threshold,
                float unmatched_threshold,
                const std::vector<float>& point_cloud_range,
                const std::vector<float>& voxel_size,
                const std::vector<int>&grid_size,
                int out_size_factor);

  anchor_stride(const anchor_stride& ) = delete;
  ~anchor_stride();
  
  void create_all_anchors_sub( V5F& anchors_v5);

  void generate_anchors();
public:  
  V1F sizes_;
  V1F strides_;
  V1F offsets_;
  V1F rotations_;
  float matched_threshold_;
  float unmatched_threshold_;
  V1F point_cloud_range_;
  V1F voxel_size_;
  std::vector<int> grid_size_;
  int out_size_factor_;
  V6F anchors_; 
};

}}

