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
#include <vector>
#include <utility>
#include <memory>

using namespace std;

namespace vitis { namespace ai {
namespace pointpillars_nus{

//constexpr uint32_t MAX_POINTS_NUM = 64;
//constexpr uint32_t MAX_VOXELS_NUM = 40000; 

class Voxelization {
  public:
    static std::unique_ptr<Voxelization> create(const std::vector<float> &input_means, 
            const std::vector<float> &input_scales, int max_points_num, int max_voxels_num); 

    explicit Voxelization(const std::vector<float> &input_means, 
                          const std::vector<float> &input_scales,
                          int max_points_num, int max_voxels_num); 
    std::vector<int> voxelize(const vector<float> &points, int dim, int8_t *input_tensor_ptr,
                              size_t input_tensor_size);  
  
  private:
    std::vector<int> voxelize_input_internal(const vector<float> &points, int dim,
                                             int8_t * input_tensor_ptr, size_t input_tensor_size);
    int voxelize_input(const std::vector<float> &points, 
                       int dim, std::vector<int> &coors, 
                       int8_t *input_tensor_ptr);

  private:
    std::vector<float> input_means_;
    std::vector<float> input_scales_;
    int max_points_num_;
    int max_voxels_num_;
    std::vector<float> voxels_size_;
    std::vector<float> coors_range_;
    int coors_dim_; // 4 = 3 + padding
};


//std::vector<int> preprocess(const vector<float> &points, int dim, 
//                            const std::vector<float> &input_mean, const std::vector<float> &input_scale, 
//                            int8_t *input_tensor_ptr);
} // end of pointpillars_nus
}}


