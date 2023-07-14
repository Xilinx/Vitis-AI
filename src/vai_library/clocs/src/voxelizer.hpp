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
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace vitis {
namespace ai {
namespace clocs {

// constexpr uint32_t MAX_POINTS_NUM = 64;
// constexpr uint32_t MAX_VOXELS_NUM = 40000;
// enum VoxelType {
//   POINTPILLARS = 0,
//   POINTPILLARS_NUS = 1,
//   CENTERPOINT_WAYMO = 2,
// };

struct VoxelConfig {
  // VoxelType type;
  std::vector<float> input_means;
  std::vector<float> input_scales;
  int max_points_num=0;              // 20
  int max_voxels_num=0;              // 60000
  int feature_dim=0;                 // 11
  std::vector<float> voxels_size;  // [0.24, 0.24, 6]
  std::vector<float> coors_range;  // [-74.88, -74.88, -2, 74.88, 74.88, 4]
  int coors_dim=0;                   // 4
  int in_channels=0;                 // 64
};

class Voxelizer {
 public:
  static std::unique_ptr<Voxelizer> create(const VoxelConfig& config);
  static std::unique_ptr<Voxelizer> create(
      const std::vector<float>& input_means,
      const std::vector<float>& input_scales, int max_points_num,
      int max_voxels_num);

  explicit Voxelizer(const VoxelConfig& config);
  explicit Voxelizer(const std::vector<float>& input_means,
                     const std::vector<float>& input_scales, int max_points_num,
                     int max_voxels_num);
  std::vector<int> voxelize(const vector<float>& points, int dim,
                            int8_t* input_tensor_ptr, size_t input_tensor_size);

 private:
  std::vector<int> voxelize_input_internal(const vector<float>& points, int dim,
                                           int8_t* input_tensor_ptr,
                                           size_t input_tensor_size);
  int voxelize_input(const std::vector<float>& points, int dim,
                     std::vector<int>& coors, int8_t* input_tensor_ptr);

  int voxelize_input_kitti(const std::vector<float>& points, int dim,
                           std::vector<int>& coors, int8_t* input_tensor_ptr);
  int points_to_voxel_reverse_kernel(const std::vector<float>& points, int dim,
                                     std::vector<int>& coors, int coors_dim,
                                     int8_t* input_tensor_ptr);

 private:
  VoxelConfig config_;
  std::vector<float> input_means_;
  std::vector<float> input_scales_;
  int max_points_num_=0;
  int max_voxels_num_=0;
  std::vector<float> voxels_size_;
  std::vector<float> coors_range_;
  int coors_dim_=0;  // 4 = 3 + padding

  std::vector<int> grid_size_;
  std::vector<int> num_points_per_pillar_;
  std::vector<int> coor_to_voxelidx_;
  // std::vector<int> coors_;
};

// std::vector<int> preprocess(const vector<float> &points, int dim,
//                            const std::vector<float> &input_mean, const
//                            std::vector<float> &input_scale, int8_t
//                            *input_tensor_ptr);
}  // namespace clocs
}  // namespace ai
}  // namespace vitis

