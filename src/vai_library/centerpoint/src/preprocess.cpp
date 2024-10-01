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
#include <memory>
#include <thread>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./preprocess.hpp"

DEF_ENV_PARAM(DEBUG_CENTERPOINT, "0");
DEF_ENV_PARAM(DEBUG_VOXELIZE, "0");
DEF_ENV_PARAM(DEBUG_PERMUTE, "0");
DEF_ENV_PARAM(DEBUG_NEON, "0");
DEF_ENV_PARAM(USE_OLD_VOXELIZE, "0");

namespace vitis { namespace ai { 
namespace centerpoint {

//constexpr uint32_t POINTS_DIM = 4;
template<typename T>
static void writefile_(string& filename, vector<T>& data) {
    ofstream output_file(filename);
      for(size_t i = 0; i < data.size(); i++)
            output_file << data[i] << endl;
}

static void dynamic_voxelize_kernel2(const vector<float> &points,
                             vector<int> &coors, // [n, 3]
                             const std::vector<float> voxel_size,
                             const std::vector<float> coors_range,
                             const std::vector<int> grid_size,
                             const int num_points, const int num_features,
                             const int NDim) {
  const int ndim_minus_1 = NDim - 1;
  bool failed = false;
  int coor[NDim];
  int c;

  for (int i = 0; i < num_points; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      // c = floor((points[i][j] - coors_range[j]) / voxel_size[j]);
       //LOG_IF(ERROR, ENV_PARAM(DEBUG_CENTERPOINT)) 
       //      << "i: " << i << ", j: " << j;
      c = std::floor((points[i * num_features + j] - coors_range[j]) / voxel_size[j]);
      // necessary to rm points out of range
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }

    for (int k = 0; k < NDim; ++k) {
      if (failed)
        //coors[i][k] = -1;
        coors[i * 3 + k] = -1;
      else
        //coors[i][k] = coor[k];
        coors[i * 3 + k] = coor[k];
    }
  }

  return;
}

static void hard_voxelize_kernel3(const vector<float> &points,
                          //DataContainer<float> &voxels,
                          //vector<float> &voxels, //(40000, 64, 4)
                          int8_t * voxels, 
                          std::vector<float> &means,
                          std::vector<float> &scales,
                          vector<int> &coors,// (n, 4)
                          int coors_dim,
                          vector<int> &num_points_per_voxel,
                          //vector<vector<vector<int>>> &coor_to_voxelidx,
                          vector<int> &coor_to_voxelidx, // 1, 400, 400
                          int& voxel_num, const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const std::vector<int> grid_size,
                          const int max_points, const int max_voxels,
                          const int num_points, const int num_features,
                          const int NDim) {
__TIC__(TEMP_COORS_INIT)
  //vector<vector<int>> temp_coors(num_points);
  //for (auto i = 0; i < num_points; ++i) {
  //  temp_coors[i].resize(3); 
  //  memset(temp_coors[i].data(), 0, 3);
  //}
  vector<int> temp_coors(num_points * 3, 0);
  LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT)) 
        << "temp_coors size:" << temp_coors.size();
__TOC__(TEMP_COORS_INIT)
  // First use dynamic voxelization to get coors,
  // then check max points/voxels constraints
  //dynamic_voxelize_kernel<T, int>(points, temp_coors.accessor<int, 2>(),
__TIC__(DYNAMIC_VOXELIZE_KERNEL2)
  dynamic_voxelize_kernel2(points, temp_coors,
                          voxel_size, coors_range, grid_size,
                          num_points, num_features, NDim);
  //auto o = std::ofstream("./temp_coors_2.txt");
  //for (auto i = 0; i < num_points; ++i) {
  //  o << temp_coors[i * 3] << " "
  //    << temp_coors[i * 3 + 1] << " "
  //    << temp_coors[i * 3 + 2] << std::endl; 
  //}
  //o.close();
__TOC__(DYNAMIC_VOXELIZE_KERNEL2)

  int voxelidx, num;
  //auto coor = temp_coors.accessor<int, 2>();
  // note : need copy?
__TIC__(SELECT_VOXELS)
    //vector<vector<int>> coor = temp_coors;
    vector<int> coor = temp_coors;
    for (int i = 0; i < num_points; ++i) {
      //if (coor[i][0] == -1) continue;
      if (coor[i * 3] == -1) continue;
      
      //voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]];
      auto idx = coor[i * 3] * 400 * 400 + coor[i * 3 + 1] * 400 + coor[i * 3 + 2];
      voxelidx = coor_to_voxelidx[idx];
      // record voxel
      if (voxelidx == -1) {
        voxelidx = voxel_num;
        if (max_voxels != -1 && voxel_num >= max_voxels) break;
        voxel_num += 1;

        //coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]] = voxelidx;
        //coor_to_voxelidx[coor[i * 3]][coor[i * 3 + 1]][coor[i * 3 + 2]] = voxelidx;
        coor_to_voxelidx[idx] = voxelidx;
        for (int k = 0; k < NDim; ++k) {
          //coors[voxelidx][k + 1] = coor[i * 3 + k];
          coors[voxelidx * coors_dim + k + 1] = coor[i * 3 + k];
        }
      }
      // put points into voxel
      num = num_points_per_voxel[voxelidx];
      if (max_points == -1 || num < max_points) {
        auto final_idx = voxelidx * 40 * num_features + num * num_features; // 64 need to read from tensor
        //if (ENV_PARAM(DEBUG_NEON)) {
        //  set_input_neon_channel(points.data() + i * 4, 4, voxels + final_idx, scales);
        //} else {
          for (int k = 0; k < num_features; ++k) {
            //voxels[voxelidx][num][k] = points[i][k];
            //voxels.at({voxelidx,num,k}) = points[i][k];
            //voxels.at({voxelidx,num,k}) = points[i * 4 + k];
            //voxels[voxelidx * 64 *4 + num * 4 + k] = points[i * 4 + k];
            //voxels[final_idx + k] = (int)(points[i * 4 + k] * scales[k]);
            //voxels[final_idx + k] = (int)((points[i * num_features + k] - means[k]) * scales[k]);
            voxels[final_idx + k] = std::round((points[i * num_features + k] - means[k]) * scales[k]);
          }
        //}
        num_points_per_voxel[voxelidx] += 1;
      }
    }

__TOC__(SELECT_VOXELS)
  return;
}


static int hard_voxelize_cpu3(const vector<float>& points, 
                      //vector<float>& voxels, // (40000, 64, 4)
                      int points_dim,
                      int8_t * voxels, 
                      std::vector<float> &means,
                      std::vector<float> &scales,
                      vector<int>& coors, // (n, 4)
                      int coors_dim,
                      vector<int>& num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3) { // coors_range dim

  std::vector<int> grid_size(NDim);
  const int num_points = points.size() / points_dim;
  LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT)) << "num_points:" << num_points;
  //const int num_features = points.size(1);
  const int num_features = points_dim; // points dim 
  LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT)) << "num_features:" << num_features;

  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT)) 
          << "grid_size[" << i << "]:" << grid_size[i];
  }

  // coors, num_points_per_voxel, coor_to_voxelidx are int Tensor
  // printf("cpu coor_to_voxelidx size: [%d, %d, %d]\n", grid_size[2],
  // grid_size[1], grid_size[0]);
  
  // at::Tensor coor_to_voxelidx =
  //     -at::ones({grid_size[2], grid_size[1], grid_size[0]}, coors.options());
  vector<int> coor_to_voxelidx(grid_size[2] * grid_size[1] * grid_size[0], -1);

  LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT)) 
        << " coor_to_voxelidx size:" << coor_to_voxelidx.size();
  int voxel_num = 0;
__TIC__(HARD_VOXELIZE_KERNEL2)
  hard_voxelize_kernel3(
        points, voxels, means, scales, coors, coors_dim, num_points_per_voxel,
        coor_to_voxelidx, voxel_num, voxel_size,
        coors_range, grid_size, max_points, max_voxels, num_points,
        num_features, NDim);
  //LOG(ERROR) << "HELLO";
__TOC__(HARD_VOXELIZE_KERNEL2)

  return voxel_num;
}


static int voxelize_input(const std::vector<float> &points, 
                          int dim,
                          std::vector<int> &coors, 
                          int8_t *input_tensor_ptr, 
                          std::vector<float> means,
                          std::vector<float> scales) {
  //const int dim = 4; 
  const int coors_dim = 4; // 3 and padding
  coors.resize(2560 * coors_dim); 
  std::vector<float> voxels_size{0.2, 0.2, 14};
  //std::vector<float> coors_range{-50, -50, -5, 50, 50, 3};
  std::vector<float> coors_range{0, -40, -6, 80, 40, 8};
__TIC__(VOXELIZE_RESULT_INIT)
  vector<int> num_points(MAX_VOXELS_NUM, 0);
__TOC__(VOXELIZE_RESULT_INIT)
 
__TIC__(HARD_VOXELIZE_CPU)
  int voxel_num = 0;
  voxel_num = hard_voxelize_cpu3(points, dim, input_tensor_ptr, means, scales, coors, coors_dim, num_points,
                    voxels_size, coors_range, MAX_POINTS_NUM, MAX_VOXELS_NUM); 
  //for (auto i = 0u; i < coors.size(); ++i) {
  //  memcpy(result.coors.data.data() + i * coors_dim, coors[i].data(), coors_dim * sizeof(int));
  //}
__TOC__(HARD_VOXELIZE_CPU)
  LOG_IF(INFO, ENV_PARAM(DEBUG_CENTERPOINT)) 
        << " voxel num:" << voxel_num;
  coors.resize(voxel_num * 4); 
  return voxel_num; 
}


std::vector<int> preprocess3(const std::vector<float> &points,  int dim, 
                             const std::vector<float> &input_mean, 
                             const std::vector<float> &input_scale, 
                             int8_t *input_tensor_ptr) {
__TIC__(VOXELIZE_INPUT)
  // 1. voxelize
  std::vector<int> coors; 
  //auto input_scale = std::vector<float>{1.0/50, 1.0/50, 1.0/5, 1.0/0.5}; // read from config
  //for (auto i = 0u; i < input_scale.size(); ++i) {
  //  input_scale[i] *= input_tensor_scale;
  //}

  voxelize_input(points, dim, coors, input_tensor_ptr, input_mean, input_scale); // tuple: voxels, num_points, coors
__TOC__(VOXELIZE_INPUT)
  return std::move(coors);
}


}
}}

