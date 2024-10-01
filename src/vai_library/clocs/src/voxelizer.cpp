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
//#include <thread>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./voxelizer.hpp"
//#include "./setinput_neon.hpp"

DEF_ENV_PARAM(DEBUG_VOXELIZE, "0");
DEF_ENV_PARAM(DEBUG_VOXELIZE_DUMP, "0");
// DEF_ENV_PARAM(DEBUG_NEON, "0");

namespace vitis {
namespace ai {
namespace clocs {

static void dynamic_voxelize_kernel(const vector<float>& points,
                                    vector<int>& coors,  // [n, 3]
                                    const std::vector<float> voxel_size,
                                    const std::vector<float> coors_range,
                                    const std::vector<int> grid_size,
                                    const int num_points,
                                    const int num_features, const int NDim) {
  const int ndim_minus_1 = NDim - 1;
  bool failed = false;
  int coor[NDim];
  int c;

  for (int i = 0; i < num_points; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      // c = floor((points[i][j] - coors_range[j]) / voxel_size[j]);
      // LOG_IF(ERROR, ENV_PARAM(DEBUG_VOXELIZE))
      //      << "i: " << i << ", j: " << j;
      c = std::floor((points[i * num_features + j] - coors_range[j]) /
                     voxel_size[j]);
      // necessary to rm points out of range
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }

    for (int k = 0; k < NDim; ++k) {
      if (failed)
        // coors[i][k] = -1;
        coors[i * 3 + k] = -1;
      else
        // coors[i][k] = coor[k];
        coors[i * 3 + k] = coor[k];
    }
  }

  return;
}

static void hard_voxelize_kernel3(
    const vector<float>& points,
    // DataContainer<float> &voxels,
    // vector<float> &voxels, //(40000, 64, 4)
    int8_t* voxels, std::vector<float>& means, std::vector<float>& scales,
    vector<int>& coors,  // (n, 4)
    int coors_dim, vector<int>& num_points_per_voxel,
    // vector<vector<vector<int>>> &coor_to_voxelidx,
    vector<int>& coor_to_voxelidx,  // 1, 400, 400
    int& voxel_num, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const std::vector<int> grid_size,
    const int max_points, const int max_voxels, const int num_points,
    const int num_features, const int NDim) {
  // declare a temp coors
  // at::Tensor temp_coors = at::zeros(
  //    {num_points, NDim},
  //    at::TensorOptions().dtype(at::kInt).device(at::kCPU));

  // DataContainer<int> temp_coors{std::vector<uint32_t>({(uint32_t)num_points,
  // 3}), 0}; vector<vector<int>> temp_coors(num_points); for (auto i = 0; i <
  // num_points; ++i) {
  //  temp_coors[i].resize(3);
  //  memset(temp_coors[i].data(), 0, 3);
  //}
  vector<int> temp_coors(num_points * 3, 0);
  LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE))
      << "temp_coors size:" << temp_coors.size();
  // First use dynamic voxelization to get coors,
  // then check max points/voxels constraints
  // dynamic_voxelize_kernel<T, int>(points, temp_coors.accessor<int, 2>(),
  dynamic_voxelize_kernel(points, temp_coors, voxel_size, coors_range,
                          grid_size, num_points, num_features, NDim);
  // auto o = std::ofstream("./temp_coors_2.txt");
  // for (auto i = 0; i < num_points; ++i) {
  //  o << temp_coors[i * 3] << " "
  //    << temp_coors[i * 3 + 1] << " "
  //    << temp_coors[i * 3 + 2] << std::endl;
  //}
  // o.close();

  int voxelidx, num;
  // auto coor = temp_coors.accessor<int, 2>();
  // note : need copy?
  // vector<vector<int>> coor = temp_coors;
  vector<int> coor = temp_coors;
  for (int i = 0; i < num_points; ++i) {
    // if (coor[i][0] == -1) continue;
    if (coor[i * 3] == -1) continue;

    // voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]];
    auto idx =
        coor[i * 3] * 400 * 400 + coor[i * 3 + 1] * 400 + coor[i * 3 + 2];
    voxelidx = coor_to_voxelidx[idx];
    // record voxel
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (max_voxels != -1 && voxel_num >= max_voxels) break;
      voxel_num += 1;

      // coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]] = voxelidx;
      // coor_to_voxelidx[coor[i * 3]][coor[i * 3 + 1]][coor[i * 3 + 2]] =
      // voxelidx;
      coor_to_voxelidx[idx] = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        // coors[voxelidx][k + 1] = coor[i * 3 + k];
        coors[voxelidx * coors_dim + k + 1] = coor[i * 3 + k];
      }
    }
    // put points into voxel
    num = num_points_per_voxel[voxelidx];
    if (max_points == -1 || num < max_points) {
      auto final_idx = voxelidx * 64 * num_features +
                       num * num_features;  // 64 need to read from tensor
                                            // if (ENV_PARAM(DEBUG_NEON)) {
      //  set_input_neon_channel(points.data() + i * 4, 4, voxels + final_idx,
      //  scales);
      //} else {
      for (int k = 0; k < num_features; ++k) {
        // voxels[voxelidx][num][k] = points[i][k];
        // voxels.at({voxelidx,num,k}) = points[i][k];
        // voxels.at({voxelidx,num,k}) = points[i * 4 + k];
        // voxels[voxelidx * 64 *4 + num * 4 + k] = points[i * 4 + k];
        // voxels[final_idx + k] = (int)(points[i * 4 + k] * scales[k]);
        // voxels[final_idx + k] = (int)((points[i * num_features + k] -
        // means[k]) * scales[k]);
        voxels[final_idx + k] =
            std::round((points[i * num_features + k] - means[k]) * scales[k]);
      }
      //}
      num_points_per_voxel[voxelidx] += 1;
    }
  }

  return;
}

static int hard_voxelize_cpu(const vector<float>& points,
                             // vector<float>& voxels, // (40000, 64, 4)
                             int points_dim, int8_t* voxels,
                             std::vector<float>& means,
                             std::vector<float>& scales,
                             vector<int>& coors,  // (n, 4)
                             int coors_dim, vector<int>& num_points_per_voxel,
                             const std::vector<float> voxel_size,
                             const std::vector<float> coors_range,
                             const int max_points, const int max_voxels,
                             const int NDim = 3) {  // coors_range dim

  std::vector<int> grid_size(NDim);
  const int num_points = points.size() / points_dim;
  LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE)) << "num_points:" << num_points;
  // const int num_features = points.size(1);
  const int num_features = points_dim;  // points dim
  LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE)) << "num_features:" << num_features;

  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE))
        << "grid_size[" << i << "]:" << grid_size[i];
  }

  // coors, num_points_per_voxel, coor_to_voxelidx are int Tensor
  // printf("cpu coor_to_voxelidx size: [%d, %d, %d]\n", grid_size[2],
  // grid_size[1], grid_size[0]);

  // at::Tensor coor_to_voxelidx =
  //     -at::ones({grid_size[2], grid_size[1], grid_size[0]}, coors.options());
  vector<int> coor_to_voxelidx(grid_size[2] * grid_size[1] * grid_size[0], -1);

  LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE))
      << " coor_to_voxelidx size:" << coor_to_voxelidx.size();
  int voxel_num = 0;
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
  //    points.scalar_type(), "hard_voxelize_forward", [&] {
  //      hard_voxelize_kernel<scalar_t, int>(
  //          points.accessor<scalar_t, 2>(), voxels.accessor<scalar_t, 3>(),
  //          coors.accessor<int, 2>(), num_points_per_voxel.accessor<int, 1>(),
  //          coor_to_voxelidx.accessor<int, 3>(), voxel_num, voxel_size,
  //          coors_range, grid_size, max_points, max_voxels, num_points,
  //          num_features, NDim);
  //    });
  hard_voxelize_kernel3(points, voxels, means, scales, coors, coors_dim,
                        num_points_per_voxel, coor_to_voxelidx, voxel_num,
                        voxel_size, coors_range, grid_size, max_points,
                        max_voxels, num_points, num_features, NDim);

  return voxel_num;
}

std::unique_ptr<Voxelizer> Voxelizer::create(const VoxelConfig& config) {
  return std::unique_ptr<Voxelizer>(new Voxelizer(config));
}

std::unique_ptr<Voxelizer> Voxelizer::create(
    const std::vector<float>& input_means,
    const std::vector<float>& input_scales, int max_points_num,
    int max_voxels_num) {
  return std::unique_ptr<Voxelizer>(
      new Voxelizer(input_means, input_scales, max_points_num, max_voxels_num));
}

void print_config(const VoxelConfig& config) {
  std::cout << "print voxel config:" << std::endl;
  std::cout << "input means:";
  for (auto i = 0u; i < config.input_means.size(); ++i) {
    std::cout << config.input_means[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "input scales:";
  for (auto i = 0u; i < config.input_scales.size(); ++i) {
    std::cout << config.input_scales[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "max points num:" << config.max_points_num << std::endl;
  std::cout << "max voxels num:" << config.max_voxels_num << std::endl;
  std::cout << "feature dim:" << config.feature_dim << std::endl;
  std::cout << "coors dim:" << config.coors_dim << std::endl;
  std::cout << "in channels:" << config.in_channels << std::endl;

  std::cout << "voxels size:";
  for (auto i = 0u; i < config.voxels_size.size(); ++i) {
    std::cout << config.voxels_size[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "coors range:";
  for (auto i = 0u; i < config.coors_range.size(); ++i) {
    std::cout << config.coors_range[i] << " ";
  }
  std::cout << std::endl;
}

Voxelizer::Voxelizer(const VoxelConfig& config) : config_(config) {
  if (ENV_PARAM(DEBUG_VOXELIZE)) {
    print_config(config);
  }
  num_points_per_pillar_.resize(config_.max_voxels_num);
  grid_size_.resize(3);
  for (int i = 0; i < 3; ++i) {
    grid_size_[i] =
        std::round((config_.coors_range[i + 3] - config_.coors_range[i]) /
                   config_.voxels_size[i]);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE))
      << "grid_size:" << grid_size_[0] << " " << grid_size_[1] << " "
      << grid_size_[2];
  coor_to_voxelidx_.resize(grid_size_[0] * grid_size_[1] * grid_size_[2]);

  // coors_.resize(4 * config_.max_pillar_num);
}

Voxelizer::Voxelizer(const std::vector<float>& input_means,
                     const std::vector<float>& input_scales, int max_points_num,
                     int max_voxels_num)
    : input_means_{input_means},
      input_scales_{input_scales},
      max_points_num_(max_points_num),
      max_voxels_num_(max_voxels_num),
      voxels_size_{0.25, 0.25, 8},
      coors_range_{-50, -50, -5, 50, 50, 3},
      coors_dim_(4) {}

std::vector<int> Voxelizer::voxelize(const vector<float>& points, int dim,
                                     int8_t* input_tensor_ptr,
                                     size_t input_tensor_size) {
  return voxelize_input_internal(points, dim, input_tensor_ptr,
                                 input_tensor_size);
}

int Voxelizer::points_to_voxel_reverse_kernel(const std::vector<float>& points,
                                              int dim, std::vector<int>& coors,
                                              int coors_dim,
                                              int8_t* input_tensor_ptr) {
  int N = points.size() / dim;
  // ndim = dim -1
  int ndim = 3;
  int ndim_minus_1 = ndim - 1;

  std::vector<float> feature_debug;
  if (ENV_PARAM(DEBUG_VOXELIZE_DUMP)) {
    feature_debug.resize(config_.max_voxels_num * config_.max_points_num * dim);
  }
  int voxel_num = 0;
  vector<int> coor(3, 0);
  bool failed = false;
  for (auto i = 0; i < N; ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE))
        << "i:" << i << ", point:" << points[i * dim] << ","
        << points[i * dim + 1] << ", " << points[i * dim + 2] << ", ";
    failed = false;
    for (auto j = 0; j < ndim; ++j) {
      int c = std::floor((points[i * dim + j] - config_.coors_range[j]) /
                         config_.voxels_size[j]);
      if (c < 0 || c >= grid_size_[j]) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed) continue;
    // grid size:x, y, z
    // coor: z, y, x
    LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE))
        << "coor:" << coor[0] << ", " << coor[1] << ", " << coor[2];
    int coor_pos = coor[0] * grid_size_[1] * grid_size_[0] +
                   coor[1] * grid_size_[0] + coor[2];

    int voxelidx = coor_to_voxelidx_[coor_pos];
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= config_.max_voxels_num) {
        continue;
      }
      voxel_num++;
      coor_to_voxelidx_[coor_pos] = voxelidx;
      for (auto j = 0; j < ndim; ++j) {
        coors[voxelidx * coors_dim + j + 1] = coor[j];
      }
    }
    int points_num = num_points_per_pillar_[voxelidx];
    if (points_num < config_.max_points_num) {
      int pos = (voxelidx * config_.max_points_num + points_num) * dim;
      // for (int j = 0; j < dim; ++j) {
      //  input_tensor_ptr[pos + j] =
      //      (int)((points[i * dim + j] - config_.input_means[j]) *
      //            config_.input_scales[j]);
      //}
      vector<float> buf(dim);
      for (int j = 0; j < 3; ++j) {
        buf[j] = (points[i * dim + j] - config_.coors_range[j]) /
                 (config_.coors_range[j + 3] - config_.coors_range[j]);
      }
      buf[3] = points[i * dim + 3];
      for (int j = 0; j < dim; ++j) {
        int val = std::round(buf[j] * 128);
        if (val < -128) {
          val = -128;
        } else if (val > 127) {
          val = 127;
        }
        input_tensor_ptr[pos + j] = val;
      }

      if (ENV_PARAM(DEBUG_VOXELIZE_DUMP)) {
        for (int j = 0; j < 4; ++j) {
          feature_debug[pos + j] = buf[j];
        }
      }
      num_points_per_pillar_[voxelidx]++;
    }
  }
  coors.resize(voxel_num * coors_dim);
  if (ENV_PARAM(DEBUG_VOXELIZE_DUMP)) {
    num_points_per_pillar_.resize(voxel_num);
    std::ofstream("num_points_per_pillar.bin")
        .write((char*)num_points_per_pillar_.data(),
               sizeof(int) * num_points_per_pillar_.size());
    coors.resize(voxel_num * coors_dim);
    std::ofstream c("coors.txt");
    for (int v = 0; v < voxel_num; ++v) {
      for (int p = 0; p < 4; ++p) {
        c << coors[v * 4 + p] << " ";
      }
      c << std::endl;
    }
    c.close();
    std::ofstream("coors.bin")
        .write((char*)coors.data(), sizeof(int) * coors.size());
    feature_debug.resize(voxel_num * config_.max_points_num * dim);
    std::ofstream("feature.bin")
        .write((char*)feature_debug.data(),
               sizeof(float) * feature_debug.size());
    std::ofstream f("feature.txt");
    int line = config_.max_points_num * dim;
    for (int v = 0; v < voxel_num; ++v) {
      for (int p = 0; p < line; ++p) {
        f << feature_debug[v * line + p] << " ";
      }
      f << std::endl;
    }
    f.close();

    for (auto i = 0u; i < feature_debug.size(); ++i) {
      feature_debug[i] = input_tensor_ptr[i] / 128.0;
    }

    std::ofstream("feature_input.bin")
        .write((char*)feature_debug.data(),
               sizeof(float) * feature_debug.size());
    std::ofstream f2("feature_input.txt");
    for (int v = 0; v < voxel_num; ++v) {
      for (int p = 0; p < line; ++p) {
        f2 << feature_debug[v * line + p] << " ";
      }
      f2 << std::endl;
    }
    f2.close();
  }
  return voxel_num;
}

int Voxelizer::voxelize_input_kitti(const std::vector<float>& points, int dim,
                                    std::vector<int>& coors,
                                    int8_t* input_tensor_ptr) {
  int coors_dim = config_.coors_dim;
  coors.resize(config_.max_voxels_num * coors_dim);
  coors.assign(coors.size(), 0);
  num_points_per_pillar_.assign(num_points_per_pillar_.size(), 0);
  coor_to_voxelidx_.assign(coor_to_voxelidx_.size(), -1);
  return points_to_voxel_reverse_kernel(points, dim, coors, coors_dim,
                                        input_tensor_ptr);
}

int Voxelizer::voxelize_input(const std::vector<float>& points, int dim,
                              std::vector<int>& coors,
                              int8_t* input_tensor_ptr) {
  // const int dim = 4;
  // const int coors_dim = 4; // 3 and padding
  coors.resize(max_voxels_num_ * coors_dim_);
  // std::vector<float> voxels_size{0.25, 0.25, 8};
  // std::vector<float> coors_range{-50, -50, -5, 50, 50, 3};
  vector<int> num_points(max_voxels_num_, 0);

  int voxel_num = 0;
  voxel_num = hard_voxelize_cpu(points, dim, input_tensor_ptr, input_means_,
                                input_scales_, coors, coors_dim_, num_points,
                                voxels_size_, coors_range_, max_points_num_,
                                max_voxels_num_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE)) << " voxel num:" << voxel_num;
  coors.resize(voxel_num * 4);
  return voxel_num;
}

std::vector<int> Voxelizer::voxelize_input_internal(
    const std::vector<float>& points, int dim, int8_t* input_tensor_ptr,
    size_t input_tensor_size) {
  // 1. voxelize
  std::vector<int> coors;

  // voxelize_input(points, dim, coors,
  //               input_tensor_ptr);  // tuple: voxels, num_points, coors
  int pillar_num = voxelize_input_kitti(
      points, dim, coors,
      input_tensor_ptr);  // tuple: voxels, num_points, coors

  LOG_IF(INFO, ENV_PARAM(DEBUG_VOXELIZE)) << "pilllar num:" << pillar_num;

  return std::move(coors);
}

}  // namespace clocs
}  // namespace ai
}  // namespace vitis

