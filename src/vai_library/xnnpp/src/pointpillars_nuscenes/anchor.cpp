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

#include <iostream>
#include <iterator>
#include <algorithm>
#include <string>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./anchor.hpp"

using namespace std;

DEF_ENV_PARAM(DEBUG_POINTPILLARS_NUS, "0")
DEF_ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS_ANCHOR, "0")

namespace vitis {
namespace ai {
namespace pointpillars_nus {

static void linspace(float x1, float x2, int n, std::vector<float>& ret, bool align_corner = false) {
  assert(n >= 1);
  ret.resize(n); 
  float d = (x2 - x1) / (n - 1);
  for (auto i = 0u; i < ret.size(); ++i) {
    ret[i] = x1 + i * d; 
    if (!align_corner) {
      ret[i] += d / 2;
    }
  }
}

// return size: [200*200*1*2, 9]
static void anchor_single_range(const AnchorInfo &params, int range_index, Anchors& all_anchors) {
  auto &feat_size = params.featmap_size; 
  auto &anchor_range = params.ranges[range_index];
  assert(anchor_range.size() >= 6);

  auto X = feat_size[1];  
  auto Y = feat_size[0];  
  auto Z = 1u;
  auto R = params.rotations.size();
  // X * Y * Z * rotation_size
  //auto anchor_num = X * Y * Z * R; 

  // feature map [Y, X] => [1, Y, X]
  // centers length z, x, y = [2, Y+1, X+1]
  // z centers
  std::vector<float> z_centers(Z + 1);
  linspace(anchor_range[2], anchor_range[5], z_centers.size(), z_centers, params.align_corner);
  if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS_ANCHOR)) {
    std::cout << "range index: " << range_index
              << " , z_centers size:" << z_centers.size()
              << std::endl;
    std::cout << "z_centers: ";
    for (auto i = 0u; i < z_centers.size(); ++i) {
      std::cout << z_centers[i] << " ";
    }
    std::cout << std::endl;
  }
  // y centers  
  std::vector<float> y_centers(Y + 1);
  linspace(anchor_range[1], anchor_range[4], y_centers.size(), y_centers, params.align_corner);
  if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS_ANCHOR)) {
    std::cout << "range index: " << range_index
              << " , y_centers size:" << y_centers.size()
              << std::endl;
    std::cout << "y_centers: ";
    for (auto i = 0u; i < y_centers.size(); ++i) {
      std::cout << y_centers[i] << " ";
    }
    std::cout << std::endl;
  }

  // x centers  
  std::vector<float> x_centers(X + 1);
  linspace(anchor_range[0], anchor_range[3], x_centers.size(), x_centers, params.align_corner);
  if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS_ANCHOR)) {
    std::cout << "range index: " << range_index
              << " , x_centers size:" << x_centers.size()
              << std::endl;
    std::cout << "x_centers: ";
    for (auto i = 0u; i < x_centers.size(); ++i) {
      std::cout << x_centers[i] << " ";
    }
    std::cout << std::endl;
  }

 
  std::vector<float> sizes(params.sizes[range_index].size()); 
  for (auto i = 0u; i < sizes.size(); ++i) {
    sizes[i] = params.sizes[range_index][i] * params.scale;
  }

  // anchors shape: [X, Y, Z, rotation_size, (x, y, z, r)] 
  //             => [X, Y, Z, 1, rotation_size, (x, y, z, r)] 
  //             => [X, Y, Z, 1, rotation_size, (x, y, z, (sizes), r)]
  //             => [X, Y, Z, 1, rotation_size, (x, y, z, (sizes), r, (custom_value))]
  //             => [Z, X, Y, 1, rotation_size, (x, y, z, (sizes), r, (custom_value))]
  auto custom_value_size = params.custom_values.size();
  auto last_dim = 4 + 3 + custom_value_size;

  for (auto ix = 0u; ix < X; ++ix) {
    for (auto iy = 0u; iy < Y; ++iy) {
      for (auto iz = 0u; iz < Z; ++iz) {
        for (auto ir = 0u; ir < R; ++ir) {
          // origin index: ix * Y * Z * R + iy * Z * R + iz * R + ir * 1;
          // final index: iz * Y * X * R + iy * X * R + ix * R + ir * 1;
          auto idx =  iz * Y * X * R + iy * X * R + ix * R + ir * 1;
          auto mapped_idx = (idx - ir )* params.ranges.size() + range_index * 2 + ir; 
          all_anchors[mapped_idx][0] = x_centers[ix];  
          all_anchors[mapped_idx][1] = y_centers[iy];
          all_anchors[mapped_idx][2] = z_centers[iz];
          std::copy_n(sizes.begin(), 3, all_anchors[mapped_idx].begin() + 3); 
          all_anchors[mapped_idx][6] = params.rotations[ir];
          for(auto i = 7u; i < last_dim; ++i) {
             all_anchors[mapped_idx][i] = params.custom_values[i - 7];
          }

          if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS_ANCHOR)) {
            std::cout << "idx: " << idx 
                      << " mapped_idx: " << mapped_idx 
                      << " ix: " << ix
                      << " iy: " << iy
                      << " iz: " << iz
                      << " ir: " << ir;
            for (auto i = 0u; i < all_anchors[mapped_idx].size(); ++i) {
              std::cout << " " << all_anchors[mapped_idx][i];
            }
            std::cout << std::endl;
          }
        }
      }
    }
  }  
}

Anchors generate_anchors(const AnchorInfo &params) {
__TIC__(GENERATE_ANCHORS)
  auto anchors_dim = 7 + params.custom_values.size();
  auto X = params.featmap_size[1];
  auto Y = params.featmap_size[0];
  auto Z = 1u;
  auto R = params.rotations.size();
  auto anchors_num = X * Y * Z * R * params.ranges.size();
__TIC__(GENERATE_ANCHORS_INIT)
  Anchors all_anchors(anchors_num, std::vector<float>(anchors_dim));
__TOC__(GENERATE_ANCHORS_INIT)
  for (auto i = 0u; i < params.ranges.size(); ++i) {
__TIC__(GENERATE_ANCHORS_SINGLE_RANGE)
    anchor_single_range(params, i, all_anchors);
__TOC__(GENERATE_ANCHORS_SINGLE_RANGE)
  }

  if (ENV_PARAM(DEBUG_XNNPP_POINTPILLARS_NUS_ANCHOR)) {
    std::cout << "anchors_dim: " << anchors_dim << std::endl;
    std::cout << "anchors: " << std::endl;
    for (auto i = 0u; i < all_anchors.size(); ++i) {
      for (auto j = 0u; j < all_anchors[i].size(); ++j) {
        std::cout << all_anchors[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
__TOC__(GENERATE_ANCHORS)
  return all_anchors;
}
}}}
