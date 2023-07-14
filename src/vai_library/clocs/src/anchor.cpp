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

#include "./anchor.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;

DEF_ENV_PARAM(DEBUG_CLOCS_POINTPILLARS, "0")
DEF_ENV_PARAM(DEBUG_XNNPP_ANCHOR, "0")

namespace vitis {
namespace ai {
namespace clocs {

static float limit_period(float val) {
  return val - floor(val / 3.14159265 + 0.5) * 3.14159265;
}

vector<vector<float>> rbbox2d_to_near_bbox(const vector<vector<float>>& v2f,
                                           const std::vector<int>& idx) {
  vector<vector<float>> v2fo(v2f.size());
  for (unsigned int i = 0; i < v2f.size(); i++) {
    if (abs(limit_period(v2f[i][idx[4]])) > 3.14159265 / 4.0) {
      vector<float>{v2f[i][idx[0]] - v2f[i][idx[3]] / 2,
                    v2f[i][idx[1]] - v2f[i][idx[2]] / 2,
                    v2f[i][idx[0]] + v2f[i][idx[3]] / 2,
                    v2f[i][idx[1]] + v2f[i][idx[2]] / 2}
          .swap(v2fo[i]);
    } else {
      vector<float>{v2f[i][idx[0]] - v2f[i][idx[2]] / 2,
                    v2f[i][idx[1]] - v2f[i][idx[3]] / 2,
                    v2f[i][idx[0]] + v2f[i][idx[2]] / 2,
                    v2f[i][idx[1]] + v2f[i][idx[3]] / 2}
          .swap(v2fo[i]);
    }
  }
  return v2fo;
}

vector<float> fused_get_anchors_area(const vector<int>& dense_map,
                                     const Anchors& anchor_bv,
                                     const vector<float> stride,
                                     const vector<float> offset,
                                     vector<int> grid_size) {
  vector<float> ret(anchor_bv.size());
  vector<int> anchor_coor(4);
  int grid_size_x = 432 - 1;
  int grid_size_y = 496 - 1;
  for (auto i = 0u; i < anchor_bv.size(); ++i) {
    anchor_coor[0] = floor((anchor_bv[i][0] - offset[0]) / stride[0]);
    anchor_coor[1] = floor((anchor_bv[i][1] - offset[1]) / stride[1]);
    anchor_coor[2] = floor((anchor_bv[i][2] - offset[0]) / stride[0]);
    anchor_coor[3] = floor((anchor_bv[i][3] - offset[1]) / stride[1]);
    anchor_coor[0] = max(anchor_coor[0], 0);
    anchor_coor[1] = max(anchor_coor[1], 0);
    anchor_coor[2] = min(anchor_coor[2], grid_size_x);
    anchor_coor[3] = min(anchor_coor[3], grid_size_y);
    auto ID = dense_map[anchor_coor[3] * 432 + anchor_coor[2]];
    auto IA = dense_map[anchor_coor[1] * 432 + anchor_coor[0]];
    auto IB = dense_map[anchor_coor[3] * 432 + anchor_coor[0]];
    auto IC = dense_map[anchor_coor[1] * 432 + anchor_coor[2]];
    ret[i] = ID - IB - IC + IA;

    // if (i == 34289) {
    //  std::cout << i << ":"
    //            << " ID:" << ID << " IA:" << IA << " IB:" << IB << " IC:" <<
    //            IC
    //            << std::endl;
    //  std::cout << "anchor_coor:";
    //  for (auto j = 0u; j < anchor_coor.size(); ++j) {
    //    std::cout << anchor_coor[j] << " ";
    //  }
    //  std::cout << std::endl;
    //  std::cout << "anchor_bv:";
    //  for (auto j = 0u; j < anchor_bv[i].size(); ++j) {
    //    std::cout << anchor_bv[i][j] << " ";
    //  }
    //  std::cout << std::endl;
    //}
  }
  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./anchors_area.txt");
    for (auto r : ret) {
      out << r << " ";
      out << std::endl;
    }
  }

  return ret;
}

vector<float> get_centers(int featmap_size, float stride, float offset) {
  vector<float> centers(featmap_size, 0);
  for (auto i = 0u; i < centers.size(); ++i) {
    centers[i] = i * stride + offset;
  }
  return centers;
}

Anchors get_anchors_bv(const Anchors& anchors) {
  vector<int> idxes{0, 1, 3, 4, 6};
  auto anchors_bv = rbbox2d_to_near_bbox(anchors, idxes);
  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./anchors_bv.txt");
    for (auto& a : anchors_bv) {
      for (auto i = 0u; i < a.size(); ++i) {
        out << a[i] << " ";
      }
      out << std::endl;
    }
  }

  return anchors_bv;
}

// another form of anchor mask
vector<size_t> get_valid_anchor_index(const vector<int>& coors, int nx, int ny,
                                      const Anchors& anchor_bv,
                                      float anchor_area_thresh,
                                      const vector<float>& voxel_size,
                                      const vector<float>& pc_range,
                                      const vector<int>& grid_size) {
  __TIC__(CUMSUM)
  vector<int> dense_map(nx * ny, 0);
  auto size = coors.size() / 4;
  for (auto i = 0u; i < size; ++i) {
    dense_map[coors[i * 4 + 2] * nx + coors[i * 4 + 3]] = 1;
  }

  // sum
  for (auto y = 1; y < ny; ++y) {
    for (auto x = 0; x < nx; ++x) {
      dense_map[y * nx + x] += dense_map[(y - 1) * nx + x];
    }
  }

  for (auto x = 1; x < nx; ++x) {
    for (auto y = 0; y < ny; ++y) {
      dense_map[y * nx + x] += dense_map[y * nx + x - 1];
    }
  }
  __TOC__(CUMSUM)
  __TIC__(FUSED_GET_ANCHORS_AREA)
  auto anchor_area = fused_get_anchors_area(dense_map, anchor_bv, voxel_size,
                                            pc_range, grid_size);
  __TOC__(FUSED_GET_ANCHORS_AREA)
  __TIC__(SELECT_ANCHOR_INDEX)
  vector<size_t> anchor_indices;
  anchor_indices.reserve(20000);
  for (auto i = 0u; i < anchor_area.size(); ++i) {
    if (anchor_area[i] > anchor_area_thresh) {
      anchor_indices.emplace_back(i);
    }
    // if (i == 34289) {
    //  std::cout << i << ":" << anchor_area[i] << std::endl;
    //}
  }
  __TOC__(SELECT_ANCHOR_INDEX)

  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./anchor_idx.txt");
    for (auto i = 0u; i < anchor_indices.size(); ++i) {
      out << anchor_indices[i] << std::endl;
    }
  }
  return std::move(anchor_indices);
}

vector<bool> get_anchor_mask(const vector<int>& coors, int nx, int ny,
                             const Anchors& anchor_bv, float anchor_area_thresh,
                             const vector<float>& voxel_size,
                             const vector<float>& pc_range,
                             const vector<int>& grid_size) {
  vector<int> dense_map(nx * ny, 0);
  auto size = coors.size() / 4;
  for (auto i = 0u; i < size; ++i) {
    dense_map[coors[i * 4 + 2] * nx + coors[i * 4 + 3]] = 1;
  }
  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./dense_map.txt");
    for (auto i = 0; i < ny; ++i) {
      for (auto j = 0; j < nx; ++j) {
        out << dense_map[i * nx + j] << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  // sum
  for (auto y = 1; y < ny; ++y) {
    for (auto x = 0; x < nx; ++x) {
      dense_map[y * nx + x] += dense_map[(y - 1) * nx + x];
    }
  }

  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./dense_map1.txt");
    for (auto i = 0; i < ny; ++i) {
      for (auto j = 0; j < nx; ++j) {
        out << dense_map[i * nx + j] << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  for (auto x = 1; x < nx; ++x) {
    for (auto y = 0; y < ny; ++y) {
      dense_map[y * nx + x] += dense_map[y * nx + x - 1];
    }
  }

  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./dense_map2.txt");
    for (auto i = 0; i < ny; ++i) {
      for (auto j = 0; j < nx; ++j) {
        out << dense_map[i * nx + j] << " ";
      }
      out << std::endl;
    }
    out.close();
  }

  auto anchor_area = fused_get_anchors_area(dense_map, anchor_bv, voxel_size,
                                            pc_range, grid_size);
  vector<bool> mask(anchor_bv.size(), false);
  for (auto i = 0u; i < mask.size(); ++i) {
    if (anchor_area[i] > anchor_area_thresh) {
      mask[i] = true;
    }
    // if (i == 34289) {
    //  std::cout << i << ":" << anchor_area[i] << std::endl;
    //}
  }
  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./anchor_mask_idx.txt");
    for (auto i = 0u; i < mask.size(); ++i) {
      if (mask[i] == true) {
        out << i << std::endl;
      }
    }
  }

  return mask;
}

Anchors generate_anchors_stride(const AnchorInfo& params) {
  auto x_centers =
      get_centers(params.featmap_size[2], params.strides[0], params.offsets[0]);
  auto y_centers =
      get_centers(params.featmap_size[1], params.strides[1], params.offsets[1]);
  auto z_centers =
      get_centers(params.featmap_size[0], params.strides[2], params.offsets[2]);
  auto rs = params.rotations;
  auto last_dim = 4 + params.sizes.size();
  auto size =
      x_centers.size() * y_centers.size() * z_centers.size() * rs.size();
  vector<vector<float>> anchors(size, vector<float>(last_dim));
  for (auto x = 0u; x < x_centers.size(); ++x) {
    for (auto y = 0u; y < y_centers.size(); ++y) {
      for (auto z = 0u; z < z_centers.size(); ++z) {
        for (auto r = 0u; r < rs.size(); ++r) {
          // auto index = x * y_centers.size() * z_centers.size() * rs.size() +
          //             y * z_centers.size() * rs.size() + z * rs.size() + r;
          auto index = z * y_centers.size() * x_centers.size() * rs.size() +
                       y * x_centers.size() * rs.size() + x * rs.size() + r;
          anchors[index][0] = x_centers[x];
          anchors[index][1] = y_centers[y];
          anchors[index][2] = z_centers[z];
          for (auto s = 0u; s < params.sizes.size(); ++s) {
            anchors[index][3 + s] = params.sizes[s];
          }
          anchors[index][last_dim - 1] = rs[r];
        }
      }
    }
  }
  if (ENV_PARAM(DEBUG_XNNPP_ANCHOR)) {
    std::ofstream out("./anchors.txt");
    for (auto& a : anchors) {
      for (auto i = 0u; i < a.size(); ++i) {
        out << a[i] << " ";
      }
      out << std::endl;
    }
  }
  return anchors;
}

}  // namespace clocs
}  // namespace ai
}  // namespace vitis
