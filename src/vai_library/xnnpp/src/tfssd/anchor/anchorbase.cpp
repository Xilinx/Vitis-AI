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
#include "./anchorbase.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <type_traits>

namespace vitis {
namespace ai {
namespace dptfssd {

using namespace std;

using std::copy_n;
using std::fill_n;
using std::make_pair;
using std::make_shared;
using std::sqrt;

AnchorBase::AnchorBase(int num_layers, int image_width, int image_height)
    : num_layers_(num_layers),
      image_width_(image_width),
      image_height_(image_height) {
  // std::cout << "layer:" << num_layers_ << " " << image_width << std::endl;
}

void AnchorBase::tile_anchors(float grid_height, float grid_width, VF& _scales,
                              VF& _aspect_ratios,
                              std::tuple<float, float> base_anchor_size,
                              std::tuple<float, float> anchor_strides,
                              std::tuple<float, float> anchor_offsets) {
#if 0
     std::cout << "grid_height:" << grid_height << std::endl;
     std::cout << "grid_width:" << grid_width << std::endl;
     std::cout << "\n _scales:" ;
     for(auto & a: _scales) {
        std::cout << a << " ";
     }
     std::cout << "\n _aspect_ratios:" ;
     for(auto & a: _aspect_ratios) {
         std::cout << a << " "; 
     }
     std::cout << "\n base anchor size:" << std::get<0>(base_anchor_size) << "  " <<  std::get<1>(base_anchor_size) <<  std::endl;
     std::cout << "anchor_strides:" << std::get<0>(anchor_strides) << " " << std::get<1>(anchor_strides)  << std::endl;
     std::cout << "anchor_offsets:" << std::get<0>(anchor_offsets) << " " << std::get<1>(anchor_offsets)  << std::endl;
#endif

  //    tile_anchors() in anchor_generators/grid_anchor_generator.py
  std::vector<float> ratio_sqrts, heights, widths;
  for (auto j = 0u; j < _aspect_ratios.size(); j++) {
    ratio_sqrts.push_back(sqrt(_aspect_ratios[j]));
    float heights_v = _scales[j] / ratio_sqrts[j];
    float widths_v = _scales[j] * ratio_sqrts[j];
    heights_v = heights_v * std::get<0>(base_anchor_size);
    widths_v = widths_v * std::get<1>(base_anchor_size);
    heights.push_back(heights_v);
    widths.push_back(widths_v);
  }

#if 0
     printv("ratio_sqrts", ratio_sqrts);
     printv("heights", heights);
     printv("widths", widths);
#endif
  std::vector<float> y_centers;
  for (int j = 0; j < grid_height; j++) {
    float v =
        1.0 * j * std::get<0>(anchor_strides) + std::get<0>(anchor_offsets);
    y_centers.push_back(v);
  }
  // printv("y_centers", y_centers);

  std::vector<float> x_centers;
  for (int j = 0; j < grid_width; j++) {
    float v =
        1.0 * j * std::get<1>(anchor_strides) + std::get<1>(anchor_offsets);
    x_centers.push_back(v);
  }
  // printv("x_centers", x_centers);

  //  meshgrid : ....  ./anchor_generators/grid_anchor_generator.py" line 173 of
  //  213
  VVF x_centers2, y_centers2;
  std::tie(x_centers2, y_centers2) = meshgrid(x_centers, y_centers);

  // printv("x_centers2", x_centers2);
  // printv("y_centers2", y_centers2);

  VVVF widths_grid, x_centers_grid;
  std::tie(widths_grid, x_centers_grid) = meshgrid(widths, x_centers2);
  // printv("widths_grid", widths_grid);
  // printv("x_centers_grid", x_centers_grid);

  VVVF heights_grid, y_centers_grid;
  std::tie(heights_grid, y_centers_grid) = meshgrid(heights, y_centers2);
  // printv("heights_grid", heights_grid);
  // printv("y_centers_grid", y_centers_grid);

  VVVVF bbox_centers, bbox_sizes;
  bbox_centers = tfstack3(y_centers_grid, x_centers_grid);
  // printv("bbox_centers", bbox_centers);

  bbox_sizes = tfstack3(heights_grid, widths_grid);
  // printv("bbox_sizes", bbox_sizes);
  VVF bbox_centers2, bbox_sizes2;
  bbox_centers2 = tfreshape(bbox_centers);

  // printv("bbox_centers2", bbox_centers2);

  bbox_sizes2 = tfreshape(bbox_sizes);
  // printv("bbox_sizes2", bbox_sizes2);

  // bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
  // tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)
  // VVF bbox_corners; bbox_corners = tfconcat(bbox_centers2, bbox_sizes2);
  tfconcat_decode(bbox_centers2, bbox_sizes2, priors_);
}

/*
 *   below part is helper function ....
 */
void mywritefile(float* conf, int size1, std::string filename) {
  ofstream Tout;
  Tout.open(filename, ios_base::out | ios_base::binary);
  if (!Tout) {
    cout << "Can't open the file!";
    return;
  }
  for (int i = 0; i < size1; i++) {
    Tout.write((char*)conf + i * 4, 4);
  }
}

std::tuple<VVF, VVF> meshgrid(const VF& v1, const VF& v2) {
  VVF vvout1, vvout2;
  // std::cout << "meshgrid: in size: " << v1.size() << " " << v2.size() <<
  // std::endl;

  for (auto i = 0u; i < v2.size(); i++) {
    vvout1.push_back(v1);
    std::vector<float> vtmp2(v1.size(), v2[i]);
    vvout2.push_back(vtmp2);
  }
  return std::make_tuple(vvout1, vvout2);
}

std::tuple<VVVF, VVVF> meshgrid(const VF& v1, const VVF& vv2) {
  // v1: len 6;  vv2: len 19x19;   out: 19x19x6
  VVVF vvvout1, vvvout2;
  VVF vvtmp1, vvtmp2;
  for (auto i = 0u; i < vv2.size(); i++) {
    for (auto j = 0u; j < vv2[0].size(); j++) {
      vvtmp1.push_back(v1);
      std::vector<float> vtmp1(v1.size(), vv2[i][j]);
      vvtmp2.push_back(vtmp1);
    }
    vvvout1.push_back(vvtmp1);
    vvtmp1.clear();
    vvvout2.push_back(vvtmp2);
    vvtmp2.clear();
  }
  return std::make_tuple(vvvout1, vvvout2);
}

// only support the case for axis=3;
VVVVF tfstack3(const VVVF& vvv1, const VVVF& vvv2) {
  VVVVF vvvvout1;
  VVVF vvvtmp1;
  VVF vvtmp1;
  VF vtmp1;
  for (auto i = 0u; i < vvv1.size(); i++) {
    for (auto j = 0u; j < vvv1[0].size(); j++) {
      for (auto k = 0u; k < vvv1[0][0].size(); k++) {
        vtmp1.push_back(vvv1[i][j][k]);
        vtmp1.push_back(vvv2[i][j][k]);
        vvtmp1.push_back(vtmp1);
        vtmp1.clear();
      }
      vvvtmp1.push_back(vvtmp1);
      vvtmp1.clear();
    }
    vvvvout1.push_back(vvvtmp1);
    vvvtmp1.clear();
  }
  return vvvvout1;
}

VF tfreshape(const VVF& vv1) {
  VF out1;
  for (auto& a : vv1) {
    out1.insert(out1.end(), a.begin(), a.end());
  }
  // for(auto &a : out1) std::cout << "tfreshape: " << a << std::endl;

  return out1;
}

// only support the case for : [-1, 2]
VVF tfreshape(const VVVVF& vvvv1) {
  VVF vvout1;
  for (auto i = 0u; i < vvvv1.size(); i++) {
    for (auto j = 0u; j < vvvv1[0].size(); j++) {
      for (auto k = 0u; k < vvvv1[0][0].size(); k++) {
        vvout1.push_back(vvvv1[i][j][k]);
      }
    }
  }
  return vvout1;
}

void tfconcat_decode(
    const VVF& vv1_center, const VVF& vv2_size,
    std::vector<std::shared_ptr<std::vector<float>>>& priors_) {
  VF vtmp1;
#if 0 
  // never open this part!
  for (auto i=0u; i<vv1_center.size(); i++) {
      vtmp1.push_back( vv1_center[i][0] - 0.5* vv2_size[i][0]); 
      vtmp1.push_back( vv1_center[i][1] - 0.5* vv2_size[i][1]); 
      vtmp1.push_back( vv1_center[i][0] + 0.5* vv2_size[i][0]); 
      vtmp1.push_back( vv1_center[i][1] + 0.5* vv2_size[i][1]); 
      priors_.push_back( std::make_shared<std::vector<float>>(vtmp1)); 
      std::cout << "V:" << vtmp1[0] << " " << vtmp1[1] << " " << vtmp1[2] << " " << vtmp1[3] << std::endl;
      vtmp1.clear();
  }
#else
  /* in box_coders/faster_rcnn_box_coder.py _decode(),
     anchors.get_center_coordinates_and_sizes we need get the anchor's center &
     size, so we don't need to calculate it here. otherwise, we need do opposite
     again which is in #if 0 part.   sequence is :  y x h w
   */

  // std::cout << "in tfconcat_decode: vv1_center's size:" << vv1_center.size()
  // << std::endl;
  for (auto i = 0u; i < vv1_center.size(); i++) {
    vtmp1.push_back(vv1_center[i][0]);
    vtmp1.push_back(vv1_center[i][1]);
    vtmp1.push_back(vv2_size[i][0]);
    vtmp1.push_back(vv2_size[i][1]);
    // std::cout << "pribox:  : " << vtmp1[0] << " " << vtmp1[1] << " " <<
    // vtmp1[2] << " " << vtmp1[3] << std::endl;
    priors_.push_back(std::make_shared<std::vector<float>>(vtmp1));
    vtmp1.clear();
  }
#endif
}

void printv(std::string info, const VF& inv) {
  std::cout << "Debug ---------- " << info << std::endl;
  for (auto& v : inv) std::cout << " " << v << " ";
  std::cout << std::endl;
}

void printv(std::string info, const VVF& inv) {
  std::cout << "Debug ---------- " << info << std::endl;
  for (auto& v : inv) {
    for (auto& v1 : v) std::cout << " " << v1 << " ";
    std::cout << std::endl;
  }
}
void printv(std::string info, const VVVF& inv) {
  std::cout << "Debug ---------- " << info << std::endl;
  for (auto& v : inv) {
    std::cout << " * ";
    for (auto& v1 : v) {
      for (auto& v2 : v1) std::cout << " " << v2 << " ";
      std::cout << std::endl;
    }
  }
}

void printv(std::string info, const VVVVF& inv) {
  std::cout << "Debug ---------- " << info << std::endl;
  for (auto& v : inv) {
    std::cout << " * ";
    for (auto& v1 : v) {
      std::cout << " X ";
      for (auto& v2 : v1) {
        for (auto& v3 : v2) std::cout << " " << v3 << " ";
        std::cout << std::endl;
      }
    }
  }
}

}  // namespace dptfssd
}  // namespace ai
}  // namespace vitis
