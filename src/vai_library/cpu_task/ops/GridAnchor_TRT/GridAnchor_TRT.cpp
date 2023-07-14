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
#include <math.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"

using namespace std;

namespace {

using VF = std::vector<float>;
using VVF = std::vector<VF>;
using VVVF = std::vector<VVF>;
using VVVVF = std::vector<VVVF>;

std::tuple<VVF, VVF> meshgrid(const VF& v1, const VF& v2);
std::tuple<VVVF, VVVF> meshgrid(const VF& v1, const VVF& vv2);
VVVVF tfstack3(const VVVF& vvv1, const VVVF& vvv2);
VVF tfreshape(const VVVVF& vvvv1);
void tfconcat_decode(const VVF& vv1_center, const VVF& vv2_size,
                     std::vector<float>& priors_);

template<typename T>
void printv(std::string info, const T& inv) {
  std::cout << "Debug ---------- " << info << std::endl;
  for (auto& v : inv) std::cout << " " << v << " ";
  std::cout << std::endl;
}

class AnchorBase {
 public:
  AnchorBase(int num_layers);

  std::vector<float> priors_;
  int num_layers_;

  void tile_anchors(float grid_height, float grid_width, VF& _scales,
                    VF& _aspect_ratios,
                    std::tuple<float, float> base_anchor_size,
                    std::tuple<float, float> anchor_strides,
                    std::tuple<float, float> anchor_offsets);
  
  void create_anchors(int num_layers, bool reduce_boxes_in_lowest_layer, float min_scale,
            float max_scale, float interpolated_scale_aspect_ratio,
            const std::vector<int>& feature_map_list,
            const std::vector<double>& aspect_ratios, int image_width,
            int image_height);
};

AnchorBase::AnchorBase(int num_layers)
    : num_layers_(num_layers){
}

void AnchorBase::tile_anchors(float grid_height, float grid_width, VF& _scales,
                              VF& _aspect_ratios,
                              std::tuple<float, float> base_anchor_size,
                              std::tuple<float, float> anchor_strides,
                              std::tuple<float, float> anchor_offsets) {
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
  std::vector<float> y_centers;
  for (int j = 0; j < grid_height; j++) {
    float v =
        1.0 * j * std::get<0>(anchor_strides) + std::get<0>(anchor_offsets);
    y_centers.push_back(v);
  }

  std::vector<float> x_centers;
  for (int j = 0; j < grid_width; j++) {
    float v =
        1.0 * j * std::get<1>(anchor_strides) + std::get<1>(anchor_offsets);
    x_centers.push_back(v);
  }

  VVF x_centers2, y_centers2;
  std::tie(x_centers2, y_centers2) = meshgrid(x_centers, y_centers);


  VVVF widths_grid, x_centers_grid;
  std::tie(widths_grid, x_centers_grid) = meshgrid(widths, x_centers2);

  VVVF heights_grid, y_centers_grid;
  std::tie(heights_grid, y_centers_grid) = meshgrid(heights, y_centers2);

  VVVVF bbox_centers, bbox_sizes;
  bbox_centers = tfstack3(y_centers_grid, x_centers_grid);

  bbox_sizes = tfstack3(heights_grid, widths_grid);
  VVF bbox_centers2, bbox_sizes2;
  bbox_centers2 = tfreshape(bbox_centers);

  bbox_sizes2 = tfreshape(bbox_sizes);

  tfconcat_decode(bbox_centers2, bbox_sizes2, priors_);
}

std::tuple<VVF, VVF> meshgrid(const VF& v1, const VF& v2) {
  VVF vvout1, vvout2;
  for (auto i = 0u; i < v2.size(); i++) {
    vvout1.push_back(v1);
    std::vector<float> vtmp2(v1.size(), v2[i]);
    vvout2.push_back(vtmp2);
  }
  return std::make_tuple(vvout1, vvout2);
}

std::tuple<VVVF, VVVF> meshgrid(const VF& v1, const VVF& vv2) {
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
    const VVF& vv1_center, const VVF& vv2_size, std::vector<float>& priors_) {
  VF vtmp1;
  for (auto i = 0u; i < vv1_center.size(); i++) {
    vtmp1.push_back(vv1_center[i][0]-vv2_size[i][0]/2.0  );
    vtmp1.push_back(vv1_center[i][1]-vv2_size[i][1]/2.0  );
    vtmp1.push_back(vv1_center[i][0]+vv2_size[i][0]/2.0  );
    vtmp1.push_back(vv1_center[i][1]+vv2_size[i][1]/2.0  );

    priors_.insert(priors_.end(), vtmp1.begin(), vtmp1.end());
    vtmp1.clear();
  }
}

void AnchorBase::create_anchors(int num_layers, bool reduce_boxes_in_lowest_layer,
                     float min_scale, float max_scale,
                     float interpolated_scale_aspect_ratio,
                     const std::vector<int>& feature_map_list,
                     const std::vector<double>& aspect_ratios, int image_width,
                     int image_height)
{
  std::vector<float> scales;
  for (auto i = 0; i < num_layers; i++) {
    scales.emplace_back(min_scale + (max_scale - min_scale) * (float)i /
                                        float(num_layers - 1));
  }
  scales.emplace_back(1.0);

  std::vector<std::tuple<float, float>> layer_box_specs;
  std::vector<std::vector<std::tuple<float, float>>> box_specs_list;

  for (auto i = 0; i < num_layers; i++) {
    layer_box_specs.clear();

    if (i == 0 && reduce_boxes_in_lowest_layer) {
      layer_box_specs.emplace_back(0.1, 1.0);
      layer_box_specs.emplace_back(scales[0], 2.0);
      layer_box_specs.emplace_back(scales[0], 0.5);
    } else {
      for (auto& r : aspect_ratios) {
        layer_box_specs.emplace_back(scales[i], r);
      }

      if (interpolated_scale_aspect_ratio > 0.0) {
        layer_box_specs.emplace_back(sqrt(scales[i] * scales[i + 1]),
                                     interpolated_scale_aspect_ratio);
      }
    }
    box_specs_list.push_back(std::move(layer_box_specs));
  }

  std::tuple<float, float> base_anchor_size =
      std::make_tuple(1.0, 1.0);  //(256,256)

  std::vector<std::vector<float>> _scales, _aspect_ratios;
  std::vector<float> _scales_item, _aspect_ratios_item;

  for (auto& box : box_specs_list) {
    for (auto& aa : box) {
      _scales_item.push_back(std::get<0>(aa));
      _aspect_ratios_item.push_back(std::get<1>(aa));
    }
    _scales.push_back(_scales_item);
    _aspect_ratios.push_back(_aspect_ratios_item);
    _scales_item.clear();
    _aspect_ratios_item.clear();
  }

  std::vector<std::tuple<float, float>> anchor_strides, anchor_offsets;
  for (auto i = 0u; i < feature_map_list.size(); i += 2) {
    anchor_strides.emplace_back(std::make_tuple(1.0 / feature_map_list[i],
                                                1.0 / feature_map_list[i + 1]));
    anchor_offsets.emplace_back(std::make_tuple(0.5 / feature_map_list[i],
                                                0.5 / feature_map_list[i + 1]));
  }

  auto min_im_shape = std::min(image_height, image_width);
  float scale_height = (min_im_shape * 1.0) / (image_height * 1.0);
  float scale_width = (min_im_shape * 1.0) / (image_width * 1.0);

  base_anchor_size =
      std::make_tuple(std::get<0>(base_anchor_size) * scale_height,
                      std::get<1>(base_anchor_size) * scale_width);

  VF grid_height, grid_width;
  for (auto i = 0u; i < feature_map_list.size() / 2; i++) {
    grid_height.push_back(feature_map_list[i * 2]);
    grid_width.push_back(feature_map_list[i * 2 + 1]);
  }

  for (auto i = 0; i < num_layers; i++) {
    tile_anchors(grid_height[i], 
                 grid_width[i], 
                 _scales[i], 
                 _aspect_ratios[i],
                 base_anchor_size, 
                 anchor_strides[i], 
                 anchor_offsets[i]);
  }
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs}{

    num_layers = op->get_attr<int>("num_layers_u_int");
    reduce_boxes_in_lowest_layer = op->get_attr<bool>("reduce_boxes_in_lowest_layer_u_bool");
    min_scale = op->get_attr<double>("min_scale_u_float");
    max_scale = op->get_attr<double>("max_scale_u_float");
    interpolated_scale_aspect_ratio = op->get_attr<double>("interpolated_scale_aspect_ratio_u_float");
    feature_map_list =  op->get_attr<std::vector<int32_t>>("feature_map_shape_list_u_ilist");
    aspect_ratios =  op->get_attr<std::vector<double>>("aspect_ratios_u_flist");
    image_height = op->get_attr<int>("im_height_u_int");
    image_width  = op->get_attr<int>("im_width_u_int");

    AnchorBase anchor(num_layers);
    anchor.create_anchors(
            num_layers,
            reduce_boxes_in_lowest_layer,
            min_scale,
            max_scale,
            interpolated_scale_aspect_ratio,
            feature_map_list,
            aspect_ratios,
            image_width,
            image_height);
    priors_.swap(anchor.priors_);
  }

  int calculate(vart::simple_tensor_buffer_t<void> output, std::vector<vart::simple_tensor_buffer_t<void>> input) {
    float* outlayer = (float*)output.data; (void)outlayer;
    output_shape = output.tensor->get_shape();  // 1917 4
    CHECK_EQ(output_shape[0]*output_shape[1], priors_.size());
    memcpy((void*)outlayer, (void*)priors_.data(), priors_.size()*sizeof(float));
    if(0)  printv("anchor :",  priors_);
    return 0;
  }
private:
  int num_layers = 6;
  bool reduce_boxes_in_lowest_layer = true;
  double min_scale = 0.2;
  double max_scale = 0.95;
  double interpolated_scale_aspect_ratio =1.0;
  std::vector<int> feature_map_list{19,19,10,10,5,5,3,3,2,2,1,1};
  std::vector<double> aspect_ratios{1.0, 2.0, 0.5, 3.0, 0.3333};
  int image_width = 300;
  int image_height = 300;
  std::vector<std::int32_t> output_shape;
  std::vector<float> priors_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
