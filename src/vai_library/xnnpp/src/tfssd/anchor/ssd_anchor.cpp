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
#include "./ssd_anchor.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace vitis {
namespace ai {
namespace dptfssd {

using namespace std;

using std::copy_n;
using std::fill_n;
using std::make_pair;
using std::make_shared;
using std::sqrt;

SSDAnchor::SSDAnchor(int num_layers, bool reduce_boxes_in_lowest_layer,
                     float min_scale, float max_scale,
                     float interpolated_scale_aspect_ratio,
                     const std::vector<int>& feature_map_list,
                     const std::vector<float>& aspect_ratios, int image_width,
                     int image_height)
    : AnchorBase(num_layers, image_width, image_height) {
  // 1. calculate box_specs_list;
  // scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i
  // in range(num_layers)] + [1.0]
  //  scales : [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
  //  create_ssd_anchors in anchor_generators/multiple_grid_anchor_generator.py
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

#if 0
  std::cout <<"enter layer_box_specs.emplace_back " << std::endl;
  for(auto ii=0u; ii< box_specs_list.size(); ii++) {
    std::cout << "val of " << ii << ":";
    for(auto jj=0u; jj< box_specs_list[ii].size(); jj++) {
        std::cout << " " << std::get<0>(box_specs_list[ii][jj]) << "-" <<  std::get<1>(box_specs_list[ii][jj]) << "  ";
    }
    std::cout << std::endl;
  }
#endif

  // 3. still in _generate      in
  // anchor_generators/multiple_grid_anchor_generator.py initial is 1.0,  in
  // __init, it is set to 256 ?  no.
  std::tuple<float, float> base_anchor_size =
      std::make_tuple(1.0, 1.0);  //(256,256)

  // 4. in __init__  in anchor_generators/multiple_grid_anchor_generator.py
  std::vector<std::vector<float>> _scales, _aspect_ratios;
  std::vector<float> _scales_item, _aspect_ratios_item;

  // already test in python: that's good.
  // box+specs_list [[(11, 12), (13, 14), (15, 16)], [(2, 3), (3, 4), (4, 5),
  // (5, 6)]] scale:         [(11, 13, 15), (2, 3, 4, 5)] aspect_ratio:  [(12,
  // 14, 16), (3, 4, 5, 6)]

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

#if 0
  std::cout <<"enter _scales " << std::endl;
  for(auto ii=0u; ii< _scales.size(); ii++) {
    std::cout << "val of " << ii << ":";
    for(auto jj=0u; jj< _scales[ii].size(); jj++) {
        std::cout << " " << _scales[ii][jj] << "---" << _aspect_ratios[ii][jj] << "  ";
    }
    std::cout << std::endl;
  }
#endif

  // __init & create_ssd_anchor both finished now.   only left the _generate()
  // logic

  // 2. calculate anchor_strides
  //   _generate      in anchor_generators/multiple_grid_anchor_generator.py
  //      anchor_strides = [(1.0 / tf.cast(pair[0], dtype=tf.float32),
  //                        1.0 / tf.cast(pair[1], dtype=tf.float32))
  //                          for pair in feature_map_shape_list]
  //       anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
  //                         for stride in anchor_strides]

  std::vector<std::tuple<float, float>> anchor_strides, anchor_offsets;
  for (auto i = 0u; i < feature_map_list.size(); i += 2) {
    anchor_strides.emplace_back(std::make_tuple(1.0 / feature_map_list[i],
                                                1.0 / feature_map_list[i + 1]));
    anchor_offsets.emplace_back(std::make_tuple(0.5 / feature_map_list[i],
                                                0.5 / feature_map_list[i + 1]));
  }

  // 5. in _generate in anchor_generators/multiple_grid_anchor_generator.py

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
    tile_anchors(grid_height[i], grid_width[i], _scales[i], _aspect_ratios[i],
                 base_anchor_size, anchor_strides[i], anchor_offsets[i]);
  }

#if 0
  std::cout << "priors____ dump: priors_[0].size: " << priors_[0]->size() << "\n";
  for(auto &aa: priors_) {
     std:: cout << (*aa)[0] << " " << (*aa)[1] << " " << (*aa)[2] << " " << (*aa)[3] << std::endl;
  }
  std::cout << "\nShili: priors____ dump: end\n";
#endif
}

}  // namespace dptfssd
}  // namespace ai
}  // namespace vitis
