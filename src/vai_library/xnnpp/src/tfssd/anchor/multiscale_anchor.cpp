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
#include "./multiscale_anchor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace vitis {
namespace ai {
namespace dptfssd {

using namespace std;

using std::make_pair;
using std::make_shared;
using std::sqrt;

using std::copy_n;

MultiscaleAnchor::MultiscaleAnchor(int min_level, int max_level,
                                   float anchor_scale, int scales_per_octave,
                                   const std::vector<int>& feature_map_list,
                                   const std::vector<float>& aspect_ratios,
                                   int image_width, int image_height)
    : AnchorBase(max_level - min_level + 1, image_width, image_height) {
  // __init__ in anchor_generators/multiscale_grid_anchor_generator.py

#if 0
  std::cout <<"MultiscaleAnchor::MultiscaleAnchor: " << min_level << " " 
	    << max_level << " " << anchor_scale << " " 
            << scales_per_octave << " featuremap_list-size:"  << feature_map_list.size() << " " 
            << aspect_ratios.size() << " " << image_width << " " << image_height << std::endl;
#endif
  std::vector<float> scales;
  for (auto i = 0; i < scales_per_octave; i++) {
    scales.push_back(pow(2.0, (i * 1.0) / scales_per_octave));
    // std::cout << "  scales:" << scales[i] << std::endl;
  }

  std::vector<std::tuple<float, float>> anchor_strides, base_anchor_size,
      anchor_offsets;

  // _generate in anchor_generators/multiscale_grid_anchor_generator.py
  for (auto i = 0; i < max_level - min_level + 1; i++) {
    auto v = pow(2.0, i + min_level);
    // see below detailed comments for why "/(image_height*1.0)" is needed.
    anchor_strides.push_back(
        std::make_tuple(v / (image_height * 1.0), v / (image_width * 1.0)));
    v *= anchor_scale;
    // base_anchor_size.push_back(std::make_tuple(v, v));
    /* Note: in _generate in
       anchor_generators/multiscale_grid_anchor_generator.py, base_anchor_size
       is not normolized, it use the big coordinate value like 64,64, which
       cause priorbox value be in real corrdinate. That's not correct.  we'd
       better fix it before transfering it to tile_anchors(). we can also fix it
       after tile_anchors(), but it's ugly.
     */
    base_anchor_size.push_back(
        std::make_tuple(v / (image_height * 1.0), v / (image_width * 1.0)));

    auto level = i + min_level;
    auto stride = pow(2.0, level);
    auto anchor_offset_w = 0, anchor_offset_h = 0;
    if (image_width % int(pow(2.0, level) + 0.01) == 0) {
      anchor_offset_w = stride / 2.0;
    }
    if (image_height % int(pow(2.0, level) + 0.01) == 0) {
      anchor_offset_h = stride / 2.0;
    }
    anchor_offsets.push_back(
        std::make_tuple(anchor_offset_h / (image_height * 1.0),
                        anchor_offset_w / (image_width * 1.0)));
#if 0
     std::cout << "anchor_offset:" << anchor_offset_h << " " << anchor_offset_w 
               << "  anchor_stride:" <<  std::get<0>(anchor_strides[i]) 
               << "  base_anchor_size:" <<  std::get<0>(base_anchor_size[i])
               << std::endl;
#endif
  }

  // ok, all variables are available now.  will call ag =
  // grid_anchor_generator.GridAnchorGenerator next, (anchor_grid,) =
  // ag.generate(feature_map_shape_list=[(feat_h, feat_w)]) scales_grid,
  // aspect_ratios_grid = ops.meshgrid  in  _generate in
  // grid_anchor_generator.py
  VVF scales_grid, aspect_ratios_grid;
  std::tie(scales_grid, aspect_ratios_grid) = meshgrid(scales, aspect_ratios);
  VF scales_grid_reshape = tfreshape(scales_grid);
  VF aspect_ratios_grid_reshape = tfreshape(aspect_ratios_grid);

  //  std::cout << "MultiscaleAnchor::MultiscaleAnchor:  scales_grid
  //  aspect_ratios_grid size:"
  //            << scales_grid.size() << " " << aspect_ratios_grid.size() <<
  //            "\n";

  for (int i = 0; i < num_layers_; i++) {
    tile_anchors(feature_map_list[i * 2 + 1], feature_map_list[i * 2],
                 scales_grid_reshape, aspect_ratios_grid_reshape,
                 base_anchor_size[i], anchor_strides[i], anchor_offsets[i]);
  }

#if 0
  std::cout << "priors____ dump: \n";
  for(auto &aa: priors_) {
     std:: cout << (*aa)[0] << " " << (*aa)[1] << " " << (*aa)[2] << " " << (*aa)[3] << std::endl;
  }
  std::cout << "\npriors____ dump: end\n";
#endif
}

}  // namespace dptfssd
}  // namespace ai
}  // namespace vitis
