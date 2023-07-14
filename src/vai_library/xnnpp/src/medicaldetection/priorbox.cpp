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
#include "priorbox.hpp"
#include <utility>
#include <cmath>

namespace vitis { namespace ai { namespace medicaldetection {

PriorBox::PriorBox(const std::vector<int>& input_shape,
	           const std::vector<int>& feature_shapes,
	           const std::vector<int>& min_sizes,
	           const std::vector<int>& max_sizes,
	           const std::vector<float>& aspect_ratios,
	           const std::vector<int>& steps,
                   float offset  )
{
/*
    input_shape = [320, 320]
    feature_shapes = [(40, 40), (20, 20), (10, 10), (5, 5)]
    min_sizes = [(32,), (64,), (128,), (256,)]
    max_sizes = [(64,), (128,), (256,), (315,)]
    aspect_ratios = [(2.,), (2.,), (2.,), (2.,)]
    steps = [(8, 8), (16, 16), (32, 32), (64, 64)]
    offset=0.5
*/

  float f_h_s, f_w_s;
  for(auto i=0u; i<feature_shapes.size()/2; i++) {
    f_h_s = input_shape[0]/steps[i*2];
    f_w_s = input_shape[1]/steps[i*2+1];

    std::vector<std::pair<float, float>> prior_whs_ratios;
    float p_w, p_h;
    p_w = float(min_sizes[i])/input_shape[1];
    p_h = float(min_sizes[i])/input_shape[0];
    prior_whs_ratios.emplace_back(std::make_pair(p_w, p_h));
    auto size = sqrt(min_sizes[i] * max_sizes[i]);
    p_w = size/input_shape[1]; 
    p_h = size/input_shape[0]; 
    prior_whs_ratios.emplace_back(std::make_pair(p_w, p_h));

    auto s_alpha = sqrt(aspect_ratios[i]);
    p_w = float(min_sizes[i])/input_shape[1];
    p_h = float(min_sizes[i])/input_shape[0];
    prior_whs_ratios.emplace_back(std::make_pair(p_w*s_alpha, p_h/s_alpha));
    prior_whs_ratios.emplace_back(std::make_pair(p_w/s_alpha, p_h*s_alpha));

    for(auto h = 0; h< feature_shapes[i*2]; h++){
      for(auto w = 0; w< feature_shapes[i*2+1]; w++){
        auto cx = (w + offset) / f_w_s;
        auto cy = (h + offset) / f_h_s;
	for (auto& it: prior_whs_ratios) {
           prior_boxes.emplace_back(std::vector<float>( {cx, cy, std::get<0>(it), std::get<1>(it) } ));
           prior_boxes_ltrb.emplace_back(std::vector<float>( {
					      cx - std::get<0>(it)/2, 
					      cy - std::get<1>(it)/2, 
					      cx + std::get<0>(it)/2, 
					      cy + std::get<1>(it)/2 } ));
        }
      } // end for w
    } // end for h    
  } // end for i
}

}}}

