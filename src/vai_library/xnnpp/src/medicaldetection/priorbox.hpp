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
#include <vector>

namespace vitis { namespace ai { namespace medicaldetection {

class PriorBox{ 
public:
  PriorBox( const std::vector<int>& input_shape,
            const std::vector<int>& feature_shapes,
            const std::vector<int>& min_sizes,
            const std::vector<int>& max_sizes,
            const std::vector<float>& aspect_ratios,
            const std::vector<int>& steps,
            float offset
	  );
  const std::vector<std::vector<float>>& get_pirors() {
     return prior_boxes;
  }
  const std::vector<std::vector<float>>& get_pirors_ltrb() {
     return prior_boxes_ltrb;
  }
private:
  std::vector<std::vector<float>> prior_boxes;	  // not used 
  std::vector<std::vector<float>> prior_boxes_ltrb; // this one used
};

}}}
