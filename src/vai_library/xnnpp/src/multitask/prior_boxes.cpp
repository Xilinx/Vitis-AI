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

#include "./prior_boxes.hpp"

#include <algorithm>
#include <cmath>

namespace vitis {
namespace ai {
namespace multitask {

using std::copy_n;
using std::fill_n;
using std::make_pair;
using std::make_shared;
using std::sqrt;
using std::vector;

PriorBoxes::PriorBoxes(int image_width, int image_height, int layer_width,
                       int layer_height, const vector<float>& variances,
                       const vector<float>& min_sizes,
                       const vector<float>& max_sizes,
                       const vector<float>& aspect_ratios, float offset,
                       float step_width, float step_height, bool flip,
                       bool clip)
    : offset_(offset), clip_(clip) {
  // CHECK_GT(min_sizes.size(), 0);
  // if (!max_sizes.empty()) CHECK_EQ(min_sizes.size(), max_sizes.size());

  // Store image dimensions and layer dimensions
  image_dims_ = std::make_pair(image_width, image_height);
  layer_dims_ = std::make_pair(layer_width, layer_height);

  // Compute step width and height
  if (step_width == 0 || step_height == 0) {
    step_dims_ =
        make_pair(static_cast<float>(image_dims_.first) / layer_dims_.first,
                  static_cast<float>(image_dims_.second) / layer_dims_.second);
  } else {
    step_dims_ = std::make_pair(step_width, step_height);
  }

  // Store box variances
  if (variances.size() == 4) {
    variances_ = variances;
  } else if (variances.size() == 1) {
    variances_.resize(4);
    fill_n(variances_.begin(), 4, variances[0]);
  } else {
    variances_.resize(4);
    fill_n(variances_.begin(), 4, 0.1f);
  }

  // Generate boxes' dimensions
  for (size_t i = 0; i < min_sizes.size(); ++i) {
    // first prior: aspect_ratio = 1, size = min_size
    boxes_dims_.emplace_back(min_sizes[i], min_sizes[i]);
    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
    if (!max_sizes.empty()) {
      boxes_dims_.emplace_back(sqrt(min_sizes[i] * max_sizes[i]),
                               sqrt(min_sizes[i] * max_sizes[i]));
    }
    // rest of priors
    for (auto ar : aspect_ratios) {
      float w = min_sizes[i] * sqrt(ar);
      float h = min_sizes[i] / sqrt(ar);
      boxes_dims_.emplace_back(w, h);
      if (flip) boxes_dims_.emplace_back(h, w);
    }
  }

  // automatically create priors
  CreatePriors();
}

void PriorBoxes::CreatePriors() {
  for (int h = 0; h < layer_dims_.second; ++h) {
    for (int w = 0; w < layer_dims_.first; ++w) {
      float center_x = (w + offset_) * step_dims_.first;
      float center_y = (h + offset_) * step_dims_.second;
      for (auto& dims : boxes_dims_) {
        auto box = make_shared<vector<float>>(12);
        // xmin, ymin, xmax, ymax
        (*box)[0] = (center_x - dims.first / 2.) / image_dims_.first;
        (*box)[1] = (center_y - dims.second / 2.) / image_dims_.second;
        (*box)[2] = (center_x + dims.first / 2.) / image_dims_.first;
        (*box)[3] = (center_y + dims.second / 2.) / image_dims_.second;

        if (clip_) {
          for (int i = 0; i < 4; ++i)
            (*box)[i] = std::min(std::max((*box)[i], 0.f), 1.f);
        }
        // variances
        copy_n(variances_.begin(), 4, box->data() + 4);
        // centers and dimensions
        (*box)[8] = 0.5f * ((*box)[0] + (*box)[2]);
        (*box)[9] = 0.5f * ((*box)[1] + (*box)[3]);
        (*box)[10] = (*box)[2] - (*box)[0];
        (*box)[11] = (*box)[3] - (*box)[1];

        priors_.push_back(std::move(box));
      }
    }
  }
}

}  // namespace multitask
}  // namespace ai
}  // namespace vitis
