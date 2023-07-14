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
#include <fstream>
#include <iostream>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"
using namespace std;
template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    min_sizes_ = op->get_attr<std::vector<float>>("min_sizes");
    for (auto min_size : min_sizes_) {
      CHECK_GT(min_size, 0) << "min_size must be positive.";
    }
    flip_ = op->get_attr<bool>("flip");
    clip_ = op->get_attr<bool>("clip");
    aspect_ratios_.push_back(1.0f);
    auto aspect_ratios_param = op->get_attr<std::vector<float>>("aspect_ratio");
    for (auto i = 0u; i < aspect_ratios_param.size(); ++i) {
      float ar = aspect_ratios_param[i];
      bool already_exist = false;
      for (auto j = 0u; j < aspect_ratios_.size(); ++j) {
        if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
          already_exist = true;
          break;
        }
      }
      if (!already_exist) {
        aspect_ratios_.push_back(ar);
        if (flip_) {
          aspect_ratios_.push_back(1.0f / ar);
        }
      }
    }
    num_priors_ = aspect_ratios_.size() * min_sizes_.size();

    max_sizes_ = op->get_attr<std::vector<float>>("max_sizes");
    if (!max_sizes_.empty()) {
      CHECK_EQ(min_sizes_.size(), max_sizes_.size())
          << "wrong parameter setting";
      for (auto i = 0u; i < max_sizes_.size(); ++i) {
        CHECK_GT(max_sizes_[i], min_sizes_[i])
            << "max_size must be greater than min_size.";
        num_priors_ += 1;
      }
    }
    variance_ = op->get_attr<std::vector<float>>("variance");
    if (variance_.size() > 1u) {
      CHECK_EQ(variance_.size(), 4u) << "must and only provide 4 variance.";
    } else if (variance_.size() == 1u) {
      // OK
    } else {
      variance_ = {1.0f};
    }
    auto input_ops = op->get_input_ops("input");
    CHECK_EQ(input_ops.size(), 2u);
    auto image_op = input_ops[1];
    auto image_shape = image_op->get_output_tensor()->get_shape();
    CHECK_EQ(image_shape.size(), 4u);
    image_height_ = image_shape[1];
    image_width_ = image_shape[2];

    auto layer_op = input_ops[0];
    auto layer_shape = layer_op->get_output_tensor()->get_shape();
    CHECK_EQ(layer_shape.size(), 4u);
    layer_height_ = layer_shape[1];
    layer_width_ = layer_shape[2];

    auto my_shape = op->get_output_tensor()->get_shape();
    CHECK_EQ(my_shape.size(), 3);
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    CHECK_EQ(my_shape[1], 2);
    auto dim = layer_height_ * layer_width_ * num_priors_ * 4;
    CHECK_EQ(my_shape[2], dim);

    step_h_ = static_cast<float>(image_height_) / layer_height_;
    step_w_ = static_cast<float>(image_width_) / layer_width_;
    if (op->has_attr("step")) {
      auto step_param = op->get_attr<std::vector<float>>("step");
      auto step_param_size = step_param.size();
      CHECK(step_param_size == 2u || step_param_size == 1u);
      switch (step_param_size) {
        case 1u:
          step_h_ = step_param[0];
          step_w_ = step_param[0];
          break;
        case 2u:
          step_h_ = step_param[0];
          step_w_ = step_param[1];
          break;
      }
    }
    offset_ = op->get_attr<float>("offset");

    top_data_.resize(dim * 2);
    int idx = 0;
    for (int h = 0; h < layer_height_; ++h) {
      for (int w = 0; w < layer_width_; ++w) {
        float center_x = (w + offset_) * step_w_;
        float center_y = (h + offset_) * step_h_;
        float box_width, box_height;
        for (auto s = 0u; s < min_sizes_.size(); ++s) {
          int min_size_ = min_sizes_[s];
          // first prior: aspect_ratio = 1, size = min_size
          box_width = box_height = min_size_;
          // xmin
          top_data_[idx++] = (center_x - box_width / 2.) / image_width_;
          // ymin
          top_data_[idx++] = (center_y - box_height / 2.) / image_height_;
          // xmax
          top_data_[idx++] = (center_x + box_width / 2.) / image_width_;
          // ymax
          top_data_[idx++] = (center_y + box_height / 2.) / image_height_;

          if (max_sizes_.size() > 0) {
            CHECK_EQ(min_sizes_.size(), max_sizes_.size());
            int max_size_ = max_sizes_[s];
            // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrt(min_size_ * max_size_);
            // xmin
            top_data_[idx++] = (center_x - box_width / 2.) / image_width_;
            // ymin
            top_data_[idx++] = (center_y - box_height / 2.) / image_height_;
            // xmax
            top_data_[idx++] = (center_x + box_width / 2.) / image_width_;
            // ymax
            top_data_[idx++] = (center_y + box_height / 2.) / image_height_;
          }

          // rest of priors
          for (auto r = 0u; r < aspect_ratios_.size(); ++r) {
            float ar = aspect_ratios_[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size_ * sqrt(ar);
            box_height = min_size_ / sqrt(ar);
            // xmin
            top_data_[idx++] = (center_x - box_width / 2.) / image_width_;
            // ymin
            top_data_[idx++] = (center_y - box_height / 2.) / image_height_;
            // xmax
            top_data_[idx++] = (center_x + box_width / 2.) / image_width_;
            // ymax
            top_data_[idx++] = (center_y + box_height / 2.) / image_height_;
          }
        }
      }
    }
    CHECK_EQ(idx, dim);
    // clip the prior's coordidate such that it is within [0, 1]
    if (clip_) {
      for (auto d = 0u; d < dim; ++d) {
        top_data_[d] = std::min(std::max(top_data_[d], 0.0f), 1.0f);
      }
    }
    // set the variance.
    if (variance_.size() == 1) {
      caffe_set<float>(
          dim, variance_[0],
          &top_data_[idx] /* abug in origin prior_box_layer.cpp? */);
      idx = idx + layer_height_ * layer_width_ * num_priors_;
    } else {
      for (int h = 0; h < layer_height_; ++h) {
        for (int w = 0; w < layer_width_; ++w) {
          for (auto i = 0u; i < num_priors_; ++i) {
            for (int j = 0; j < 4; ++j) {
              top_data_[idx++] = variance_[j];
            }
          }
        }
      }
    }
    CHECK_EQ(idx, dim * 2);
  }

  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    // TODO: optimization
    CHECK_EQ(output.mem_size, top_data_.size() * sizeof(top_data_[0]));
    memcpy(&output.data[0], &top_data_[0], output.mem_size);
    return 0;
  }

 private:
  std::vector<float> min_sizes_;
  bool flip_;
  bool clip_;
  std::vector<float> aspect_ratios_;
  size_t num_priors_;
  std::vector<float> max_sizes_;
  std::vector<float> variance_;
  int image_height_;
  int image_width_;
  float step_h_;
  float step_w_;
  int layer_height_;
  int layer_width_;
  float offset_;
  std::vector<float> top_data_;
};

DEF_XIR_OP_IMP(MyOpImp)
