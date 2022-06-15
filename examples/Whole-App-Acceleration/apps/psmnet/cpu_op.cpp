/*
 * Copyright 2019 Xilinx Inc.
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
#include <glog/logging.h>
#include <cmath>
#include <numeric>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include "cpu_op.hpp"

DEF_ENV_PARAM(DEBUG_CPU_OPS, "0")

Resize::Resize(vitis::ai::library::OutputTensor& input_,
         vitis::ai::library::InputTensor& output_) {
  i_shape_ = {input_.batch, input_.height, input_.width, input_.channel};
  o_shape_ = {output_.batch, output_.height, output_.width, output_.channel};
  //batch_ = input_.batch;
  align_corners_ = false;
  half_pixel_centers_ = true;
  LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_OPS))
    << "the scales is "
    << tensor_scale(input_) << " vs " << tensor_scale(output_);
  fix_scale = tensor_scale(input_) * tensor_scale(output_);

  auto size = std::accumulate(
      std::begin(o_shape_), std::end(o_shape_), 1, std::multiplies<int>());
  LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_OPS))
     << "output size is: " << size;

  output_f_.resize(size);
  for (size_t i = 0; i < i_shape_[0]; ++i) {
    data_in_ptr_.push_back((int8_t*)input_.get_data(i));
    data_out_ptr_.push_back((int8_t*)output_.get_data(i));
  }
}

Resize::~Resize() {}

struct CachedInterpolation {
  int lower;
  int upper;
  float lerp;
};

static int8_t dpu_round(float num) {
  if (num - floor(num) == 0.5)
    return ceil(num);
  else
    return round(num);
}

void Resize::run() {
  auto cal_scale =
    [](std::int32_t in, std::int32_t out, bool align_corners) -> float {
    return (align_corners && out > 1) ? (in - 1) / static_cast<float>(out - 1)
                                      : in / static_cast<float>(out);
  };
  auto scaler =
    [](std::int32_t out, float scale, bool half_pixel_centers) -> float {
    return (half_pixel_centers)
             ? (static_cast<float>(out) + 0.5f) * scale - 0.5f
             : static_cast<float>(out) * scale;
  };
  auto compute_interpolation_weights = [&](const int out_size,
                                           const int in_size,
                                           const float scale,
                                           CachedInterpolation* interpolation) {
    interpolation[out_size].lower = 0;
    interpolation[out_size].upper = 0;
    for (int i = out_size - 1; i >= 0; --i) {
      const float in = scaler(i, scale, half_pixel_centers_);
      interpolation[i].lower =
        std::max(static_cast<int>(std::floor(in)), static_cast<int>(0));
      interpolation[i].upper = std::min(static_cast<int>(std::ceil(in)),
                                        static_cast<int>(in_size - 1));
      interpolation[i].lerp = in - interpolation[i].lower;
    }
  };

  std::vector<CachedInterpolation> xs(o_shape_[2] + 1);
  std::vector<CachedInterpolation> ys(o_shape_[1] + 1);
  auto h_scale = cal_scale(i_shape_[1], o_shape_[1], align_corners_);
  auto w_scale = cal_scale(i_shape_[2], o_shape_[2], align_corners_);
  compute_interpolation_weights(o_shape_[1], i_shape_[1], h_scale, ys.data());
  compute_interpolation_weights(o_shape_[2], i_shape_[2], w_scale, xs.data());
  for (uint i = 0; i < xs.size(); ++i) {
    xs[i].lower *= i_shape_[3];
    xs[i].upper *= i_shape_[3];
  }
  const size_t in_row_size = i_shape_[2] * i_shape_[3];
  //const size_t in_batch_num_values = i_shape_[1] * in_row_size;
  const size_t out_row_size = o_shape_[2] * i_shape_[3];
  const size_t out_batch_num_values = o_shape_[1] * out_row_size;

  auto compute_lerp = [&](const float top_left,
                          const float top_right,
                          const float bottom_left,
                          const float bottom_right,
                          const float x_lerp,
                          const float y_lerp) -> float {
    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    return top + (bottom - top) * y_lerp;
  };
  int idx_start = 0;
  for (size_t b = 0; b < o_shape_[0]; ++b) {
    int8_t* in_ptr = data_in_ptr_[b];
    for (size_t y = 0; y < o_shape_[1]; ++y) {
      const int8_t* ys_input_lower_ptr = in_ptr + ys[y].lower * in_row_size;
      const int8_t* ys_input_upper_ptr = in_ptr + ys[y].upper * in_row_size;
      const float ys_lerp = ys[y].lerp;
      for (size_t x = 0; x < o_shape_[2]; ++x) {
        auto xs_lower = xs[x].lower;
        auto xs_upper = xs[x].upper;
        auto xs_lerp = xs[x].lerp;
        for (size_t c = 0; c < i_shape_[3]; ++c) {
          const float top_left(ys_input_lower_ptr[xs_lower + c]);
          const float top_right(ys_input_lower_ptr[xs_upper + c]);
          const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
          const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
          output_f_[x * i_shape_[3] + c + idx_start] = compute_lerp(
            top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp);
        }
      }
      idx_start += out_row_size;
    }
    //in_ptr += in_batch_num_values;
  }
  for(size_t i = 0; i < output_f_.size(); ++i) {
    data_out_ptr_[i/out_batch_num_values][i%out_batch_num_values] =  dpu_round(output_f_[i]*fix_scale);
  }
}

// cpu softmax and sum part
CPUsfm::CPUsfm(vitis::ai::library::OutputTensor& input_) {
  i_shape_ = {input_.batch, input_.height, input_.width, input_.channel};
  auto size = input_.batch * input_.height * input_.width;
  outputs_.resize(size);
  LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_OPS))
     << "sft size is: " << size;
  fix_scale = tensor_scale(input_);
  for(size_t i = 0; i < i_shape_[0]; ++i)
    inputs_.push_back((int8_t*)input_.get_data(i));

  disp_.resize(i_shape_[3]);
  for(size_t i = 0; i < i_shape_[3]; ++i)
    disp_[i] = i;
}

CPUsfm::~CPUsfm() {}

void CPUsfm::run() {
  auto fea_size = i_shape_[1] * i_shape_[2];
  auto out_size = i_shape_[3] * fea_size;
  std::vector<float> output(out_size);
  for(size_t b = 0; b < i_shape_[0]; ++b) {
    vitis::ai::softmax(inputs_[b], fix_scale, i_shape_[3],
		       fea_size, output.data());
    for(size_t h = 0; h < i_shape_[1]; ++h) {
      for(size_t w = 0; w < i_shape_[2]; ++w) {
        float sum = 0.0;
        int pos = h*i_shape_[2] + w;
        for(size_t c = 0; c < i_shape_[3]; ++c) {
          sum += disp_[c] * output[pos*i_shape_[3]+c];
        }
        outputs_[b*fea_size + pos] = sum;
      }
    }
  }
}

float* CPUsfm::get_output() {
  return outputs_.data();
}

