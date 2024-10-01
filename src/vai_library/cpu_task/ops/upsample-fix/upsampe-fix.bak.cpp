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
#include <iostream>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

using namespace std;

namespace {
enum MODE { BILINEAR };
struct MyOp : public vart::experimental::OpImpBase {
  MyOp(xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto input_op = op->get_input_op("input", 0);
    auto input_shape = input_op->get_output_tensor()->get_shape();
    auto output_shape = op->get_output_tensor()->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    CHECK_EQ(input_shape.size(), output_shape.size());
    CHECK_EQ(input_shape[0], output_shape[0]);
    batch_ = input_shape[0];
    CHECK_EQ(input_shape[3], 3);
    channel_ = input_shape[3];
    CHECK_EQ(input_shape[3], output_shape[3]);
    input_h_ = input_shape[1];
    input_w_ = input_shape[2];
    output_h_ = output_shape[1];
    output_w_ = output_shape[2];
    interpolation_ = BILINEAR;
    auto mode = op->get_attr<std::string>("mode");
    if (mode == "BILINEAR") {
      interpolation_ = BILINEAR;
    } else {
      LOG(FATAL) << "unkonwn mode = " << mode;
    }
    half_pixel_centers_ = op->get_attr<bool>("half_pixel_centers");
    CHECK(half_pixel_centers_) << "TODO";
    CHECK_EQ(output_w_ % input_w_, 0) << "only support upsample by integer.";
    CHECK_EQ(output_h_ % input_h_, 0) << "only support upsample by integer.";
    N_h_ = output_h_ / input_h_;
    N_w_ = output_w_ / input_w_;
    CHECK_EQ(N_h_ % 2, 0);
    CHECK_EQ(N_w_ % 2, 0);
    auto input_fix_pos =
        input_op->get_output_tensor()->get_attr<int>("fix_point");
    auto output_fix_pos = op->get_output_tensor()->get_attr<int>("fix_point");
    shift_fix_pos_ = output_fix_pos - input_fix_pos;
  };
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input) {
    for (auto b = 0; b < batch_; b++) {
      calculate_batch(&output.data[b * output_h_ * output_w_ * channel_],
                      &input.data[b * input_h_ * input_w_ * channel_]);
    }
    return 0;
  }
  int calculate_batch(int8_t* output, int8_t* input) {
    int ret = 0;
    if (half_pixel_centers_) {
      ret = calculate_batch_half_pixel_centers(output, input);
    } else {
      // TODO
    }
    return ret;
  }
  int calculate_batch_half_pixel_centers(int8_t* output, int8_t* input) {
    auto p = output;
    for (auto h = 0; h < output_h_; ++h) {
      auto weight_h = calculate_weight(input_h_, N_h_, h);
      const auto& input_top_index = std::get<0>(weight_h);
      const auto& input_bottom_index = std::get<0>(weight_h);
      const auto& distance_to_top = std::get<0>(weight_h);
      const auto& distance_to_bottom = std::get<0>(weight_h);
      for (auto w = 0; w < output_w_; ++w) {
        auto weight_w = calculate_weight(input_w, N_w_, w);
        const auto& input_left_index = std::get<0>(weight_w);
        const auto& input_right_index = std::get<0>(weight_w);
        const auto& distance_to_left = std::get<0>(weight_w);
        const auto& distance_to_right = std::get<0>(weight_w);
        for (auto c = 0; c < channel_; ++c) {
          const auto q_top_left =
              get_input(input, input_top_index, input_left_index, c);
          const auto q_top_right =
              get_input(input, input_top_index, input_right_index, c);
          const auto q_bottom_left =
              get_input(input, input_bottom_index, input_left_index, c);
          const auto q_bottom_right =
              get_input(input, input_bottom_index, input_right_index, c);
          *p++ = bilinear_interplation(distance_to_top, distance_to_bottom,
                                       distance_to_left, distance_to_right,
                                       q_top_left, q_top_right, q_bottom_left,
                                       q_bottom_right);
        }
      }
    }
  }

  inline int8_t get_input(int8_t* p, int h, int w, int c) {
    return p[h * input_w_ * channel_ + w * channel_ + c];
  }

  inline int8_t get_output(int8_t* p, int h, int w, int c) {
    return p[h * output_w_ * channel_ + w * channel_ + c];
  }

  int8_t bilinear_interplation(int distance_to_top, int distance_to_bottom,
                               int distance_to_left, int distance_to_right,
                               int q_top_left, int q_top_right,
                               int q_bottom_left, int q_bottom_right) {
    const auto w_top_left = distance_to_right * distance_to_bottom;
    const auto w_top_right = distance_to_left * distance_to_bottom;
    const auto w_bottom_left = distance_to_top * distance_to_right;
    const auto w_bottom_right = distance_to_top * distance_to_left;
    const auto value = w_top_left * q_top_left + w_top_right * q_top_right +
                       w_bottom_left * q_bottom_left +
                       w_bottom_right * q_bottom_right;
    return value / (N_h_ * N_w_ * 4);
  }
  // L, the lenght of origin sequence
  // N, upsample by N times, the lenth of new sequence is N*L
  // to center the pexil in between
  // in [0, 2NL-1] coordinate:
  //   for x_in in [0, L-1]
  //     x_in => 2*N * x_in + N, normalized to [0, 2NL-1]
  //   for x_out in [0, N*L-1]
  //     x_out => 2 * x_out + 1 , normalized to [0, 2NL-1]
  //
  std::tuple<int, int, int, int> calculate_weight(int L, int N, int x_out) {
    const auto _2N = 2 * N;
    const auto _2NL = _2N * L;
    const auto x = 2 * x_out + 1;  // coordinate normalized to [0, _2NL-1]
    // according to C99, "truncation toward zero" , x_in will always
    // larger than zero.
    const auto x_in = (x - N) / _2N;
    assert(x_in >= 0);  // coordinate normalized to [0, _2NL-1]
    const auto x_in_plus_1 = x_in == L - 1 ? L - 1 : (x + N) / _2N;
    const auto x_lower = _2N * x_in + N;
    const auto x_upper = _2N * x_in_plus_1 + N;
    const auto distance_to_left = x - x_lower;
    const auto distance_to_right = x_upper - x;
    return std::make_tuple(x_in, x_in_plus_1, distance_to_left,
                           distance_to_right);  // x_in == 0 ? 0 : x_in - 1;
  }

  int batch_;
  int input_w_;
  int input_h_;
  int output_w_;
  int output_h_;
  int channel_;
  MODE interpolation_;
  bool half_pixel_centers_;
  int N_h_;
  int N_w_;
  std::vector<int> w_h_;
  std::vector<int> w_w_;
  int shift_fix_pos_;
};  // namespace

}  // namespace

DEF_XIR_OP_IMP(DEF_XIR_OP_IMP)
