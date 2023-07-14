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

#include <iostream>
#include <vitis/ai/dim_calc.hpp>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
namespace {

struct Pad_t {
  int l;  // left
  int t;  // top
  int r;  // right
  int b;  // bottom
};

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    mode_ = op->get_attr<std::string>("mode");
    auto paddings = op->get_attr<std::vector<int32_t>>("paddings");
    CHECK_EQ(paddings.size(), 8) << "only support 4d pad current;" << std::endl;
    pad_ = Pad_t{paddings[4], paddings[2], paddings[5], paddings[3]};
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape[0], output_shape[0]);
    CHECK_EQ(input_shape[1] + pad_.t + pad_.b, output_shape[1]);
    CHECK_EQ(input_shape[2] + pad_.l + pad_.r, output_shape[2]);
    CHECK_EQ(input_shape[3], output_shape[3]);

    auto src_row_size = input_shape[2] * input_shape[3];
    auto dst_row_size = output_shape[2] * output_shape[3];

    auto src_batch_size = input_shape[1] * src_row_size;
    auto dst_batch_size = output_shape[1] * dst_row_size;

    if (pad_.l == 0 && pad_.t == 0 && pad_.r == 0 && pad_.b == 0) {
      std::copy_n(input.data, src_batch_size * input_shape[0], output.data);
      return 0;
    }

    if (mode_ == "CONSTANT") {
      auto pad_value = 0;
      for (auto i = 0; i < input_shape[0]; i++) {
        auto src = input.data + i * src_batch_size;
        auto dst = output.data + i * dst_batch_size;
        // pad top and bottom
        if (pad_.t > 0) {
          std::fill_n(dst, pad_.t * dst_row_size, pad_value);
        }
        if (pad_.b > 0) {
          std::fill_n(dst + (output_shape[1] - pad_.b) * dst_row_size,
                      pad_.b * dst_row_size, pad_value);
        }
        // pad left and right
        if (pad_.l > 0) {
          for (auto h = pad_.t; h < output_shape[1] - pad_.b; h++) {
            auto offset = h * dst_row_size;
            std::fill_n(dst + offset, pad_.l * output_shape[3], pad_value);
          }
        }
        if (pad_.r > 0) {
          for (auto h = pad_.t; h < output_shape[1] - pad_.b; h++) {
            auto offset =
                h * dst_row_size + (output_shape[2] - pad_.r) * output_shape[3];
            std::fill_n(dst + offset, pad_.r * output_shape[3], pad_value);
          }
        }
        // copy source data
        for (auto h = pad_.t; h < output_shape[1] - pad_.b; h++) {
          auto src_offset = (h - pad_.t) * src_row_size;
          auto dst_offset = h * dst_row_size + pad_.l * output_shape[3];
          std::copy_n(src + src_offset, src_row_size, dst + dst_offset);
        }
      }
    } else if (mode_ == "SYMMETRIC") {
      for (int n = 0; n < output_shape[0]; n++) {
        for (int h = 0; h < output_shape[1]; h++) {
          for (int w = 0; w < output_shape[2]; w++) {
            for (int c = 0; c < output_shape[3]; c++) {
              int h_idx = h - pad_.t;
              int w_idx = w - pad_.l;
              if (h < pad_.t)
                h_idx = pad_.t - 1 - std::min(h, pad_.t - 1);
              else if (h >= input_shape[1] + pad_.t)
                h_idx =
                    input_shape[1] - 1 -
                    std::min(h - pad_.t - input_shape[1], input_shape[1] - 1);
              if (w < pad_.l)
                w_idx = pad_.l - 1 - std::min(w, pad_.l - 1);
              else if (w >= input_shape[2] + pad_.l)
                w_idx =
                    input_shape[2] - 1 -
                    std::min(w - pad_.l - input_shape[2], input_shape[2] - 1);
              auto out_idx = ((n * output_shape[1] + h) * output_shape[2] + w) *
                                 output_shape[3] +
                             c;
              auto in_idx =
                  ((n * input_shape[1] + h_idx) * input_shape[2] + w_idx) *
                      input_shape[3] +
                  c;
              output.data[out_idx] = input.data[in_idx];
            }
          }
        }
      }
    }
    return 0;
  }

 private:
  std::string mode_;
  Pad_t pad_;
};

}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
