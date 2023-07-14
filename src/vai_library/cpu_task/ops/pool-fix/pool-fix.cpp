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
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
using namespace std;

namespace {

struct PairHash {
 public:
  template <typename T, typename U>
  size_t operator()(const std::pair<T, U>& x) const {
    auto h1 = std::hash<T>{}(x.first);
    auto h2 = std::hash<U>{}(x.second);
    return h1 ^ h2;
  }
};

struct ApproximateParam {
  ApproximateParam(int t_kh, int t_kw, int t_N, int t_M, float t_alpha)
      : kh(t_kh), kw(t_kw), N(t_N), M(t_M), alpha(t_alpha) {
    scale = (float)N / std::pow(2, M);
  }
  // formula: N/2^M = alpha * 1 / (kh * kw)
  int kh;  // pool kernel height
  int kw;  // pool kernel width
  int N;
  int M;
  float alpha;
  float scale;
};

std::pair<int32_t, int32_t> get_avgpool_dpu_factors(
    const std::vector<std::int32_t>& kernels) {
  auto rec = kernels[0] * kernels[1];
  auto multi_factor = 0;
  auto shift_factor = 0;
  auto diff = 1.f;
  if (kernels[0] == 3 && kernels[1] == 3) {
    multi_factor = 7;
    shift_factor = 6;
  } else if (kernels[0] == 5 && kernels[1] == 5) {
    multi_factor = 10;
    shift_factor = 8;
  } else if (kernels[0] == 6 && kernels[1] == 6) {
    multi_factor = 7;
    shift_factor = 8;
  } else if (kernels[0] == 7 && kernels[1] == 7) {
    multi_factor = 21;
    shift_factor = 10;
  } else if (kernels[0] == 14 && kernels[1] == 14) {
    multi_factor = 21;
    shift_factor = 12;
  } else {
    auto max_factor = std::ceil(std::log2(rec * 128));
    for (auto shift_factor_ = 0; shift_factor_ < max_factor; shift_factor_++) {
      auto factor = std::round(std::exp2(shift_factor_) / rec);
      auto diff_ = std::abs(factor / std::exp2(shift_factor_) - 1.f / rec);
      if (diff_ < diff) {
        multi_factor = factor;
        diff = diff_;
        shift_factor = shift_factor_;
      }
    }
  }
  return {multi_factor, shift_factor};
}

float get_avgpool_dpu_coefficient(const std::vector<std::int32_t>& kernels) {
  auto factors = get_avgpool_dpu_factors(kernels);
  return float(factors.first) / std::exp2(factors.second);
}

enum class POOLTYPE { MAX, AVG };
struct PoolFix_OpImp : public vart::experimental::OpImpBase {
  PoolFix_OpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto kernel = op->get_attr<std::vector<int32_t>>("kernel");
    CHECK_EQ(kernel.size(), 2u)
        << "pool-fix kernel must be :{kernel_width, kernel_height}";
    kernel_w = kernel[0];
    kernel_h = kernel[1];

    auto stride = op->get_attr<std::vector<int32_t>>("stride");
    CHECK_EQ(stride.size(), 2u)
        << "pool-fix stride must be :{stride_width, stride_height}";
    stride_w = stride[0];
    stride_h = stride[1];

    auto pad = std::vector<int32_t>(4, 0);
    if (op->has_attr("pad")) {
      pad = op->get_attr<std::vector<int32_t>>("pad");
    }
    CHECK_EQ(pad.size(), 4)
        << "pool-fix pad must be : {left, right, top, bottom}";
    pad_left = pad[0];
    pad_right = pad[1];
    pad_top = pad[2];
    pad_bottom = pad[3];

    auto type_str = op->get_attr<std::string>("type");
    if (type_str == "MAX") {
      pool_type_ = POOLTYPE::MAX;
    } else if (type_str == "AVG") {
      pool_type_ = POOLTYPE::AVG;
    } else {
      CHECK(false) << "Unknown pool-fix type: " << type_str;
    }

    auto input_ops = op->get_input_ops("input");
    CHECK_EQ(input_ops.size(), 1u);
    auto input_op = input_ops[0];
    auto fix_pos_input =
        input_op->get_output_tensor()->get_attr<int>("fix_point");
    auto fix_pos_output = op->get_output_tensor()->get_attr<int>("fix_point");
    shift_cut = (fix_pos_input - fix_pos_output);
    shift_cut_scale = std::pow(2.0, shift_cut);

    auto output_shape = op->get_output_tensor()->get_shape();
    CHECK_EQ(output_shape.size(), 4u);
    oh = output_shape[1];
    ow = output_shape[2];
    oc = output_shape[3];
    auto input_shape = input_op->get_output_tensor()->get_shape();
    ih = input_shape[1];
    iw = input_shape[2];
    scale_ = get_avgpool_dpu_coefficient({kernel_h, kernel_w});
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> result,
                vart::simple_tensor_buffer_t<int8_t> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    // input tensor shape is [batch, in_height, in_width, in_channels]
    CHECK_EQ(input_shape.size(), 4u);
    CHECK_EQ(input_shape.size(), output_shape.size());
    auto batch = input_shape[0];
    for (auto n = 0; n < batch; ++n) {
      auto input_offset = n * input_shape[1] * input_shape[2] * input_shape[3];
      auto output_offset =
          n * output_shape[1] * output_shape[2] * output_shape[3];
      calculate_single_batch(&result.data[output_offset],
                             &input.data[input_offset]);
    }
    return 0;
  }
  void calculate_single_batch(int8_t* output, int8_t* input) {
    int c = 0;
    for (auto i = 0; i < oh; ++i) {
      for (auto j = 0; j < ow; ++j) {
        for (auto k = 0; k < oc; k++) {
          auto x = filter(input, i, j, k);
          auto y = fix(x / shift_cut_scale);
          output[c] = y;
          c = c + 1;
        }
      }
    }
  }
  float filter(int8_t* input, int i, int j, int k) {
    float ret = 0.0f;
    switch (pool_type_) {
      case POOLTYPE::AVG:
        ret = 0.0f;
        break;
      case POOLTYPE::MAX:
        ret = std::numeric_limits<float>::min();
        break;
      default:
        break;
    }
    for (int di = 0; di < kernel_h; di++) {
      for (int dj = 0; dj < kernel_w; dj++) {
        auto input_h_idx = ((i * stride_h - pad_top) + di);
        auto input_w_idx = ((j * stride_w - pad_left) + dj);
        if (input_w_idx < 0) {
          continue;
        }
        if (input_h_idx < 0) {
          continue;
        }
        if (input_h_idx >= ih) {
          continue;
        }
        if (input_w_idx >= iw) {
          continue;
        }
        int in = input[input_h_idx * iw * oc +  //
                       input_w_idx * oc +       //
                       k];

        LOG_IF(INFO, i == 0 && j == 0 && k == 745 && false)
            << "in " << in << " "                     //
            << "di " << di << " "                     //
            << "dj " << dj << " "                     //
            << "input_h_indx " << input_h_idx << " "  //
            << "input_w_indx " << input_w_idx << " "  //
            << endl;

        switch (pool_type_) {
          case POOLTYPE::AVG:
            ret = ret + in;
            break;
          case POOLTYPE::MAX:
            ret = std::max(ret, (float)in);
            break;
          default:
            break;
        }
      }
    }
    LOG_IF(INFO, i == 0 && j == 0 && k == 745 && false) << "ret = " << ret;
    if (pool_type_ == POOLTYPE::AVG) {
      ret = ret * scale_;
    }

    return ret;
  }

  int fix(float data) {
    auto data_max = 127.0;
    auto data_min = -128.0;
    if (data > data_max) {
      data = data_max;
    } else if (data < data_min) {
      data = data_min;
    } else if (data < 0 && (data - floor(data)) == 0.5) {
      data = static_cast<float>(ceil(data));
    } else {
      data = static_cast<float>(round(data));
    }
    return (int)data;
  }

 private:
  int32_t kernel_w;
  int32_t kernel_h;
  int32_t stride_w;
  int32_t stride_h;
  int32_t pad_left;
  int32_t pad_right;
  int32_t pad_top;
  int32_t pad_bottom;
  enum POOLTYPE pool_type_;
  int shift_cut;
  float shift_cut_scale;
  int oh;
  int ow;
  int oc;
  int ih;
  int iw;
  float scale_;
};

}  // namespace

DEF_XIR_OP_IMP(PoolFix_OpImp)
