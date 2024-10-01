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
enum class NONLINEAR { NONE, RELU, PRELU, LEAKYRELU, RELU6 };

static enum NONLINEAR get_nonlinear(const std::string& nonlinear_type_str) {
  auto nonlinear_type_ = NONLINEAR::NONE;
  if (nonlinear_type_str == "NONE" || nonlinear_type_str == "") {
    nonlinear_type_ = NONLINEAR::NONE;
  } else if (nonlinear_type_str == "RELU") {
    nonlinear_type_ = NONLINEAR::RELU;
  } else if (nonlinear_type_str == "PRELU") {
    nonlinear_type_ = NONLINEAR::PRELU;
  } else if (nonlinear_type_str == "LEAKYRELU") {
    nonlinear_type_ = NONLINEAR::LEAKYRELU;
  } else if (nonlinear_type_str == "RELU6") {
    nonlinear_type_ = NONLINEAR::RELU6;
  } else {
    LOG(FATAL) << "not supported: " << nonlinear_type_str;
  }
  return nonlinear_type_;
}
// https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
// tf.nn.depthwise_conv2d(
//     input, filter, strides, padding, data_format=None, dilations=None,
//     name=None
// )

// Given a 4D input tensor ('NHWC' or 'NCHW' data formats) and a filter tensor
// of shape [filter_height, filter_width, in_channels, channel_multiplier]
// containing in_channels convolutional filters of depth 1, depthwise_conv2d
// applies a different filter to each input channel (expanding from 1 channel to
// channel_multiplier channels for each), then concatenates the results
// together. The output has in_channels * channel_multiplier channels.

// In detail, with the default NHWC format,

// output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}
//      filter[di, dj, k, q] * input[b, strides[1] * i + rate[0] * di,
//                                      strides[2] * j + rate[1] * dj, k]
//
struct MyOp : public vart::experimental::OpImpBase {
  MyOp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto stride = op->get_attr<std::vector<int>>("stride");
    CHECK_EQ(stride.size(), 2u);
    stride_h = stride[0];
    stride_w = stride[1];
    auto pad = op->get_attr<std::vector<int>>("pad");
    CHECK_EQ(pad.size(), 4u);
    pad_u = pad[0];
    pad_h = pad[1];
    pad_w = pad[2];
    pad_d = pad[3];
    auto weight_ops = op->get_input_ops("weights");
    CHECK_EQ(weight_ops.size(), 1u);
    auto weight_op = weight_ops.front();
    auto weight_shape = weight_op->get_output_tensor()->get_shape();
    CHECK_EQ(weight_shape.size(), 4u);
    channel_multiplier = weight_shape[0];
    filter_height = weight_shape[1];
    filter_width = weight_shape[2];
    pad_u = pad[0];
    pad_prime_h =
        filter_height - pad_h -
        1;  // see
            // http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html
    pad_prime_w = filter_width - pad_w - 1;
    pad_prime_d = pad_prime_u = pad_prime_w;  // TODO
    CHECK_EQ(pad_prime_w, pad_prime_h) << "TODO: not supported yet";
    in_channels = weight_shape[3];
    auto output_shape = op->get_output_tensor()->get_shape();
    oh = output_shape[1];
    ow = output_shape[2];
    auto input_ops = op->get_input_ops("input");
    CHECK_EQ(input_ops.size(), 1u);
    auto input_op = input_ops[0];
    auto input_shape = input_op->get_output_tensor()->get_shape();
    ih = input_shape[1];
    iw = input_shape[2];
    fix_pos_weight = weight_op->get_output_tensor()->get_attr<int>("fix_point");
    fix_pos_bias = 0;
    auto bias_ops = op->get_input_ops("bias");
    CHECK_LE(bias_ops.size(), 1u);
    if (bias_ops.empty()) {
      fix_pos_bias = 0;
    } else {
      fix_pos_bias =
          bias_ops[0]->get_output_tensor()->get_attr<int>("fix_point");
    }
    fix_pos_input = input_op->get_output_tensor()->get_attr<int>("fix_point");
    fix_pos_output = op->get_output_tensor()->get_attr<int>("fix_point");
    // LOG(INFO) << "fix_pos_weight " << fix_pos_weight << " "  //
    //           << "fix_pos_bias " << fix_pos_bias << " "      //
    //           << "fix_pos_input " << fix_pos_input << " "    //
    //           << "fix_pos_output " << fix_pos_output << " "  //
    //     ;
    shift_bias = (fix_pos_weight + fix_pos_input - fix_pos_bias);
    shift_cut = (fix_pos_weight + fix_pos_input - fix_pos_output);
    shift_cut_scale = std::pow(2.0, shift_cut + 1);
    nonlinear = get_nonlinear(op->get_attr<string>("nonlinear"));
    prelu = 0.0;  // TODO: read parameter
  };
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input,
                vart::simple_tensor_buffer_t<int8_t> weight,
                std::unique_ptr<vart::simple_tensor_buffer_t<int8_t>> bias) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape.size(), output_shape.size());
    CHECK_EQ(input_shape.size(), 4u);
    auto batch = input_shape[0];
    for (auto n = 0; n < batch; ++n) {
      auto input_offset = n * input_shape[1] * input_shape[2] * input_shape[3];
      auto output_offset =
          n * output_shape[1] * output_shape[2] * output_shape[3];
      calculate_single_batch(&output.data[output_offset],
                             &input.data[input_offset], weight.data,
                             bias != nullptr ? bias->data : nullptr);
    }
    return 0;
  }
  void calculate_single_batch(int8_t* output, int8_t* input, int8_t* weight,
                              int8_t* bias) {
    int c = 0;
    for (auto i = 0; i < oh; ++i) {
      for (auto j = 0; j < ow; ++j) {
        for (auto k = 0; k < in_channels; k++) {
          for (auto q = 0; q < channel_multiplier; q++) {
            int x = filter(input, weight, i, j, k, q);
            if (bias) {
              x = x + (bias[k] << shift_bias);
            }
            x = x * 2;  // see above fix_pos_cut + 1
            auto y = fix(nonlinear_fun(((float)x) / shift_cut_scale));
            output[c] = y;
            c = c + 1;
          }
        }
      }
    }
  }
  int filter(int8_t* input, int8_t* filter, int i, int j, int k, int q) {
    int ret = 0;
    // filter : (channel_multiplier, filter_height, filter_width, in_channels)
    for (int di = 0; di < filter_height; di++) {
      auto in_i = get_index(i + di, pad_prime_h, stride_h, ih);
      if (in_i < 0) {
        continue;
      }
      for (int dj = 0; dj < filter_width; dj++) {
        auto in_j = get_index(j + dj, pad_prime_w, stride_w, iw);
        if (in_j < 0) {
          continue;
        }
        int w = filter[q * filter_height * filter_width * in_channels +  //
                       di * filter_width * in_channels +                 //
                       dj * in_channels +                                //
                       k];
        int in = input[in_i * iw * in_channels +  //
                       in_j * in_channels +       //
                       k];
        ret = ret + w * in;
      }
    }
    return ret;
  }
  int get_index(int i, int p, int s, int max) {
    // p for padding, s for stride, i is the index,
    // try to map it to index of origin input data;
    //
    // max
    auto ret = -1;
    if (i >= p) {
      if ((i - p) % s == 0) {
        ret = (i - p) / s;
        if (ret >= max) {
          ret = -1;
        }
      }
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

  float nonlinear_fun(float x) {
    auto ret = x;
    switch (nonlinear) {
      case NONLINEAR::RELU:
        if (ret < 0) {
          ret = 0;
        }
        break;
      case NONLINEAR::PRELU:
        LOG(FATAL) << "read parameter from tensor:";
        if (ret < 0) {
          ret = ret * prelu;
        }
        break;
      case NONLINEAR::LEAKYRELU:
        if (ret < 0) {
          ret = ret * 0.01;
        }
        break;
      case NONLINEAR::RELU6:
        if (ret < 0) {
          ret = 0;
        }
        if (ret >= 6.0f) {
          ret = 6.0f;
        }
        break;
      case NONLINEAR::NONE:
      default:
        break;
    }
    return ret;
  }

 private:
  int stride_h;
  int stride_w;
  int pad_u;
  int pad_d;
  int pad_h;
  int pad_w;
  int pad_prime_u;
  int pad_prime_d;
  int pad_prime_h;
  int pad_prime_w;
  int channel_multiplier;
  int filter_height;
  int filter_width;
  int in_channels;
  int oh;
  int ow;
  int ih;
  int iw;
  int fix_pos_weight;
  int fix_pos_input;
  int fix_pos_bias;
  int fix_pos_output;
  int shift_cut;
  float shift_cut_scale;
  int shift_bias;
  enum NONLINEAR nonlinear;
  float prelu;
};  // namespace
}  // namespace

DEF_XIR_OP_IMP(MyOp)
