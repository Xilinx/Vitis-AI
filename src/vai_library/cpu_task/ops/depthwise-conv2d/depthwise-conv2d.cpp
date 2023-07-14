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
    auto weight_ops = op->get_input_ops("weights");
    CHECK_EQ(weight_ops.size(), 1u);
    auto weight_op = weight_ops.front();
    auto weight_shape = weight_op->get_output_tensor()->get_shape();
    CHECK_EQ(weight_shape.size(), 4u);
    channel_multiplier = weight_shape[0];
    filter_height = weight_shape[1];
    filter_width = weight_shape[2];
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
    auto bias_ops = op->get_input_ops("bias");
    CHECK_LE(bias_ops.size(), 1u);
    auto pad = op->get_attr<std::vector<int>>("pad");
    CHECK_EQ(pad.size(), 4u);
    pad_left = pad[0];
    pad_right = pad[1];
    pad_top = pad[2];
    pad_bottom = pad[3];
    auto stride = op->get_attr<std::vector<int>>("stride");
    CHECK_EQ(stride.size(), 2u);
    stride_h = stride[0];
    stride_w = stride[1];
  };
  int calculate(vart::simple_tensor_buffer_t<float> output,
                vart::simple_tensor_buffer_t<float> input,
                vart::simple_tensor_buffer_t<float> weight,
                std::unique_ptr<vart::simple_tensor_buffer_t<float>> bias) {
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
  void calculate_single_batch(float* output, float* input, float* weight,
                              float* bias) {
    int c = 0;
    for (auto i = 0; i < oh; ++i) {
      for (auto j = 0; j < ow; ++j) {
        for (auto k = 0; k < in_channels; k++) {
          for (auto q = 0; q < channel_multiplier; q++) {
            float x = filter(input, weight, i, j, k, q);
            if (bias) {
              x = x + bias[k];
            }
            output[c] = x;
            c = c + 1;
          }
        }
      }
    }
  }
  float filter(float* input, float* filter, int i, int j, int k, int q) {
    float ret = 0;
    // filter : (channel_multiplier, filter_height, filter_width, in_channels)
    for (int di = 0; di < filter_height; di++) {
      for (int dj = 0; dj < filter_width; dj++) {
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
        auto w = filter[q * filter_height * filter_width * in_channels +  //
                        di * filter_width * in_channels +                 //
                        dj * in_channels +                                //
                        k];
        auto in = input[input_h_idx * iw * in_channels +  //
                        input_w_idx * in_channels +       //
                        k];
        LOG_IF(INFO, false) << "w " << w << " "                       //
                            << "in " << in << " "                     //
                            << "di " << di << " "                     //
                            << "dj " << dj << " "                     //
                            << "input_h_indx " << input_h_idx << " "  //
                            << "input_w_indx " << input_w_idx << " "  //
                            << endl;
        ret = ret + w * in;
      }
    }
    return ret;
  }

 private:
  int channel_multiplier;
  int filter_height;
  int filter_width;
  int in_channels;
  int oh;
  int ow;
  int ih;
  int iw;
  int pad_left;
  int pad_right;
  int pad_top;
  int pad_bottom;
  int stride_h;
  int stride_w;
};
}  // namespace

DEF_XIR_OP_IMP(MyOp)
