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

#include <cmath>
#include <iostream>
#include <numeric>

#include "UniLog/UniLog.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph.hpp"
#include "xir/op/shape_inference.hpp"
#include "xir/tensor/tensor.hpp"
#include "xir/util/data_type.hpp"
#include "xir/util/internal_util.hpp"
#include "xir/util/tool_function.hpp"

namespace xir {
using namespace std;
void shape_infer_unsupported(xir::Op* cur) {
  UNI_LOG_FATAL(XIR_INVALID_ARG_OCCUR)
      << "\"" << cur->get_name() << "\" is a \"" << cur->get_type()
      << "\" op which is not supported now. So, it's shape can't be infered.";
}

void shape_infer_remain(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto out = cur->get_output_tensor();
  auto its = cur->get_input_tensors();
  auto out_shape = in->get_shape();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  for (auto it : its) {
    if (it->has_attr("shape_info")) {
      auto si = it->get_attr<std::vector<std::int32_t>>("shape_info");
      output_tensor->set_attr("shape_info", si);
    }
  }
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_broadcast(xir::Op* cur) {
  auto in_num = cur->get_input_num();
  auto input_tensors = cur->get_input_tensors();
  if (in_num == 1) {
    auto output_tensor_ori = cur->get_output_tensor();
    auto output_tensor_new = xir::Tensor::create(
        output_tensor_ori->get_name(), input_tensors[0]->get_shape(),
        output_tensor_ori->get_data_type());
    output_tensor_new->set_attrs(output_tensor_ori->get_attrs());
    cur->replace_output_tensor(std::move(output_tensor_new));
    return;
  }
  auto sbc_rlt = size_broadcast(input_tensors[0]->get_shape(),
                                input_tensors[1]->get_shape());
  UNI_LOG_CHECK(std::get<0>(sbc_rlt), XIR_INVALID_ARG_OCCUR)
      << cur->to_string() << "'s input size are not matching. One is "
      << xir::to_string(input_tensors[0]->get_shape()) << " and another is "
      << xir::to_string(input_tensors[1]->get_shape()) << ".";
  if (in_num > 2) {
    for (unsigned int i = 2; i < input_tensors.size(); i++) {
      auto out =
          size_broadcast(std::get<1>(sbc_rlt), input_tensors[i]->get_shape());
      UNI_LOG_CHECK(std::get<0>(out), XIR_INVALID_ARG_OCCUR)
          << cur->to_string() << "'s input size are not matching. One is "
          << xir::to_string(std::get<1>(sbc_rlt)) << " and another is "
          << xir::to_string(input_tensors[i]->get_shape()) << ".";
      sbc_rlt = out;
    }
  }
  auto output_tensor_ori = cur->get_output_tensor();
  auto output_tensor_new =
      xir::Tensor::create(output_tensor_ori->get_name(), std::get<1>(sbc_rlt),
                          output_tensor_ori->get_data_type());
  output_tensor_new->set_attrs(output_tensor_ori->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor_new));
}

std::int32_t cal_out_1d(std::vector<std::int32_t> in_shape,
                        std::vector<std::int32_t> padding,
                        std::vector<std::int32_t> kernel,
                        std::vector<std::int32_t> stride,
                        std::vector<std::int32_t> dilation,
                        std::string pad_mode) {
  std::int32_t ol = 0;
  if (pad_mode == "FLOOR") {
    ol = std::floor(1.0 *
                    (in_shape[1] + padding[0] + padding[1] -
                     ((kernel[0] - 1) * dilation[0] + 1) + stride[0]) /
                    stride[0]);
  } else if (pad_mode == "CEIL") {
    ol = std::ceil(1.0 *
                   (in_shape[1] + padding[0] + padding[1] -
                    ((kernel[0] - 1) * dilation[0] + 1) + stride[0]) /
                   stride[0]);
  } else if (pad_mode == "SAME") {
    ol = std::ceil(1.0 * (in_shape[1] + padding[0] + padding[1]) / stride[0]);
  } else if (pad_mode == "VALID") {
    ol = std::ceil(1.0 *
                   (in_shape[1] + padding[0] + padding[1] -
                    (kernel[0] - 1) * dilation[0]) /
                   stride[0]);
  } else {
    UNI_LOG_CHECK(pad_mode == "FLOOR" || pad_mode == "CEIL" ||
                      pad_mode == "SAME" || pad_mode == "VALID",
                  XIR_INVALID_ARG_OCCUR)
        << "pad_mode here is " << pad_mode
        << ", but it should be one of \"FLOOR\","
           "\"CEIL\", \"SAME\" or \"VALID\".";
  }
  return ol;
}

std::vector<std::int32_t> cal_out(std::vector<std::int32_t> in_shape,
                                  std::vector<std::int32_t> padding,
                                  std::vector<std::int32_t> kernel,
                                  std::vector<std::int32_t> stride,
                                  std::vector<std::int32_t> dilation,
                                  std::string pad_mode) {
  std::int32_t ow = 0, oh = 0;
  if (pad_mode == "FLOOR") {
    ow = std::floor(1.0 *
                    (in_shape[2] + padding[0] + padding[1] -
                     ((kernel[0] - 1) * dilation[0] + 1) + stride[0]) /
                    stride[0]);
    oh = std::floor(1.0 *
                    (in_shape[1] + padding[2] + padding[3] -
                     ((kernel[1] - 1) * dilation[1] + 1) + stride[1]) /
                    stride[1]);
  } else if (pad_mode == "CEIL") {
    ow = std::ceil(1.0 *
                   (in_shape[2] + padding[0] + padding[1] -
                    ((kernel[0] - 1) * dilation[0] + 1) + stride[0]) /
                   stride[0]);
    oh = std::ceil(1.0 *
                   (in_shape[1] + padding[2] + padding[3] -
                    ((kernel[1] - 1) * dilation[1] + 1) + stride[1]) /
                   stride[1]);
  } else if (pad_mode == "SAME") {
    ow = std::ceil(1.0 * (in_shape[2] + padding[0] + padding[1]) / stride[0]);
    oh = std::ceil(1.0 * (in_shape[1] + padding[2] + padding[3]) / stride[1]);
  } else if (pad_mode == "VALID") {
    ow = std::ceil(1.0 *
                   (in_shape[2] + padding[0] + padding[1] -
                    (kernel[0] - 1) * dilation[0]) /
                   stride[0]);
    oh = std::ceil(1.0 *
                   (in_shape[1] + padding[2] + padding[3] -
                    (kernel[1] - 1) * dilation[1]) /
                   stride[1]);
  } else {
    UNI_LOG_CHECK(pad_mode == "FLOOR" || pad_mode == "CEIL" ||
                      pad_mode == "SAME" || pad_mode == "VALID",
                  XIR_INVALID_ARG_OCCUR)
        << "pad_mode here is " << pad_mode
        << ", but it should be one of \"FLOOR\","
           "\"CEIL\", \"SAME\" or \"VALID\".";
  }
  return {oh, ow};
}

std::vector<std::int32_t> cal_out_3d(std::vector<std::int32_t> in_shape,
                                     std::vector<std::int32_t> padding,
                                     std::vector<std::int32_t> kernel,
                                     std::vector<std::int32_t> stride,
                                     std::vector<std::int32_t> dilation,
                                     std::string pad_mode) {
  std::int32_t ow = 0, oh = 0, od = 0;
  if (pad_mode == "FLOOR") {
    oh = std::floor(1.0 *
                    (in_shape[1] + padding[2] + padding[3] -
                     ((kernel[1] - 1) * dilation[1] + 1) + stride[1]) /
                    stride[1]);
    ow = std::floor(1.0 *
                    (in_shape[2] + padding[0] + padding[1] -
                     ((kernel[0] - 1) * dilation[0] + 1) + stride[0]) /
                    stride[0]);
    od = std::floor(1.0 *
                    (in_shape[3] + padding[4] + padding[5] -
                     ((kernel[2] - 1) * dilation[2] + 1) + stride[2]) /
                    stride[2]);

  } else if (pad_mode == "CEIL") {
    oh = std::ceil(1.0 *
                   (in_shape[1] + padding[2] + padding[3] -
                    ((kernel[1] - 1) * dilation[1] + 1) + stride[1]) /
                   stride[1]);
    ow = std::ceil(1.0 *
                   (in_shape[2] + padding[0] + padding[1] -
                    ((kernel[0] - 1) * dilation[0] + 1) + stride[0]) /
                   stride[0]);
    od = std::ceil(1.0 *
                   (in_shape[3] + padding[4] + padding[5] -
                    ((kernel[2] - 1) * dilation[2] + 1) + stride[2]) /
                   stride[2]);
  } else if (pad_mode == "SAME") {
    oh = std::ceil(1.0 * (in_shape[1] + padding[2] + padding[3]) / stride[1]);
    ow = std::ceil(1.0 * (in_shape[2] + padding[0] + padding[1]) / stride[0]);
    od = std::ceil(1.0 * (in_shape[3] + padding[4] + padding[5]) / stride[2]);
  } else if (pad_mode == "VALID") {
    oh = std::ceil(1.0 *
                   (in_shape[1] + padding[2] + padding[3] -
                    (kernel[1] - 1) * dilation[1]) /
                   stride[1]);
    ow = std::ceil(1.0 *
                   (in_shape[2] + padding[0] + padding[1] -
                    (kernel[0] - 1) * dilation[0]) /
                   stride[0]);
    od = std::ceil(1.0 *
                   (in_shape[3] + padding[4] + padding[5] -
                    (kernel[2] - 1) * dilation[2]) /
                   stride[2]);
  } else {
    UNI_LOG_CHECK(pad_mode == "FLOOR" || pad_mode == "CEIL" ||
                      pad_mode == "SAME" || pad_mode == "VALID",
                  XIR_INVALID_ARG_OCCUR)
        << "pad_mode here is " << pad_mode
        << ", but it should be one of \"FLOOR\","
           "\"CEIL\", \"SAME\" or \"VALID\".";
  }
  return {oh, ow, od};
}

template <typename Dtype>
std::vector<Dtype> read_data_in_attr(xir::Op* op) {
  auto data = op->get_attr<std::vector<char>>("data");
  auto output_tensor = op->get_output_tensor();
  auto data_type = output_tensor->get_data_type().type;
  auto data_num = output_tensor->get_element_num();
  auto bit_width = output_tensor->get_data_type().bit_width;

  std::vector<Dtype> data_vec;
  if (data_type == xir::DataType::INT && bit_width == 8) {
    int8_t* s = reinterpret_cast<int8_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::UINT && bit_width == 8) {
    uint8_t* s = reinterpret_cast<uint8_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::INT && bit_width == 16) {
    int16_t* s = reinterpret_cast<int16_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::UINT && bit_width == 16) {
    uint16_t* s = reinterpret_cast<uint16_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::INT && bit_width == 32) {
    int32_t* s = reinterpret_cast<int32_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::UINT && bit_width == 32) {
    uint32_t* s = reinterpret_cast<uint32_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::INT && bit_width == 64) {
    int64_t* s = reinterpret_cast<int64_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::UINT && bit_width == 64) {
    uint64_t* s = reinterpret_cast<uint64_t*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::FLOAT && bit_width == 32) {
    float* s = reinterpret_cast<float*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else if (data_type == xir::DataType::FLOAT && bit_width == 64) {
    double* s = reinterpret_cast<double*>(data.data());
    for (auto i = 0; i < data_num; i++)
      data_vec.push_back(static_cast<Dtype>(s[i]));
  } else {
    UNI_LOG_FATAL(XIR_INVALID_ARG_OCCUR) << "do not support this data type.";
  }
  return data_vec;
}

void shape_infer_conv1d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 3, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of weights here is " << w_shape.size()
      << ", but the number of dimension should be 3.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  std::vector<std::int32_t> padding = {0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 2, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 2.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  auto ol = cal_out_1d(in_shape, padding, kernel, stride, dilation, pad_mode);
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), ol,
                                             w_shape[0]};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_conv2d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 4, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of weights here is " << w_shape.size()
      << ", but the number of dimension should be 4.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::int32_t group = 1;
  if (attrs->has_attr("group")) {
    group = attrs->get_attr<std::int32_t>("group");
  }
  UNI_LOG_CHECK(w_shape.back() == in_shape.back() / group,
                XIR_INVALID_ARG_OCCUR)
      << "The number of channel of weights is " << w_shape.back()
      << ", but the number of channel of input / group is "
      << in_shape.back() / group;
  UNI_LOG_CHECK(in_shape.back() % group == 0, XIR_INVALID_ARG_OCCUR)
      << "The number of channel of input should be divisible by group.";
  UNI_LOG_CHECK(w_shape.front() % group == 0, XIR_INVALID_ARG_OCCUR)
      << "The number of weight kernel should be divisible by group.";
  auto oh_ow = cal_out(in_shape, padding, kernel, stride, dilation, pad_mode);
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), oh_ow[0],
                                             oh_ow[1], w_shape[0]};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_depthwise_conv2d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 4, XIR_INVALID_ARG_OCCUR)
      << "The number of the dimension of weights here is " << w_shape.size()
      << ", but the number of the dimension should be 4.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  auto oh_ow = cal_out(in_shape, padding, kernel, stride, dilation, pad_mode);
  auto out_channel = w_shape[0] * w_shape[3];
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), oh_ow[0],
                                             oh_ow[1], out_channel};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_conv2d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  int ow = 0, oh = 0, oc = 0;
  if (pad_mode == "FLOOR") {
    // referred to
    // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp
    ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
         padding[0] - padding[1];
    oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
         padding[2] - padding[3];
  } else if (pad_mode == "CEIL") {
    // SAME with case 0
    // see
    // https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_transpose_op.cc
    ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
         padding[0] - padding[1];
    oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
         padding[2] - padding[3];
  } else if (pad_mode == "SAME") {
    // SAME
    // see
    // https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
    ow = in_shape[2] * stride[0];
    oh = in_shape[1] * stride[1];
  } else if (pad_mode == "VALID") {
    // VALID
    ow = (in_shape[2] - 1) * stride[0] + kernel[0];
    oh = (in_shape[1] - 1) * stride[1] + kernel[1];
  } else {
    UNI_LOG_CHECK(pad_mode == "FLOOR" || pad_mode == "CEIL" ||
                      pad_mode == "SAME" || pad_mode == "VALID",
                  XIR_INVALID_ARG_OCCUR)
        << "pad_mode here is " << pad_mode
        << ", but it should be one of \"FLOOR\","
           "\"CEIL\", \"SAME\" or \"VALID\".";
  }
  oc = w_shape[0];
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), {in->get_shape().at(0), oh, ow, oc},
                          out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_depthwise_conv2d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  int ow = 0, oh = 0, oc = 0;
  if (pad_mode == "FLOOR") {
    // referred to
    // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp
    ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
         padding[0] - padding[1];
    oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
         padding[2] - padding[3];
  } else if (pad_mode == "CEIL") {
    // SAME with case 0
    // see
    // https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_transpose_op.cc
    ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
         padding[0] - padding[1];
    oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
         padding[2] - padding[3];
  } else if (pad_mode == "SAME") {
    // SAME
    // see
    // https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
    ow = in_shape[2] * stride[0];
    oh = in_shape[1] * stride[1];
  } else if (pad_mode == "VALID") {
    // VALID
    ow = (in_shape[2] - 1) * stride[0] + kernel[0];
    oh = (in_shape[1] - 1) * stride[1] + kernel[1];
  } else {
    UNI_LOG_CHECK(pad_mode == "FLOOR" || pad_mode == "CEIL" ||
                      pad_mode == "SAME" || pad_mode == "VALID",
                  XIR_INVALID_ARG_OCCUR)
        << "pad_mode here is " << pad_mode
        << ", but it should be one of \"FLOOR\","
           "\"CEIL\", \"SAME\" or \"VALID\".";
  }
  oc = in_shape[3];
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), {in->get_shape().at(0), oh, ow, oc},
                          out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_conv3d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  UNI_LOG_CHECK(in_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of input here is " << in_shape.size()
      << ", but the number of dimension should be 5.";
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of weights here is " << w_shape.size()
      << ", but the number of dimension should be 5.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  UNI_LOG_CHECK(w_shape[4] == in_shape[4], XIR_INVALID_ARG_OCCUR)
      << "The channel of weights and the channel of input feature maps should "
         "be "
         "the same."
      << " But they are " << w_shape[4] << " and " << in_shape[4] << " now.";
  auto oh_ow_od =
      cal_out_3d(in_shape, padding, kernel, stride, dilation, pad_mode);
  std::vector<std::int32_t> new_out_shape = {
      in->get_shape().at(0), oh_ow_od[0], oh_ow_od[1], oh_ow_od[2], w_shape[0]};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_depthwise_conv3d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  UNI_LOG_CHECK(in_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of input here is " << in_shape.size()
      << ", but the number of dimension should be 5.";
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of weights here is " << w_shape.size()
      << ", but the number of dimension should be 5.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  UNI_LOG_CHECK(w_shape[4] == in_shape[4], XIR_INVALID_ARG_OCCUR)
      << "The channel of weights and the channel of input feature maps should "
         "be "
         "the same."
      << " But they are " << w_shape[4] << " and " << in_shape[4] << " now.";
  auto oh_ow_od =
      cal_out_3d(in_shape, padding, kernel, stride, dilation, pad_mode);
  std::vector<std::int32_t> new_out_shape = {
      in->get_shape().at(0), oh_ow_od[0], oh_ow_od[1], oh_ow_od[2], w_shape[4]};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_conv3d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  UNI_LOG_CHECK(in_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of input here is " << in_shape.size()
      << ", but the number of dimension should be 5.";
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of weights here is " << w_shape.size()
      << ", but the number of dimension should be 5.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  UNI_LOG_CHECK(w_shape[4] == in_shape[4], XIR_INVALID_ARG_OCCUR)
      << "The channel of weights and the channel of input feature maps should "
         "be "
         "the same."
      << " But they are " << w_shape[4] << " and " << in_shape[4] << " now.";
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  int ow = 0, oh = 0, od = 0, oc = 0;
  if (pad_mode == "FLOOR" || pad_mode == "CEIL") {
    oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
         padding[2] - padding[3];
    ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
         padding[0] - padding[1];
    od = (in_shape[3] - 1) * stride[2] + dilation[2] * (kernel[2] - 1) + 1 -
         padding[4] - padding[5];
  } else if (pad_mode == "SAME") {
    oh = in_shape[1] * stride[1];
    ow = in_shape[2] * stride[0];
    od = in_shape[3] * stride[2];
  } else if (pad_mode == "VALID") {
    oh = (in_shape[1] - 1) * stride[1] + kernel[1];
    ow = (in_shape[2] - 1) * stride[0] + kernel[0];
    od = (in_shape[3] - 1) * stride[2] + kernel[2];
  } else {
    UNI_LOG_CHECK(pad_mode == "FLOOR" || pad_mode == "CEIL" ||
                      pad_mode == "SAME" || pad_mode == "VALID",
                  XIR_INVALID_ARG_OCCUR)
        << "pad_mode here is " << pad_mode
        << ", but it should be one of \"FLOOR\","
           "\"CEIL\", \"SAME\" or \"VALID\".";
  }
  oc = w_shape[0];
  auto out = cur->get_output_tensor();
  auto output_tensor = xir::Tensor::create(
      out->get_name(), {in->get_shape().at(0), oh, ow, od, oc},
      out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_depthwise_conv3d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  UNI_LOG_CHECK(in_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of input here is " << in_shape.size()
      << ", but the number of dimension should be 5.";
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of weights here is " << w_shape.size()
      << ", but the number of dimension should be 5.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  UNI_LOG_CHECK(w_shape[4] == in_shape[4], XIR_INVALID_ARG_OCCUR)
      << "The channel of weights and the channel of input feature maps should "
         "be "
         "the same."
      << " But they are " << w_shape[4] << " and " << in_shape[4] << " now.";
  auto pad_mode = attrs->get_attr<std::string>("pad_mode");
  int ow = 0, oh = 0, od = 0, oc = 0;
  if (pad_mode == "FLOOR" || pad_mode == "CEIL") {
    oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
         padding[2] - padding[3];
    ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
         padding[0] - padding[1];
    od = (in_shape[3] - 1) * stride[2] + dilation[2] * (kernel[2] - 1) + 1 -
         padding[4] - padding[5];
  } else if (pad_mode == "SAME") {
    oh = in_shape[1] * stride[1];
    ow = in_shape[2] * stride[0];
    od = in_shape[3] * stride[2];
  } else if (pad_mode == "VALID") {
    oh = (in_shape[1] - 1) * stride[1] + kernel[1];
    ow = (in_shape[2] - 1) * stride[0] + kernel[0];
    od = (in_shape[3] - 1) * stride[2] + kernel[2];
  } else {
    UNI_LOG_CHECK(pad_mode == "FLOOR" || pad_mode == "CEIL" ||
                      pad_mode == "SAME" || pad_mode == "VALID",
                  XIR_INVALID_ARG_OCCUR)
        << "pad_mode here is " << pad_mode
        << ", but it should be one of \"FLOOR\","
           "\"CEIL\", \"SAME\" or \"VALID\".";
  }
  oc = w_shape[4];
  auto out = cur->get_output_tensor();
  auto output_tensor = xir::Tensor::create(
      out->get_name(), {in->get_shape().at(0), oh, ow, od, oc},
      out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_pool1d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  UNI_LOG_CHECK(in_shape.size() == 3, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of input feature maps here is "
      << in_shape.size() << ", but the number of dimension should be 3.";
  auto out = cur->get_output_tensor();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> padding = {0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 2, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 2.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::int32_t ol;
  if ((attrs->has_attr("global")) && (attrs->get_attr<bool>("global"))) {
    ol = 1;
  } else {
    std::string pad_mode = "";
    if (attrs->has_attr("pad_mode"))
      pad_mode = attrs->get_attr<std::string>("pad_mode");
    ol = cal_out_1d(in_shape, padding, kernel, stride, {1}, pad_mode);
  }
  auto output_tensor =
      xir::Tensor::create(out->get_name(),                           //
                          {in->get_shape().at(0), ol, in_shape[2]},  //
                          out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_maxpool1d(xir::Op* cur) { shape_infer_pool1d(cur); }

void shape_infer_pool2d(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  UNI_LOG_CHECK(in_shape.size() == 4, XIR_INVALID_ARG_OCCUR)
      << "The number of dimension of input feature maps here is "
      << in_shape.size() << ", but the number of dimension should be 4.";
  auto out = cur->get_output_tensor();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::vector<std::int32_t> oh_ow;
  if ((attrs->has_attr("global")) && (attrs->get_attr<bool>("global"))) {
    oh_ow = {1, 1};
  } else {
    std::string pad_mode = "";
    if (attrs->has_attr("pad_mode"))
      pad_mode = attrs->get_attr<std::string>("pad_mode");
    oh_ow = cal_out(in_shape, padding, kernel, stride, {1, 1}, pad_mode);
  }
  auto output_tensor = xir::Tensor::create(
      out->get_name(),                                           //
      {in->get_shape().at(0), oh_ow[0], oh_ow[1], in_shape[3]},  //
      out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_maxpool2d(xir::Op* cur) { shape_infer_pool2d(cur); }
void shape_infer_avgpool2d(xir::Op* cur) { shape_infer_pool2d(cur); }

void shape_infer_eltwise(xir::Op* cur) {
  auto ins = cur->get_input_tensors();
  if (ins.size() > 2) {
    std::for_each(ins.begin(), ins.end(), [ins](xir::Tensor* t) {
      UNI_LOG_CHECK(t->get_shape() == ins[0]->get_shape(),
                    XIR_INVALID_ARG_OCCUR)
          << "shape of input tensors in element-wise add op should be same."
          << " But here one of the shapes is " << xir::to_string(t->get_shape())
          << ". one of the shape is " << xir::to_string(ins[0]->get_shape());
    });
    shape_infer_remain(cur);
  } else if (ins.size() == 2) {
    shape_infer_broadcast(cur);
  } else if (ins.size() == 1) {
    shape_infer_remain(cur);
    // do nonthing
  } else {
    UNI_LOG_FATAL(XIR_INVALID_ARG_OCCUR)
        << cur->to_string() << " requires at least one input.";
  }
}

void shape_infer_concat(xir::Op* cur) {
  auto axis = cur->get_attr<std::int32_t>("axis");
  auto ins = cur->get_input_tensors();
  axis = axis < 0 ? axis + ins[0]->get_shape().size() : axis;
  for (unsigned int i = 0; i < ins.size(); i++)
    for (auto d = 0; d < static_cast<std::int32_t>(ins[0]->get_shape().size());
         d++)
      if (d != axis)
        UNI_LOG_CHECK(ins[0]->get_shape().at(d) == ins[i]->get_shape().at(d),
                      XIR_INVALID_ARG_OCCUR)
            << "Wrong axis! "
            << "The dimensions except the axis defined by user of input "
               "feature "
               "maps "
               "of concat op should be same."
            << " But here one of the shapes is "
            << xir::to_string(ins[i]->get_shape()) << ", one of the shape is "
            << xir::to_string(ins[0]->get_shape()) << ", and the axis is "
            << axis;
  auto out = cur->get_output_tensor();
  auto o_axis = static_cast<int32_t>(
      std::accumulate(ins.begin(), ins.end(), 0,
                      [out, axis](std::int32_t o_axis, xir::Tensor* t) {
                        return o_axis + t->get_shape().at(axis);
                      }));
  std::vector<std::int32_t> out_shape;
  for (auto i = 0; i < static_cast<std::int32_t>(ins[0]->get_shape().size());
       i++) {
    if (i == axis)
      out_shape.push_back(o_axis);
    else
      out_shape.push_back(ins[0]->get_shape().at(i));
  }
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_inner_product(xir::Op* cur) {
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  auto weights_shape = cur->get_input_tensor("weights")->get_shape();
  auto axis = cur->get_attr<std::int32_t>("axis");
  std::vector<std::int32_t> out_shape;
  int n = 1;
  for (auto i = 0; i < axis; i++) {
    out_shape.push_back(in_shape[i]);
    n *= in_shape[i];
  }
  auto c = cur->get_input_tensor("input")->get_element_num() / n;
  auto k = cur->get_input_tensor("weights")->get_element_num() / c;
  UNI_LOG_CHECK(k * c == cur->get_input_tensor("weights")->get_element_num(),
                XIR_UNEXPECTED_VALUE);
  out_shape.push_back(k);
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_relu(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_leaky_relu(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_prelu(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_relu6(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_elu(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_celu(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_selu(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_gelu(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_mish(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_sigmoid(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_swish(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_tanh(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_hard_sigmoid(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_hard_swish(xir::Op* cur) { shape_infer_remain(cur); }
void shape_infer_hard_tanh(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_reorg(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto scale = cur->get_attr<std::int32_t>("scale");
  auto if_reverse = cur->get_attr<bool>("reverse");
  auto out = cur->get_output_tensor();
  if (!if_reverse) {
    auto ow = in->get_shape().at(2) / scale;
    auto oh = in->get_shape().at(1) / scale;
    auto oc = in->get_shape().at(3) * scale * scale;
    auto output_tensor = xir::Tensor::create(
        out->get_name(), {in->get_shape().at(0), oh, ow, oc},
        out->get_data_type());
    output_tensor->set_attrs(out->get_attrs());
    cur->replace_output_tensor(std::move(output_tensor));
  } else {
    auto ow = in->get_shape().at(2) * scale;
    auto oh = in->get_shape().at(1) * scale;
    auto oc = in->get_shape().at(3) / scale / scale;
    auto output_tensor = xir::Tensor::create(
        out->get_name(), {in->get_shape().at(0), oh, ow, oc},
        out->get_data_type());
    output_tensor->set_attrs(out->get_attrs());
    cur->replace_output_tensor(std::move(output_tensor));
  }
}

void shape_infer_fix(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_softmax(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_cast(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto out = cur->get_output_tensor();
  auto its = cur->get_input_tensors();
  auto out_shape = in->get_shape();
  auto dtype = cur->get_attr<std::string>("data_type");
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, DataType(dtype));
  output_tensor->set_attrs(out->get_attrs());
  for (auto it : its) {
    if (it->has_attr("shape_info")) {
      auto si = it->get_attr<std::vector<std::int32_t>>("shape_info");
      output_tensor->set_attr("shape_info", si);
    }
  }
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_squeeze(xir::Op* cur) {
  std::vector<std::int32_t> index;
  auto in = cur->get_input_tensor("input");
  if (cur->has_attr("axis")) {
    auto dims = cur->get_attr<std::vector<std::int32_t>>("axis");
    for (auto dim : dims) index.push_back(dim);
    if (dims.size() == 0)
      for (int i = 0; i < static_cast<int>(in->get_shape().size()); i++)
        if (in->get_shape().at(i) == 1) index.push_back(i);
  } else {
    for (int i = 0; i < static_cast<int>(in->get_shape().size()); i++)
      if (in->get_shape().at(i) == 1) index.push_back(i);
  }
  auto out_shape = in->get_shape();
  if (index.size() != 0) {
    std::sort(index.begin(), index.end());
    for (int i = index.size() - 1; i > -1; i--)
      out_shape.erase(out_shape.begin() + index[i]);
  }
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_reshape(xir::Op* cur) {
  std::vector<std::int32_t> reshape;
  auto shape_ops = cur->get_input_ops("shape");
  if (shape_ops.size()) {
    UNI_LOG_CHECK(shape_ops.size() == 1, XIR_INVALID_ARG_OCCUR)
        << cur->to_string() << " requires one and only one shape arg.";
    auto reshape_op = shape_ops[0];
    UNI_LOG_CHECK(reshape_op != nullptr, XIR_UNEXPECTED_VALUE)
        << cur->to_string() << "'s shape op is invalid.";
    if (reshape_op->has_attr("data")) {
      reshape = read_data_in_attr<std::int32_t>(reshape_op);
    } else {
      auto inot = reshape_op->get_output_tensor();
      if (inot->has_attr("shape_info")) {
        reshape = inot->get_attr<std::vector<std::int32_t>>("shape_info");
      } else {
        UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
            << "I don't know how to reshape the feature maps of "
            << cur->get_name();
      }
    }
  } else {
    reshape = cur->get_attr<std::vector<std::int32_t>>("shape");
  }

  int size_r = 1;
  auto in = cur->get_input_tensor("input");

  for (unsigned int i = 0; i < reshape.size(); i++) {
    auto dim = reshape[i];
    if (dim <= 0) {
      size_r *= 1;
    } else if (dim == 0) {
      size_r *= in->get_shape()[i];
    } else {
      size_r *= dim;
    }
  }

  int value = in->get_element_num() / size_r;
  for (unsigned int i = 0; i < reshape.size(); i++) {
    if (reshape[i] < 0)
      reshape[i] = value;
    else if (reshape[i] == 0)
      reshape[i] = in->get_shape()[i];
  }
  int r_size = 1;
  for (auto s : reshape) r_size *= s;
  UNI_LOG_CHECK(in->get_element_num() == r_size, XIR_UNEXPECTED_VALUE)
      << cur->to_string() << "'s input elements number is "
      << in->get_element_num() << ", but the output elements number is "
      << r_size << ". Input shape is " << to_string(in->get_shape())
      << ", and the output shape is " << to_string(reshape) << ".";
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), reshape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_float2fix(xir::Op* cur) {
  auto if_signed = cur->get_attr<bool>("if_signed");
  DataType data_type{(if_signed ? "XINT" : "XUINT") +
                     std::to_string(cur->get_attr<std::int32_t>("bit_width"))};
  auto input_tensor = cur->get_input_tensor("input");
  auto output_tensor = cur->get_output_tensor();
  auto new_output_tensor = xir::Tensor::create(
      output_tensor->get_name(), input_tensor->get_shape(), data_type);
  new_output_tensor->set_attrs(output_tensor->get_attrs());
  cur->replace_output_tensor(std::move(new_output_tensor));
}
void shape_infer_fix2float(xir::Op* cur) {
  auto input_tensor = cur->get_input_tensor("input");
  auto output_tensor = cur->get_output_tensor();
  auto new_output_tensor =
      xir::Tensor::create(output_tensor->get_name(), input_tensor->get_shape(),
                          DataType{DataType::FLOAT, sizeof(float) * 8});
  new_output_tensor->set_attrs(output_tensor->get_attrs());
  cur->replace_output_tensor(std::move(new_output_tensor));
}

void shape_infer_pad(xir::Op* cur) {
  auto padding = cur->get_attr<std::vector<std::int32_t>>("paddings");
  auto in = cur->get_input_tensor("input");
  UNI_LOG_CHECK(padding.size() == in->get_shape().size() * 2,
                XIR_UNEXPECTED_VALUE);
  auto out = cur->get_output_tensor();
  std::vector<std::int32_t> out_shape;
  for (unsigned int i = 0; i < in->get_shape().size(); i++) {
    auto shape = in->get_shape().at(i) + padding[2 * i] + padding[2 * i + 1];
    out_shape.push_back(shape);
  }
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_reduction(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto dims = cur->get_attr<std::vector<std::int32_t>>("axis");
  auto keep_dims = cur->get_attr<bool>("keep_dims");
  std::int32_t dim_size = in->get_shape().size();
  std::vector<bool> bitmap(dim_size, false);
  for (auto index : dims) {
    UNI_LOG_CHECK(index >= -dim_size && index < dim_size, XIR_INVALID_ARG_OCCUR)
        << "ERROR: index < -data.dims() || index >= data.dims()";
    index = (index + dim_size) % dim_size;
    bitmap[index] = true;
  }
  std::vector<std::int32_t> out_shape;
  for (auto i = 0; i < dim_size; i++) {
    if (!bitmap[i]) {
      out_shape.push_back(in->get_shape().at(i));
    } else if (keep_dims) {
      out_shape.push_back(1);
    }
  }
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

// referred to
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/reduction_ops_common.cc
void shape_infer_reduction_mean(xir::Op* cur) { shape_infer_reduction(cur); }

void forward_reduction_product(xir::Op* cur) {
  auto inv = cur->get_input_tensor("input", 0);
  std::vector<int> shape{};
  if (inv->has_attr("shape_info"))
    shape = inv->get_attr<std::vector<std::int32_t>>("shape_info");
  else
    return;
  auto dims = cur->get_attr<std::vector<std::int32_t>>("axis");
  UNI_LOG_CHECK(dims.size() == 1 || dims[0] == 0, XIR_INVALID_ARG_OCCUR);
  int mul = 1;
  for (auto s : shape) mul *= s;
  auto out = cur->get_output_tensor();
  out->set_attr("shape_info", std::vector<int>({mul}));
}

void shape_infer_reduction_product(xir::Op* cur) {
  shape_infer_reduction(cur);
  forward_reduction_product(cur);
}

void forward_reduction_sum(xir::Op* cur) {
  auto inv = cur->get_input_tensor("input", 0);
  std::vector<int> shape{};
  if (inv->has_attr("shape_info"))
    shape = inv->get_attr<std::vector<std::int32_t>>("shape_info");
  else
    return;
  auto dims = cur->get_attr<std::vector<std::int32_t>>("axis");
  UNI_LOG_CHECK(dims.size() == 1 || dims[0] == 0, XIR_INVALID_ARG_OCCUR);
  int sum = 0;
  for (auto s : shape) sum += s;
  auto out = cur->get_output_tensor();
  out->set_attr("shape_info", std::vector<int>({sum}));
}

void shape_infer_reduction_sum(xir::Op* cur) {
  shape_infer_reduction(cur);
  forward_reduction_sum(cur);
}

void forward_reduction_max(xir::Op* cur) {
  auto inv = cur->get_input_tensor("input", 0);
  std::vector<int> shape{};
  if (inv->has_attr("shape_info"))
    shape = inv->get_attr<std::vector<std::int32_t>>("shape_info");
  else
    return;
  auto dims = cur->get_attr<std::vector<std::int32_t>>("axis");
  UNI_LOG_CHECK(dims.size() == 1 || dims[0] == 0, XIR_INVALID_ARG_OCCUR);
  int max = 0;
  for (auto s : shape)
    if (s > max) max = s;
  auto out = cur->get_output_tensor();
  out->set_attr("shape_info", std::vector<int>({max}));
}

void shape_infer_reduction_max(xir::Op* cur) {
  shape_infer_reduction(cur);
  forward_reduction_max(cur);
}

void shape_infer_reduction_max_fix(xir::Op* cur) {
  shape_infer_reduction_max(cur);
}

void shape_infer_l2_normalize(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_exp(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_scale(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_resize(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto out = cur->get_output_tensor();
  std::vector<std::int32_t> out_shape = in->get_shape();
  auto mode = cur->get_attr<std::string>("mode");
  if (cur->get_input_num() > 1) {
    auto ops = internal::vec_input_ops(cur->get_input_ops());
    for (auto op : ops) {
      auto out_tensor = op->get_output_tensor();
      if (op->has_attr("data")) {
        auto size = read_data_in_attr<int32_t>(op);
        if (size.size() == 2 || size.size() == 4) {
          UNI_LOG_CHECK(mode == "NEAREST" || mode == "BILINEAR",
                        XIR_INVALID_ARG_OCCUR)
              << "the dimension number of the output feature maps of resize "
                 "operation is "
              << size.size() << ", the mode of resize is " << mode
              << ". We support dimension number of output feature maps to be 2 "
                 "or 4 for NEAREST and BILINEAR, 3 or 5 for TRILINEAR.";
          if (size.size() == 2) {
            out_shape[1] = size[0];
            out_shape[2] = size[1];
          } else {
            UNI_LOG_CHECK(out_shape[0] == size[0] && out_shape[3] == size[3],
                          XIR_INVALID_ARG_OCCUR)
                << "resize 4-d feature maps could only implement along the "
                   "diemnsion of the width and the height.";
            out_shape = size;
          }
        } else if (size.size() == 3 || size.size() == 5) {
          UNI_LOG_CHECK(mode == "TRILINEAR", XIR_INVALID_ARG_OCCUR)
              << "the dimension number of the output feature maps of resize "
                 "operation is "
              << size.size() << ", the mode of resize is " << mode
              << ". We support dimension number of output feature maps to be 2 "
                 "or 4 for NEAREST and BILINEAR, 3 or 5 for TRILINEAR.";
          if (size.size() == 3) {
            out_shape[1] = size[0];
            out_shape[2] = size[1];
            out_shape[3] = size[2];
          } else {
            UNI_LOG_CHECK(out_shape[0] == size[0] && out_shape[4] == size[4],
                          XIR_INVALID_ARG_OCCUR)
                << "resize 5-d feature maps could only implement along the "
                   "diemnsion of the width, the height and the depth.";
            out_shape = size;
          }
        }
        UNI_LOG_CHECK(out_shape.size() ==
                          static_cast<unsigned int>(in->get_shape().size()),
                      XIR_INVALID_ARG_OCCUR)
            << "the number of dimensions shoud be the same after the resize "
               "op.";
      } else if (out_tensor->has_attr("shape_info")) {
        auto size =
            out_tensor->get_attr<std::vector<std::int32_t>>("shape_info");
        UNI_LOG_CHECK(size.size() == 2 || size.size() == 3,
                      XIR_INVALID_ARG_OCCUR)
            << "New size of feature maps should have two dimensions H, W or "
               "three dimensions H, W, D.";
        out_shape[1] = size[0];
        out_shape[2] = size[1];
        if (size.size() == 3) out_shape[3] = size[2];
      }
      UNI_LOG_CHECK(
          out_shape.size() == static_cast<unsigned int>(in->get_shape().size()),
          XIR_INVALID_ARG_OCCUR)
          << "the number of dimensions shoud be the same after the resize "
             "op.";
    }
  } else if (cur->has_attr("scale")) {
    auto scale = get_float_vec_from_any(cur->get_attr("scale"));
    UNI_LOG_CHECK(scale.size() == 2 || scale.size() == 3, XIR_INVALID_ARG_OCCUR)
        << "Scale should have two dimensions (scale_w, scale_h) or three "
           "dimensions (scale_w, scale_h, scale_d).";
    UNI_LOG_CHECK(
        std::all_of(scale.begin(), scale.end(), [](auto s) { return s > 0; }),
        XIR_INVALID_ARG_OCCUR)
        << "Scale should be > 0.";
    if (scale.size() == 2) {
      UNI_LOG_CHECK(mode == "NEAREST" || mode == "BILINEAR",
                    XIR_INVALID_ARG_OCCUR)
          << "the number of the attribute scale of resize "
             "operation is "
          << scale.size() << ", the mode of resize is " << mode
          << ". We support the number of scale to be 2 for NEAREST and "
             "BILINEAR, 3 for TRILINEAR.";
    } else if (scale.size() == 3) {
      UNI_LOG_CHECK(mode == "TRILINEAR", XIR_INVALID_ARG_OCCUR)
          << "the number of the attribute scale of resize "
             "operation is "
          << scale.size() << ", the mode of resize is " << mode
          << ". We support the number of scale to be 2 for NEAREST and "
             "BILINEAR, 3 for TRILINEAR.";
    }
    auto ow = static_cast<std::int32_t>(in->get_shape().at(2) * scale[0]);
    auto oh = static_cast<std::int32_t>(in->get_shape().at(1) * scale[1]);
    out_shape[1] = oh;
    out_shape[2] = ow;
    if (scale.size() == 3) {
      auto od = static_cast<std::int32_t>(in->get_shape().at(3) * scale[2]);
      out_shape[3] = od;
    }
  } else {
    UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
        << "I don't know how to resize the feature maps.";
  }
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_gstiling(xir::Op* cur) {
  auto reverse = cur->get_attr<bool>("reverse");
  auto stride = cur->get_attr<std::int32_t>("stride");
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  auto ic = in_shape[3];
  auto iw = in_shape[2];
  auto ih = in_shape[1];
  auto stride_sq = stride * stride;
  std::int32_t ow, oh, oc;
  if (reverse) {
    oc = ic / stride_sq;
    ow = iw * stride;
    oh = ih * stride;
    UNI_LOG_CHECK(ic % stride_sq == 0, XIR_INVALID_ARG_OCCUR)
        << "The number of input channels for tiling layer must be multiples "
        << "of the stride * stride.";
  } else {
    oc = ic * stride_sq;
    ow = iw / stride;
    oh = ih / stride;
    UNI_LOG_CHECK(iw % stride == 0, XIR_INVALID_ARG_OCCUR)
        << "The number of input width for tiling layer must be multiples "
        << "of the stride.";
    UNI_LOG_CHECK(ih % stride == 0, XIR_INVALID_ARG_OCCUR)
        << "The number of input height for tiling layer must be multiples "
        << "of the stride.";
  }
  std::vector<std::int32_t> out_shape = {in_shape[0], oh, ow, oc};
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_pixel_shuffle(xir::Op* cur) {
  auto upscale = cur->get_attr<bool>("upscale");
  auto scale = cur->get_attr<std::int32_t>("scale");
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  auto ic = in_shape[3];
  auto iw = in_shape[2];
  auto ih = in_shape[1];
  auto scale_sq = scale * scale;
  std::int32_t ow, oh, oc;
  if (upscale) {
    oc = ic / scale_sq;
    ow = iw * scale;
    oh = ih * scale;
    UNI_LOG_CHECK(ic % scale_sq == 0, XIR_INVALID_ARG_OCCUR)
        << "The number of input channels for pixel_shuffle layer must be "
           "multiples "
        << "of the scale * scale.";
  } else {
    oc = ic * scale_sq;
    ow = iw / scale;
    oh = ih / scale;
    UNI_LOG_CHECK(iw % scale == 0, XIR_INVALID_ARG_OCCUR)
        << "The number of input width for pixel_shuffle layer must be "
           "multiples "
        << "of the scale.";
    UNI_LOG_CHECK(ih % scale == 0, XIR_INVALID_ARG_OCCUR)
        << "The number of input height for pixel_shuffle layer must be "
           "multiples "
        << "of the scale.";
  }
  std::vector<std::int32_t> out_shape = {in_shape[0], oh, ow, oc};
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_pixel_shuffle_fix(xir::Op* cur) {
  shape_infer_pixel_shuffle(cur);
}

void shape_infer_batchnorm(xir::Op* cur) {
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  auto gamma = cur->get_input_tensor("gamma")->get_shape();
  auto beta = cur->get_input_tensor("beta")->get_shape();
  auto axis = cur->get_attr<int>("axis");
  UNI_LOG_CHECK(in_shape[axis] == gamma[0] && in_shape[axis] == beta[0],
                XIR_INVALID_ARG_OCCUR)
      << "The number of input channels for bn must be equal with "
      << "the number of beta and gamma.";
  shape_infer_remain(cur);
}

void shape_infer_instancenorm(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_groupnorm(xir::Op* cur) { shape_infer_remain(cur); }

/// Note: The shape info for constant operator has been written into
/// its output tensor when generating XIR model , so there is no need to infer
/// shape for it. But for safety consideration, it's better to check it.
void shape_infer_const(xir::Op* cur) { shape_infer_data(cur); }

void shape_infer_data(xir::Op* cur) {
  auto out = cur->get_output_tensor();
  auto out_shape = cur->get_attr<std::vector<std::int32_t>>("shape");
  UNI_LOG_CHECK(out_shape.size() > 0, XIR_INVALID_ARG_OCCUR)
      << cur->to_string() << "'s output tensor's shape is zero dimension.";
  UNI_LOG_CHECK(std::all_of(out_shape.begin(), out_shape.end(),
                            [](const std::int32_t& dim) { return (dim > 0); }),
                XIR_INVALID_ARG_OCCUR)
      << cur->to_string()
      << " has negtive shape : " << xir::to_string(out_shape) << ".";
  auto out_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  out_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(out_tensor));
}

void shape_infer_final(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_shape(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto out = cur->get_output_tensor();
  std::vector<std::int32_t> shape = in->get_shape();
  std::vector<std::int32_t> out_shape = {
      static_cast<std::int32_t>(in->get_shape().size())};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
  auto new_out = cur->get_output_tensor();
  new_out->set_attr("shape_info", shape);
}

void shape_infer_neg(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_add(xir::Op* cur) { shape_infer_broadcast(cur); }

void shape_infer_sub(xir::Op* cur) { shape_infer_broadcast(cur); }

void forward_mul(xir::Op* cur) {
  auto inv = cur->get_input_tensor("input", 0);
  auto ops = internal::vec_input_ops(cur->get_input_ops());
  auto mul0 = inv->get_attr<std::vector<std::int32_t>>("shape_info");
  std::vector<std::int32_t> mul1;
  for (auto op : ops) {
    if (op->has_attr("data")) {
      mul1 = read_data_in_attr<int32_t>(op);
    }
  }
  std::vector<std::int32_t> value;
  if (mul0.size() == mul1.size())
    for (unsigned int i = 0; i < mul0.size(); ++i)
      value.push_back(mul0[i] * mul1[i]);
  auto out = cur->get_output_tensor();
  out->set_attr("shape_info", value);
}

void shape_infer_mul(xir::Op* cur) {
  shape_infer_broadcast(cur);
  if (cur->get_input_num() > 1) {
    auto inv = cur->get_input_tensor("input", 0);
    if (inv->has_attr("shape_info")) {
      forward_mul(cur);
    }
  }
}

void forward_div(xir::Op* cur) {
  auto inv = cur->get_input_tensor("input", 0);
  auto shape = inv->get_attr<std::vector<std::int32_t>>("shape_info");
  auto base = read_data_in_attr<int32_t>(cur->get_input_ops("input")[1]);
  if (base.size() == 1)
    for (unsigned int i = 0; i < shape.size(); i++)
      shape[i] = shape[i] / base[0];
  else {
    UNI_LOG_CHECK(shape.size() == base.size(), XIR_INVALID_ARG_OCCUR);
    for (unsigned int i = 0; i < shape.size(); i++)
      shape[i] = shape[i] / base[i];
  }
  auto out = cur->get_output_tensor();
  out->set_attr("shape_info", shape);
}

void shape_infer_div(xir::Op* cur) {
  shape_infer_broadcast(cur);
  if (cur->get_input_num() > 1) {
    auto inv = cur->get_input_tensor("input", 0);
    if (inv->has_attr("shape_info")) {
      forward_div(cur);
    }
  }
}

void shape_infer_min(xir::Op* cur) { shape_infer_broadcast(cur); }

void shape_infer_max(xir::Op* cur) { shape_infer_broadcast(cur); }

void shape_infer_argmax(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto axis = cur->get_attr<std::int32_t>("axis");
  axis = axis < 0 ? axis + in->get_shape().size() : axis;
  auto out_shape = in->get_shape();
  out_shape[axis] = 1;
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_argmax_fix(xir::Op* cur) { shape_infer_argmax(cur); }

void shape_infer_threshold(xir::Op* cur) { shape_infer_remain(cur); }

void forward_strided_slice(xir::Op* cur, std::vector<std::int32_t> begin,
                           std::vector<std::int32_t> end,
                           std::vector<std::int32_t> stride) {
  // auto shrink_axis_mask = cur->get_attr<std::int32_t>("shrink_axis_mask");
  // if (shrink_axis_mask - 1 >= 0)
  //   end[shrink_axis_mask - 1] = begin[shrink_axis_mask - 1] + 1;
  std::int32_t in_shape_num =
      cur->get_input_tensor("input")->get_shape().size();
  std::vector<std::int32_t> indata;
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  auto in_size = cur->get_input_tensor("input")->get_element_num();
  auto its = cur->get_input_tensors();
  for (auto it : its) {
    if (it->has_attr("shape_info")) {
      auto si = it->get_attr<std::vector<std::int32_t>>("shape_info");
      indata.resize(si.size());
      for (auto i = 0; i < in_size; i++) indata[i] = si[i];
    }
  }
  if (indata.empty()) return;
  UNI_LOG_CHECK(begin.size() == (unsigned int)in_shape_num,
                XIR_INVALID_ARG_OCCUR)
      << "the size of begin is: " << begin.size()
      << ", the size of input tensor is: " << in_shape_num << ".";
  std::vector<std::int32_t> idx_stride{1};
  int32_t is = 1;
  if (begin.size() > 1) {
    for (std::int32_t i = begin.size() - 1; i >= 0; i--) {
      is *= in_shape[i];
      idx_stride.insert(idx_stride.begin(), is);
    }
  }
  std::vector<std::int32_t> out_value;
  std::vector<std::int32_t> idx;

  for (int32_t i = 0; i < in_size; i++) {
    auto x = i;
    bool if_push = true;
    for (int j = 0; j < in_shape_num; j++) {
      if (x / idx_stride[j] < begin[j] || x / idx_stride[j] >= end[j] ||
          abs(x - begin[j]) % stride[j] != 0) {
        if_push = false;
        break;
      }
      x = x % idx_stride[j];
    }
    if (if_push) out_value.push_back(indata[i]);
  }
  auto out = cur->get_output_tensor();
  out->set_attr<std::vector<std::int32_t>>("shape_info", out_value);
}

// The documentations of strided_slice in tensorflow are confusing and
// ambiguous, as a result, i use the code in
// tensorflow/core/util/strided_slice_op.cc as reference.
void shape_infer_strided_slice(xir::Op* cur) {
  std::int32_t in_shape_num =
      cur->get_input_tensor("input")->get_shape().size();
  std::vector<int32_t> begin(in_shape_num);
  std::vector<int32_t> end(in_shape_num);
  std::vector<int32_t> strides(in_shape_num);
  std::vector<int32_t> out_shape;
  validate_strided_slice(cur, begin, end, strides, out_shape);
  auto out = cur->get_output_tensor();
  // This is a strange condition, if the input is an one-dimension tensor,
  // shrink makes the output tensor to be a zero-dimension tensor, but
  // this tensor would contain one data.
  // So, here, we make the output shape to be [1].
  if (out_shape.size() == 0) out_shape.push_back(1);
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
  forward_strided_slice(cur, begin, end, strides);
}

void shape_infer_space_to_batch_nd(xir::Op* cur) {
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  auto scale = cur->get_attr<std::vector<std::int32_t>>("block_shape");
  UNI_LOG_CHECK(scale.size() == 2, XIR_INVALID_ARG_OCCUR)
      << "the dim_num of block_shape should be 2.";
  auto b = in_shape[0] * scale[0] * scale[0];
  auto pad = cur->get_attr<std::vector<std::int32_t>>("paddings");
  UNI_LOG_CHECK(pad.size() == 4, XIR_INVALID_ARG_OCCUR)
      << "the dim_num of pad should be 4.";
  auto h = (in_shape[1] + pad[2] + pad[3]) / scale[1];
  auto w = (in_shape[2] + pad[0] + pad[1]) / scale[0];
  auto c = in_shape[3];
  std::vector<std::int32_t> out_shape = {b, h, w, c};
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_batch_to_space_nd(xir::Op* cur) {
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  auto scale = cur->get_attr<std::vector<std::int32_t>>("block_shape");
  UNI_LOG_CHECK(scale.size() == 2, XIR_INVALID_ARG_OCCUR)
      << "the dim_num of block_shape should be 2.";
  auto b = in_shape[0] / scale[0] / scale[0];
  auto crops = cur->get_attr<std::vector<std::int32_t>>("crops");
  UNI_LOG_CHECK(crops.size() == 4, XIR_INVALID_ARG_OCCUR)
      << "the dim_num of crops should be 4.";
  auto h = in_shape[1] * scale[1] - crops[2] - crops[3];
  auto w = in_shape[2] * scale[0] - crops[0] - crops[1];
  auto c = in_shape[3];
  std::vector<std::int32_t> out_shape = {b, h, w, c};
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

// only support concat-like stack now.
void forward_stack(xir::Op* cur) {
  auto inputs = internal::vec_input_ops(cur->get_input_ops());
  std::vector<std::int32_t> shape_info;
  for (auto it : inputs) {
    auto tensor = it->get_output_tensor();
    if (tensor->has_attr("shape_info")) {
      auto i_shape_info =
          tensor->get_attr<std::vector<std::int32_t>>("shape_info");
      std::transform(i_shape_info.begin(), i_shape_info.end(),
                     std::back_inserter(shape_info),
                     [&](std::int32_t v) { return v; });
    } else {
      auto other_shape = read_data_in_attr<int32_t>(it);
      std::transform(other_shape.begin(), other_shape.end(),
                     std::back_inserter(shape_info),
                     [&](std::int32_t v) { return v; });
    }
  }
  auto output_tensor = cur->get_output_tensor();
  output_tensor->set_attr<std::vector<std::int32_t>>("shape_info", shape_info);
}

void shape_infer_stack(xir::Op* cur) {
  auto in_shape = cur->get_input_tensor("input")->get_shape();
  UNI_LOG_CHECK(cur->has_attr("axis"), XIR_INVALID_ARG_OCCUR)
      << "stack op should have parameter \"axis\".";
  auto axis = cur->get_attr<int>("axis");
  auto N = cur->get_input_num();
  auto out = cur->get_output_tensor();
  std::vector<std::int32_t> out_shape = in_shape;
  out_shape.insert(out_shape.begin() + axis, N);
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
  forward_stack(cur);
}

void shape_infer_matmul(xir::Op* cur) {
  auto inputs = cur->get_input_ops("input");
  UNI_LOG_CHECK(inputs.size() == 2, XIR_INVALID_ARG_OCCUR)
      << "\"" << cur->get_name() << "\"(\"" << cur->get_type()
      << "\") requires two and only two inputs, but there's " << inputs.size()
      << " input(s) in the graph.";
  auto bias = cur->get_input_ops("bias");
  UNI_LOG_CHECK(bias.size() <= 1, XIR_INVALID_ARG_OCCUR)
      << cur->to_string() << " has more than one bias.";
  auto op_mat_a = inputs[0];
  auto op_mat_b = inputs[1];
  auto mat_a_tensor = op_mat_a->get_output_tensor();
  auto mat_b_tensor = op_mat_b->get_output_tensor();
  auto mat_a_size = mat_a_tensor->get_shape();
  auto mat_b_size = mat_b_tensor->get_shape();
  UNI_LOG_CHECK(mat_a_size.size() > 1, XIR_UNEXPECTED_VALUE)
      << cur->to_string() << "'s input " << op_mat_a->to_string()
      << "'s output tensor shape is " << to_string(mat_a_size)
      << ", it should be larger than 1 dimensions.";
  UNI_LOG_CHECK(mat_b_size.size() > 1, XIR_UNEXPECTED_VALUE)
      << cur->to_string() << "'s input " << op_mat_b->to_string()
      << "'s output tensor shape is " << to_string(mat_b_size)
      << ", it should be larger than 1 dimensions.";
  bool transpose_a = cur->get_attr<bool>("transpose_a");
  bool transpose_b = cur->get_attr<bool>("transpose_b");
  // tranpose the last the second last dim
  auto transpose = [&](std::vector<std::int32_t>& size) {
    auto temp = *size.rbegin();
    *size.rbegin() = *(size.rbegin() + 1);
    *(size.rbegin() + 1) = temp;
  };
  if (transpose_a) {
    transpose(mat_a_size);
  }
  if (transpose_b) {
    transpose(mat_b_size);
  }
  if (bias.size()) {
    auto bias_shape = bias.at(0)->get_output_tensor()->get_shape();
    UNI_LOG_CHECK(bias_shape.size() == 1, XIR_INVALID_ARG_OCCUR)
        << cur->to_string() << "'s bias' shape is " << to_string(bias_shape)
        << ", but xir only support one dimension bias.";
    UNI_LOG_CHECK(bias_shape.at(0) == (*mat_b_size.rbegin()),
                  XIR_INVALID_ARG_OCCUR)
        << cur->to_string() << "'s bias' shape is " << to_string(bias_shape)
        << ", but the last dimension of the output tensor is "
        << *mat_b_size.rbegin() << ", so they are not matching.";
  }
  // broad cast the input shape das and dbs to dos
  auto sbc_rlt = size_broadcast(                                            //
      std::vector<std::int32_t>{mat_a_size.begin(), mat_a_size.end() - 2},  //
      std::vector<std::int32_t>{mat_b_size.begin(), mat_b_size.end() - 2});
  UNI_LOG_CHECK((mat_a_size.at(mat_a_size.size() - 1) ==
                 mat_b_size.at(mat_b_size.size() - 2))  //
                    && std::get<0>(sbc_rlt),
                XIR_INVALID_ARG_OCCUR)
      << cur->to_string()
      << "'s two inputs shape unmatch, the multiplier's shape is "
      << to_string(mat_a_size) << ", but the multiplicand's shape is "
      << to_string(mat_b_size)
      << ", according to the definetion of xir::matmul, their shape are "
         "unmatching.";
  auto out = cur->get_output_tensor();
  std::vector<std::int32_t> out_shape{std::get<1>(sbc_rlt)};
  out_shape.push_back(*(mat_a_size.rbegin() + 1));
  out_shape.push_back(*(mat_b_size.rbegin()));
  auto output_tensor =
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_flatten(xir::Op* cur) {
  auto input_tensor = cur->get_input_tensor("input");
  auto input_size = input_tensor->get_shape();
  auto tensor = cur->get_output_tensor();
  // according to
  // https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten,
  // flatten op keeps the batch size. according to
  // https://pytorch.org/docs/stable/nn.html?highlight=flatten#torch.nn.Flatten,
  // flatten op keeps the batch size and can be user-defined.
  auto start_dim = cur->get_attr<std::int32_t>("start_axis");
  auto end_dim = cur->get_attr<std::int32_t>("end_axis");
  start_dim = start_dim < 0 ? (start_dim + input_size.size()) : start_dim;
  end_dim = end_dim < 0 ? (end_dim + input_size.size()) : end_dim;
  UNI_LOG_CHECK(start_dim >= 0, XIR_UNEXPECTED_VALUE)
      << cur->to_string() << "'s attribute \"start_dim\" is " << start_dim
      << ", less than zero.";
  UNI_LOG_CHECK(end_dim < (std::int32_t)input_size.size(), XIR_UNEXPECTED_VALUE)
      << cur->to_string() << "'s attribute \"end_dim\" is " << end_dim
      << ", larger than its input tensor's dimension number.";
  UNI_LOG_CHECK(start_dim <= end_dim, XIR_UNEXPECTED_VALUE)
      << cur->to_string() << "'s attribute \"start_dim\" is " << start_dim
      << ", but the attribute \"end_dim\" is " << end_dim
      << ", less than the \"start_dim\".";
  std::int32_t flattened_dim = 1;
  for (std::int32_t idx = start_dim; idx < end_dim + 1; idx++) {
    flattened_dim *= input_size[idx];
  }
  std::vector<std::int32_t> new_size;
  for (auto i = 0; i < start_dim; i++) new_size.push_back(input_size[i]);
  new_size.push_back(flattened_dim);
  for (unsigned int i = end_dim + 1; i < input_size.size(); i++)
    new_size.push_back(input_size[i]);
  auto new_tensor = xir::Tensor::create(tensor->get_name(), new_size,
                                        tensor->get_data_type());
  new_tensor->set_attrs(tensor->get_attrs());
  cur->replace_output_tensor(std::move(new_tensor));
}

void shape_infer_identity(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_placeholder(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_upload(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_download(xir::Op* cur) { shape_infer_remain(cur); }

void shape_infer_transpose(xir::Op* cur) {
  auto input_tensor = cur->get_input_tensor("input");
  auto input_shape = input_tensor->get_shape();
  auto output_tensor = cur->get_output_tensor();
  auto order = cur->get_attr<std::vector<std::int32_t>>("order");
  std::vector<std::int32_t> output_shape;
  for (unsigned int i = 0; i < order.size(); i++)
    output_shape.push_back(input_shape[order[i]]);
  auto new_tensor = xir::Tensor::create(output_tensor->get_name(), output_shape,
                                        output_tensor->get_data_type());
  new_tensor->set_attrs(output_tensor->get_attrs());
  cur->replace_output_tensor(std::move(new_tensor));
}

void shape_infer_priorbox(xir::Op* cur) {
  // The implementation of shape_infer_priorbox follows
  // the caffe implementation in the following link:
  // https://github.com/intel/caffe/blob/master/src/caffe/layers/prior_box_layer.cpp
  auto input_tensor = cur->get_input_tensor("input");
  auto input_shape = input_tensor->get_shape();
  UNI_LOG_CHECK(input_shape.size() == 4, XIR_UNEXPECTED_VALUE)
      << cur->to_string() << " requires 4 dimensions input tensor.";
  auto output_tensor = cur->get_output_tensor();
  std::vector<std::int32_t> output_shape;
  auto min_sizes = cur->get_attr<std::vector<float>>("min_sizes");
  auto max_sizes = cur->get_attr<std::vector<float>>("max_sizes");
  auto aspect_ratios = cur->get_attr<std::vector<float>>("aspect_ratio");
  std::vector<float> aspect_ratios_;
  aspect_ratios_.push_back(1.);
  auto flip = cur->get_attr<bool>("flip");
  for (unsigned int i = 0; i < aspect_ratios.size(); ++i) {
    float ar = aspect_ratios[i];
    bool already_exist = false;
    for (unsigned int j = 0; j < aspect_ratios_.size(); ++j) {
      if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      aspect_ratios_.push_back(ar);
      if (flip) {
        aspect_ratios_.push_back(1. / ar);
      }
    }
  }
  int num_priors_ = aspect_ratios_.size() * min_sizes.size();
  if (max_sizes.size() > 0) {
    for (unsigned int i = 0; i < max_sizes.size(); ++i) {
      num_priors_ += 1;
    }
  }
  auto n = 1;
  auto c = 2;
  auto tmp = input_shape[1] * input_shape[2] * num_priors_ * 4;
  auto new_tensor = xir::Tensor::create(output_tensor->get_name(), {n, c, tmp},
                                        output_tensor->get_data_type());
  new_tensor->set_attrs(output_tensor->get_attrs());
  cur->replace_output_tensor(std::move(new_tensor));
}

std::tuple<bool, std::vector<std::int32_t>> size_broadcast(
    const std::vector<std::int32_t>& in_a,
    const std::vector<std::int32_t>& in_b) {
  // new imp
  bool if_success = false;
  std::vector<std::int32_t> ret;
  std::vector<std::int32_t> in_a_local, in_b_local;
  // here make the in_a_local is longer than in_b_local
  if (in_a.size() > in_b.size()) {
    in_a_local = in_a;
    in_b_local = in_b;
  } else {
    in_a_local = in_b;
    in_b_local = in_a;
  }
  auto size_a = in_a_local.size();
  auto size_b = in_b_local.size();
  ret.resize(size_a);
  if (in_a_local == in_b_local) {
    if_success = true;
    ret = in_a_local;
  } else if ((size_a == 0) || (size_b == 0)) {
    if_success = true;
    ret = in_a_local;
  } else {
    std::int32_t idx_a = size_a - 1;
    std::int32_t idx_b = size_b - 1;
    std::int32_t idx_ret = size_a - 1;
    for (; idx_ret >= 0; idx_ret--) {
      if ((idx_a >= 0) && (idx_b >= 0)) {
        auto dim_a = in_a_local[idx_a];
        auto dim_b = in_b_local[idx_b];
        if (dim_a == 1) {
          ret[idx_ret] = dim_b;
          idx_a--;
          idx_b--;
        } else if (dim_b == 1) {
          ret[idx_ret] = dim_a;
          idx_a--;
          idx_b--;
        } else if (dim_a == dim_b) {
          ret[idx_ret] = dim_a;
          idx_a--;
          idx_b--;
        } else {
          break;
        }
      } else if ((idx_a >= 0) && (idx_b < 0)) {
        auto dim_a = in_a_local[idx_a];
        ret[idx_ret] = dim_a;
        idx_a--;
      } else if ((idx_a < 0) && (idx_b >= 0)) {
        break;
      } else {
        UNI_LOG_FATAL(XIR_UNEXPECTED_VALUE)
            << "Broadcasting the " << xir::to_string(in_a) << " and "
            << xir::to_string(in_b) << " failed.";
      }
    }
    if_success = (idx_ret < 0);
  }
  return std::make_tuple(if_success, ret);
}

void shape_infer_conv2d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 4, XIR_INVALID_ARG_OCCUR)
      << "Op" << cur->to_string()
      << ". The size of dimension of weights here is " << w_shape.size()
      << ", but the size of dimension should be 4.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  int ow = 0, oh = 0, oc = 0;
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The size of dimension of paddings here is " << tmp.size()
        << ", but the size of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::int32_t group = 1;
  if (attrs->has_attr("group")) {
    group = attrs->get_attr<std::int32_t>("group");
  }
  UNI_LOG_CHECK(w_shape.back() == in_shape.back() / group,
                XIR_INVALID_ARG_OCCUR)
      << "The number of channel of weights is " << w_shape.back()
      << ", but the number of channel of input / group is "
      << in_shape.back() / group;
  UNI_LOG_CHECK(in_shape.back() % group == 0, XIR_INVALID_ARG_OCCUR)
      << "The number of channel of input should be divisible by group.";
  UNI_LOG_CHECK(w_shape.front() % group == 0, XIR_INVALID_ARG_OCCUR)
      << "The number of weight kernel should be divisible by group.";
  oh = std::floor(1.0f *
                  (in_shape[1] + padding[2] + padding[3] -
                   (kernel[1] - 1) * dilation[1] - 1) /
                  stride[1]) +
       1;
  ow = std::floor(1.0f *
                  (in_shape[2] + padding[0] + padding[1] -
                   (kernel[0] - 1) * dilation[0] - 1) /
                  stride[0]) +
       1;
  oc = w_shape[0];
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), oh, ow, oc};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_depthwise_conv2d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 4, XIR_INVALID_ARG_OCCUR)
      << "Op" << cur->to_string()
      << ". The size of dimension of weights here is " << w_shape.size()
      << ", but the size of dimension should be 4.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  int ow = 0, oh = 0, oc = 0;
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The size of dimension of paddings here is " << tmp.size()
        << ", but the size of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  oh = std::floor(1.0f *
                  (in_shape[1] + padding[2] + padding[3] -
                   (kernel[1] - 1) * dilation[1] - 1) /
                  stride[1]) +
       1;
  ow = std::floor(1.0f *
                  (in_shape[2] + padding[0] + padding[1] -
                   (kernel[0] - 1) * dilation[0] - 1) /
                  stride[0]) +
       1;
  oc = w_shape[0] * w_shape[3];
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), oh, ow, oc};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_conv2d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::int32_t ow, oh, oc;
  // referred to
  // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp
  ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
       padding[0] - padding[1];
  oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
       padding[2] - padding[3];
  oc = w_shape[0];
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), {in->get_shape().at(0), oh, ow, oc},
                          out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_depthwise_conv2d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::int32_t ow, oh, oc;
  // referred to
  // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp
  ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
       padding[0] - padding[1];
  oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
       padding[2] - padding[3];
  oc = w_shape[0] * w_shape[3];
  auto out = cur->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), {in->get_shape().at(0), oh, ow, oc},
                          out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_conv3d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "Op" << cur->to_string()
      << ". The size of dimension of weights here is " << w_shape.size()
      << ", but the size of dimension should be 5.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  int ow = 0, oh = 0, od = 0, oc = 0;
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The size of dimension of paddings here is " << tmp.size()
        << ", but the size of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  oh = std::floor(1.0 *
                  (in_shape[1] + padding[2] + padding[3] -
                   (kernel[1] - 1) * dilation[1] - 1) /
                  stride[1]) +
       1;
  ow = std::floor(1.0 *
                  (in_shape[2] + padding[0] + padding[1] -
                   (kernel[0] - 1) * dilation[0] - 1) /
                  stride[0]) +
       1;
  od = std::floor(1.0 *
                  (in_shape[3] + padding[4] + padding[5] -
                   (kernel[2] - 1) * dilation[2] - 1) /
                  stride[2]) +
       1;
  oc = w_shape[0];
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), oh, ow, od,
                                             oc};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_depthwise_conv3d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  UNI_LOG_CHECK(w_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
      << "Op" << cur->to_string()
      << ". The size of dimension of weights here is " << w_shape.size()
      << ", but the size of dimension should be 5.";
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  int ow = 0, oh = 0, od = 0, oc = 0;
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The size of dimension of paddings here is " << tmp.size()
        << ", but the size of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  oh = std::floor(1.0 *
                  (in_shape[1] + padding[2] + padding[3] -
                   (kernel[1] - 1) * dilation[1] - 1) /
                  stride[1]) +
       1;
  ow = std::floor(1.0 *
                  (in_shape[2] + padding[0] + padding[1] -
                   (kernel[0] - 1) * dilation[0] - 1) /
                  stride[0]) +
       1;
  od = std::floor(1.0 *
                  (in_shape[3] + padding[4] + padding[5] -
                   (kernel[2] - 1) * dilation[2] - 1) /
                  stride[2]) +
       1;
  oc = w_shape[4];
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), oh, ow, od,
                                             oc};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_conv3d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::int32_t ow, oh, od, oc;
  // referred to
  // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp
  ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
       padding[0] - padding[1];
  oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
       padding[2] - padding[3];
  od = (in_shape[3] - 1) * stride[2] + dilation[2] * (kernel[2] - 1) + 1 -
       padding[4] - padding[5];
  oc = w_shape[0];
  auto out = cur->get_output_tensor();
  auto output_tensor = xir::Tensor::create(
      out->get_name(), {in->get_shape().at(0), oh, ow, od, oc},
      out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_transposed_depthwise_conv3d_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto weights = cur->get_input_tensor("weights");
  auto w_shape = weights->get_shape();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<std::int32_t> dilation = {1, 1, 1};
  if (attrs->has_attr("dilation")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("dilation");
    std::transform(tmp.begin(), tmp.end(), dilation.begin(),
                   [](auto t) { return t; });
  }
  std::vector<std::int32_t> padding = {0, 0, 0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 6, XIR_INVALID_ARG_OCCUR)
        << "The number of dimension of paddings here is " << tmp.size()
        << ", but the number of dimension should be 6.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  std::int32_t ow, oh, od, oc;
  // referred to
  // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp
  ow = (in_shape[2] - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 -
       padding[0] - padding[1];
  oh = (in_shape[1] - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 -
       padding[2] - padding[3];
  od = (in_shape[3] - 1) * stride[2] + dilation[2] * (kernel[2] - 1) + 1 -
       padding[4] - padding[5];
  oc = w_shape[4];
  auto out = cur->get_output_tensor();
  auto output_tensor = xir::Tensor::create(
      out->get_name(), {in->get_shape().at(0), oh, ow, od, oc},
      out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_const_fix(xir::Op* cur) { shape_infer_const(cur); }

void shape_infer_data_fix(xir::Op* cur) { shape_infer_data(cur); }

void shape_infer_split_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto out = cur->get_output_tensor();
  auto out_shape = in->get_shape();
  cur->replace_output_tensor(
      xir::Tensor::create(out->get_name(), out_shape, out->get_data_type()));
}

void shape_infer_eltwise_fix(xir::Op* cur) { shape_infer_eltwise(cur); }

void shape_infer_depthwise_fix(xir::Op* cur) { shape_infer_mul(cur); }

void shape_infer_pool_fix(xir::Op* cur) {
  auto in = cur->get_input_tensor("input");
  auto in_shape = in->get_shape();
  auto out = cur->get_output_tensor();
  auto attrs = cur->get_attrs();
  auto kernel = attrs->get_attr<std::vector<std::int32_t>>("kernel");
  auto stride = attrs->get_attr<std::vector<std::int32_t>>("stride");
  int ow = 0, oh = 0, oc = 0;
  std::vector<std::int32_t> padding = {0, 0, 0, 0};
  if (attrs->has_attr("pad")) {
    auto tmp = attrs->get_attr<std::vector<std::int32_t>>("pad");
    UNI_LOG_CHECK(tmp.size() == 4, XIR_INVALID_ARG_OCCUR)
        << "The size of dimension of paddings here is " << tmp.size()
        << ", but the size of dimension should be 4.";
    std::transform(tmp.begin(), tmp.end(), padding.begin(),
                   [](auto i) { return i; });
  }
  oh = std::floor(1.0f * (in_shape[1] + padding[2] + padding[3] - kernel[1]) /
                  stride[1]) +
       1;
  ow = std::floor(1.0f * (in_shape[2] + padding[0] + padding[1] - kernel[0]) /
                  stride[0]) +
       1;
  oc = in_shape[3];
  std::vector<std::int32_t> new_out_shape = {in->get_shape().at(0), oh, ow, oc};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_concat_fix(xir::Op* cur) { return shape_infer_concat(cur); }

static void shape_infer_updownsample_fix(xir::Op* cur) {
  auto op_type = cur->get_type();
  UNI_LOG_CHECK(("downsample-fix" == op_type) || ("upsample-fix" == op_type),
                XIR_UNEXPECTED_VALUE);
  auto scale = get_float_vec_from_any(cur->get_attr("scale"));
  auto scale_w = scale.at(0);
  auto scale_h = scale.at(1);
  auto input_shape = cur->get_input_tensors("input").at(0)->get_shape();
  UNI_LOG_CHECK(input_shape.size() == 4, XIR_INVALID_ARG_OCCUR)
      << cur->to_string()
      << "'s input shape can only be 4 dimensions, but now, its input shape is "
      << to_string(input_shape) << ".";
  auto output_shape = input_shape;
  auto& o_w = output_shape.at(2);
  auto& o_h = output_shape.at(1);
  if ("downsample-fix" == op_type) {
    o_w = std::ceil(o_w / ((float)scale_w));
    o_h = std::ceil(o_h / ((float)scale_h));
  } else {
    o_w *= scale_w;
    o_h *= scale_h;
  }
  auto ot = cur->get_output_tensor();
  auto o_tensor =
      Tensor::create(ot->get_name(), output_shape, ot->get_data_type());
  o_tensor->set_attrs(ot->get_attrs());
  cur->replace_output_tensor(std::move(o_tensor));
}
void shape_infer_upsample_fix(xir::Op* cur) {
  shape_infer_updownsample_fix(cur);
};
void shape_infer_downsample_fix(xir::Op* cur) {
  shape_infer_updownsample_fix(cur);
}
void shape_infer_reorg_fix(xir::Op* cur) { shape_infer_reorg(cur); }

void shape_infer_ddr_flatten_concat(xir::Op* cur) {
  auto input_ops = internal::vec_input_ops(cur->get_input_ops());
  std::vector<std::vector<std::int32_t>> flatten_input_size;
  flatten_input_size.reserve(input_ops.size());
  for (auto& input_op : input_ops) {
    auto dims = input_op->get_output_tensor()->get_shape();
    UNI_LOG_CHECK(dims.size() > 1, XIR_UNEXPECTED_VALUE)
        << cur->to_string() << "'s input " << input_op->to_string()
        << "'s input dims size is less than 2, its dims are "
        << xir::to_string(dims);
    auto flatten_size = flatten_dims(dims, 1, dims.size() - 1);
    flatten_input_size.push_back(flatten_size);
    UNI_LOG_CHECK((flatten_size.size() == 2) && (flatten_size[1] > 0),
                  XIR_UNEXPECTED_VALUE)
        << input_op->to_string()
        << "'s output tensor size is flatten failed. It's "
        << xir::to_string(dims);
  }
  std::vector<std::int32_t> output_size = flatten_input_size[0];
  auto batch = output_size[0];
  for (auto it = flatten_input_size.begin(); it != flatten_input_size.end();
       it++) {
    auto flatten_size = *it;
    UNI_LOG_CHECK(flatten_size[0] == batch, XIR_UNEXPECTED_VALUE)
        << cur->to_string() << "'s inputs batch size error.";
    output_size[1] += flatten_size[1];
  }
  auto output_tensor_ori = cur->get_output_tensor();
  auto output_tensor_new =
      xir::Tensor::create(output_tensor_ori->get_name(), output_size,
                          output_tensor_ori->get_data_type());
  output_tensor_new->set_attrs(output_tensor_ori->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor_new));
}

void shape_infer_ddr_flatten_concat_fix(xir::Op* cur) {
  shape_infer_ddr_flatten_concat(cur);
}

void shape_infer_tile_fix(xir::Op* cur) { shape_infer_gstiling(cur); }

void shape_infer_pad_fix(xir::Op* cur) { shape_infer_pad(cur); }

void shape_infer_reshape_fix(xir::Op* cur) {
  auto input_element_num = cur->get_input_tensor("input", 0)->get_element_num();
  auto output_tensor = cur->get_output_tensor();
  auto output_dims = cur->get_attr<std::vector<std::int32_t>>("shape");
  std::int32_t output_element_num{1};
  std::for_each(output_dims.begin(), output_dims.end(),
                [&](const std::int32_t& dim) { output_element_num *= dim; });
  UNI_LOG_CHECK(input_element_num == output_element_num, XIR_VALUE_UNMATCH)
      << cur->to_string()
      << "'s input and output size is unmatch. The input size is "
      << xir::to_string(cur->get_input_tensor("input", 0)->get_shape())
      << " and the output size is " << xir::to_string(output_dims);
  auto output_tensor_new = xir::Tensor::create(
      output_tensor->get_name(), output_dims, output_tensor->get_data_type());
  cur->replace_output_tensor(std::move(output_tensor_new));
}

void shape_infer_correlation2d_elemwise(xir::Op* cur) {
  auto pad_size = cur->get_attr<std::int32_t>("pad_size");
  auto ins = cur->get_input_tensors("input");
  auto in_shape = ins[0]->get_shape();
  auto out = cur->get_output_tensor();
  int od = 0;
  od = std::floor((2 * pad_size + 1) * (2 * pad_size + 1));

  std::vector<std::int32_t> new_out_shape = {in_shape[0], in_shape[1],
                                             in_shape[2], od, in_shape[3]};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_correlation1d_elemwise(xir::Op* cur) {
  auto pad_size = cur->get_attr<std::int32_t>("pad_size");
  auto ins = cur->get_input_tensors("input");
  auto in_shape = ins[0]->get_shape();
  auto out = cur->get_output_tensor();
  int od = pad_size + 1;

  std::vector<std::int32_t> new_out_shape = {in_shape[0], in_shape[1],
                                             in_shape[2], od, in_shape[3]};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

void shape_infer_cost_volume(xir::Op* cur) {
  auto maxdisp = cur->get_attr<std::int32_t>("maxdisp");
  auto ins = cur->get_input_tensors("input");
  auto in_shape = ins[0]->get_shape();
  auto out = cur->get_output_tensor();
  int od = 0;
  od = std::floor(maxdisp / 4);
  std::vector<std::int32_t> new_out_shape = {in_shape[0], in_shape[1],
                                             in_shape[2], od, in_shape[3] * 2};
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  cur->replace_output_tensor(std::move(output_tensor));
}

// helper function
std::vector<std::int32_t> flatten_dims(const std::vector<std::int32_t>& dims,
                                       const std::int32_t& start,
                                       const std::int32_t& end) {
  UNI_LOG_CHECK((start > 0) && (start < (std::int32_t)dims.size()),
                XIR_UNEXPECTED_VALUE)
      << "Flatten start idex error.";
  UNI_LOG_CHECK((end > 0) && (end < (std::int32_t)dims.size()),
                XIR_UNEXPECTED_VALUE)
      << "Flatten start idex error.";
  std::vector<std::int32_t> ret;
  for (std::int32_t idx = 0; idx < (std::int32_t)dims.size(); idx++) {
    if ((idx >= start) && (idx <= end)) {
      *ret.rbegin() += dims[idx];
    } else {
      ret.push_back(dims[idx]);
    }
  }
  return ret;
}

}  // namespace xir
