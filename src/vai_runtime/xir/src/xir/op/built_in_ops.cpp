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
#include "xir/op/built_in_ops.hpp"

#include <glog/logging.h>

#include <functional>

#include "xir/op/built_in_ops.hpp"
#include "xir/op/shape_inference.hpp"

using namespace xir;
using std::string;
#define XIR_REGISTER_BUILT_IN_OP(OP)                                           \
  static BuiltInOPsRegister BUILT_IN_OPDEFS_##OP(OP);

class BuiltInOPsRegister {
 public:
  BuiltInOPsRegister(const xir::OpDef& def) { add_op_def(def); }

  static void add_op_def(const xir::OpDef& def) {
    BUILT_IN_OPS_.push_back(std::move(def));
  }

  static std::vector<xir::OpDef> BUILT_IN_OPS_;
};

std::vector<xir::OpDef> BuiltInOPsRegister::BUILT_IN_OPS_ = {};

void register_built_in_ops(xir::OpDefFactory* self) {
  std::for_each(BuiltInOPsRegister::BUILT_IN_OPS_.begin(),
                BuiltInOPsRegister::BUILT_IN_OPS_.end(),
                [self](const xir::OpDef& def) { self->register_h(def); });
}

namespace xir {

std::function<void(xir::OpDef& op_def)> SrcOpGenerator() {
  return [=](xir::OpDef& op_def) {
    auto tensor_shape = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "shape", AttrDef::REQUIRED, 0,
        "`Datatype`: `vector<int>`\n\n"
        "The shape of the output tensor");
    auto tensor_data_type = xir::AttrDefBuilder<std::string>::build(
        "data_type", AttrDef::REQUIRED,
        "`Datatype`: `string`\n\n"
        "The data type of the data of output feature maps, "
        "we use FLOAT32 as the default.");
    op_def
        .add_attr(tensor_shape)  //
        .add_attr(tensor_data_type);
  };
}

auto data = xir::OpDef("data")
                .inherit_from(SrcOpGenerator())
                .set_annotation(
                    "A placeholder which stores the float-point input data, \n"
                    "data operator would always be fed by users.")
                .set_shape_infer(xir::shape_infer_data);

auto const_op =
    xir::OpDef("const")
        .inherit_from(SrcOpGenerator())
        .add_attr(xir::AttrDefBuilder<std::vector<char>>::build(
            "data", AttrDef::REQUIRED, 0,
            "Constant values stored in this operator, \n"
            "float-point data in vector<char>.\n"))
        .set_annotation(
            "A placeholder which stores the parameters, \n"
            "such as weights, bias, etc.\n\n"
            "How to transform float-point values into vector<char>: \n\n"
            "    const std::vector<float> float_data = {...};\n"
            "    std::vector<char> data;\n"
            "    for (uint outer = 0; outer < float_data.size(); outer++)\n"
            "      for (auto inner = 0; inner < sizeof(float) / sizeof(char); "
            "inner++)\n"
            "        data.push_back(*(reinterpret_cast<char*>(&float_data) + "
            "inner));\n")
        .set_shape_infer(xir::shape_infer_const);

auto data_fix =
    xir::OpDef("data-fix")
        .inherit_from(SrcOpGenerator())
        .set_annotation(
            "A placeholder which stores the fixed-point input data, \n"
            "data operator would always be fed by users.")
        .set_shape_infer(xir::shape_infer_data_fix);

auto const_fix = xir::OpDef("const-fix")
                     .inherit_from(SrcOpGenerator())
                     .add_attr(xir::AttrDefBuilder<std::vector<char>>::build(
                         "data", AttrDef::REQUIRED, 0,
                         "Constant values stored in this operator, \n"
                         "fixed-point data in vector<char>.\n"))
                     .set_annotation(
                         "A placeholder which stores the parameters, \n"
                         "such as fixed-point weights, bias, etc.")
                     .set_shape_infer(xir::shape_infer_const_fix);

XIR_REGISTER_BUILT_IN_OP(data);
XIR_REGISTER_BUILT_IN_OP(const_op);
XIR_REGISTER_BUILT_IN_OP(data_fix);
XIR_REGISTER_BUILT_IN_OP(const_fix);

// Random Generator
auto random_standard_normal =
    xir::OpDef("random_standard_normal")  //
        .inherit_from(SrcOpGenerator())   //
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "seed", AttrDef::OPTIONAL,
            "`DataType`: `int`\n\nDefaults to 0. If either `seed` or `seed2` "
            "are set to be non-zero, the random number generator is seeded by "
            "the given seed. Otherwise, it is seeded by a random seed.",
            0))
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "seed2", AttrDef::OPTIONAL,
            "`DataType`: `int`\n\nDefaults to 0. A second seed to avoid seed "
            "collision.",
            0))
        .set_annotation(
            "Outputs random values from a normal distribution.\n\nAnd the "
            "generated values will have mean 0 and standard deviation 1.")
        .set_shape_infer(xir::shape_infer_data);

XIR_REGISTER_BUILT_IN_OP(random_standard_normal);

std::function<void(xir::OpDef&)> Conv1dOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "An input tensor with shape "
                               "`[batch, in_length, in_channels]`."};
    auto weights =
        xir::OpArgDef{"weights", OpArgDef::REQUIRED, T,
                      "A filter tensor with shape "
                      "`[output_channels, kernel_length, in_channels]`."};
    auto bias = xir::OpArgDef{"bias", OpArgDef::OPTIONAL, T,
                              "A bias tensor with shape "
                              "`[output_channels]`."};
    auto kernel = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "kernel", AttrDef::REQUIRED, 1,
        "`Datatype`: `vector<int>`\n\n"
        "The kernel sizes of the filter. "
        "The value must be: `{kernel_length}`.");
    auto stride = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "stride", AttrDef::REQUIRED, 1,
        "`Datatype`: `vector<int>`\n\n"
        "The strides of the filter. "
        "The value must be: `{stride_length}`.");
    auto dilation = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "dilation", AttrDef::OPTIONAL, 1,
        "`Datatype`: `vector<int>`\n\n"
        "The dilation of the filter. "
        "The value must be: `{dilation_length}`, "
        "The dilation in the batch or depth are 1 in default.",
        {1});
    auto pad_mode = xir::AttrDefBuilder<std::string>::build(
        "pad_mode", AttrDef::REQUIRED,
        "`Datatype`: `string`\n\n"
        "We support 4 padding mode: `FLOOR, CEIL, SAME, VALID`. "
        "For example, when you parsing models from other frameworks, "
        "`caffe, pytorch->\"FLOOR\", tensorflow->\"SAME\" or \"VALID\"`.");
    auto pad = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "pad", AttrDef::OPTIONAL, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The padding sizes of input feature maps. "
        "The value must be `{left, right}`.\n\n"
        "For transposed convolutions, the padding here denotes the "
        "`{kernel_size - 1 - actual_padding}`."
        "This is an optional attribute, when the pad_mode is SAME or VALID, "
        "you don't need to specify this attribute.",
        {0, 0});
    op_def.add_input_arg(input)
        .add_input_arg(weights)
        .add_input_arg(bias)
        .add_attr(kernel)
        .add_attr(stride)
        .add_attr(dilation)
        .add_attr(pad_mode)
        .add_attr(pad);
  };
}

auto conv1d = xir::OpDef("conv1d")
                  .inherit_from(Conv1dOpDefGenerator(xir::DataType::FLOAT))
                  .set_annotation(
                      "1D convolution.\n\n"
                      "    output[batch, ol, oc] =\n"
                      "        sum_{kl, ic} input[batch, strides[0] * ol + "
                      "kl, ic] *\n"
                      "                        filter[oc, kl, ic]\n"
                      "(1). if pad_mode == \"`FLOOR`\":\n\n"
                      "    output_shape = floor((input_shape + pad - (kernel - "
                      "1) * dilation + 1) / stride) + 1\n"
                      "(2). if pad_mode == \"`CEIL`\":\n\n"
                      "    output_shape = ceil((input_shape + pad - (kernel - "
                      "1) * dilation + 1) / stride) + 1\n"
                      "(3). if pad_mode == \"`SAME`\":\n\n"
                      "    output_shape = ceil((input_shape + pad) / stride)\n"
                      "(4). if pad_mode == \"`VALID`\":\n\n"
                      "    output_shape = ceil((input_shape + pad - (kernel - "
                      "1) * dilation) / stride)\n")
                  .set_shape_infer(xir::shape_infer_conv1d);

XIR_REGISTER_BUILT_IN_OP(conv1d);

std::function<void(xir::OpDef&)> ConvOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "An input tensor with shape "
                               "`[batch, in_height, in_width, in_channels]`."};
    auto weights = xir::OpArgDef{
        "weights", OpArgDef::REQUIRED, T,
        "A filter tensor with shape "
        "`[output_channels, kernel_height, kernel_width, in_channels]`."};
    auto bias = xir::OpArgDef{"bias", OpArgDef::OPTIONAL, T,
                              "A bias tensor with shape "
                              "`[output_channels]`."};
    auto kernel = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "kernel", AttrDef::REQUIRED, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The kernel sizes of the filter. "
        "The value must be: `{kernel_width, kernel_height}`.");
    auto stride = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "stride", AttrDef::REQUIRED, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The strides of the filter. "
        "The value must be: `{stride_width, stride_height}`.");
    auto dilation = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "dilation", AttrDef::OPTIONAL, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The dilation of the filter. "
        "The value must be: `{dilation_width, dilation_height}`, "
        "The dilation in the batch or depth are 1 in default.",
        {1, 1});
    auto pad_mode = xir::AttrDefBuilder<std::string>::build(
        "pad_mode", AttrDef::REQUIRED,
        "`Datatype`: `string`\n\n"
        "We support 4 padding mode: `FLOOR, CEIL, SAME, VALID`. "
        "For example, when you parsing models from other frameworks, "
        "`caffe, pytorch->\"FLOOR\", tensorflow->\"SAME\" or \"VALID\"`.");
    auto pad = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "pad", AttrDef::OPTIONAL, 4,
        "`Datatype`: `vector<int>`\n\n"
        "The padding sizes of input feature maps. "
        "The value must be `{left, right, top, bottom}`.\n\n"
        "For transposed convolutions, the padding here denotes the "
        "`{kernel_size - 1 - actual_padding}`."
        "This is an optional attribute, when the pad_mode is SAME or VALID, "
        "you don't need to specify this attribute.",
        {0, 0, 0, 0});
    op_def.add_input_arg(input)
        .add_input_arg(weights)
        .add_input_arg(bias)
        .add_attr(kernel)
        .add_attr(stride)
        .add_attr(dilation)
        .add_attr(pad_mode)
        .add_attr(pad);
  };
}

auto conv2d =
    xir::OpDef("conv2d")
        .inherit_from(ConvOpDefGenerator(xir::DataType::FLOAT))
        .add_attr(xir::AttrDefBuilder<int32_t>::build(
            "group", AttrDef::OPTIONAL,
            "`Datatype`: `int`\n\n"
            "controls the connections between inputs and outputs."
            "in_channels and out_channels must both be divisible by groups",
            1))
        .set_annotation(
            "2D convolution.\n\n"
            "    output[batch, oh, ow, oc] =\n"
            "        sum_{kw, kh, ic} input[batch, strides[1] * oh + "
            "kh, strides[0] * ow + kw, ic] *\n"
            "                        filter[oc, kh, kw, ic]\n"
            "(1). if pad_mode == \"`FLOOR`\":\n\n"
            "    output_shape = floor((input_shape + pad - (kernel - "
            "1) * dilation + 1) / stride) + 1\n"
            "(2). if pad_mode == \"`CEIL`\":\n\n"
            "    output_shape = ceil((input_shape + pad - (kernel - "
            "1) * dilation + 1) / stride) + 1\n"
            "(3). if pad_mode == \"`SAME`\":\n\n"
            "    output_shape = ceil((input_shape + pad) / stride)\n"
            "(4). if pad_mode == \"`VALID`\":\n\n"
            "    output_shape = ceil((input_shape + pad - (kernel - "
            "1) * dilation) / stride)\n")
        .set_shape_infer(xir::shape_infer_conv2d);

auto depthwiseconv2d =
    xir::OpDef("depthwise-conv2d")
        .inherit_from(ConvOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Depth-wise 2D convolution.\n\n"
            "    output[batch, oh, ow, b * c] =\n"
            "        sum_{kw, kh} input[batch, strides[1] * oh + kh, "
            "strides[0] * ow + kw, c] *\n"
            "                        filter[b, kh, kw, c]\n"
            "(1). if pad_mode == \"`FLOOR`\":\n\n"
            "    output_shape = floor((input_shape + pad - (kernel - 1) * "
            "dilation + 1) / stride) + 1\n"
            "(2). if pad_mode == \"`CEIL`\":\n\n"
            "    output_shape = ceil((input_shape + pad - (kernel - 1) * "
            "dilation + 1) / stride) + 1\n"
            "(3). if pad_mode == \"`SAME`\":\n\n"
            "    output_shape = ceil((input_shape + pad) / stride)\n"
            "(4). if pad_mode == \"`VALID`\":\n\n"
            "    output_shape = ceil((input_shape + pad - (kernel - 1) * "
            "dilation) / stride)\n"
            "For example, in Tensorflow, tf.nn.depthwise_conv2d is:\n\n"
            "    output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}\n"
            "        filter[di, dj, k, q] * input[b, stride[1] * i + rate[0] * "
            "di,\n"
            "                                        stride[2] * j + rate[1] * "
            "dk, k]\n"
            "Given a 4D input tensor ('NHWC' or 'NCHW' data formats) and a "
            "filter tensor of shape "
            "[filter_height, filter_width, in_channels, channel_multiplier]"
            "if we want to transform tf.nn.depthwise_conv2d into XIR "
            "depthwise-conv2d, then in XIR\n\n"
            "    output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}\n"
            "        filter[q, di, dj, k] * input[b, stride[1] * i + rate[0] * "
            "di,\n"
            "                                        stride[0] * j + rate[1] * "
            "dk, k]\n"
            "In another example, for convolution in caffe, if the attribute "
            "`group` "
            "euqals to the input channels of the input feature maps, then this "
            "convolution"
            "can be transformed into a XIR depthwise-conv2d.")
        .set_shape_infer(xir::shape_infer_depthwise_conv2d)
        .add_constraint([](xir::Op* op) {
          auto weights = op->get_input_tensor("weights");
          auto w_shape = weights->get_shape();
          auto in = op->get_input_tensor("input");
          auto in_shape = in->get_shape();
          UNI_LOG_CHECK(w_shape[3] == in_shape[3], XIR_INVALID_ARG_OCCUR)
              << "The channel of weights should be equal to the channel of "
                 "input in "
                 "depthwise conv2d";
        });

auto transposed_conv2d =
    xir::OpDef("transposed-conv2d")
        .inherit_from(ConvOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "2D transposed convolution, our equivalent implementations:\n"
            "Firstly, we dilate the input feature maps by `stride`:\n\n"
            "    dilated_input[batch, h, w, c] =\n"
            "        ((h mod stride[1] == 0) && (w mod stride[0] == 0))\n"
            "        ? input[batch, h / stride[1], h / stride[0], ic]\n"
            "        : 0\n"
            "Secondly, we do 2D-convolution on the feature maps:\n\n"
            "    output[batch, oh, ow, oc] =\n"
            "        sum_{kw, kh, ic} dilated_input[batch, oh + kh, ow + kw, "
            "ic] *\n"
            "                        filter[oc, kh, kw, ic]\n"
            "If pad is set:\n\n"
            "    actual_padding[n] = kernel (h or w) - 1 - pad[n]\n"
            "    padded_dilated_input[batch, h - actual_padding[2], w - "
            "actual_padding[0], c] =\n"
            "        dilated_input[batch, h, w, c]\n"
            "    padded_dilated_input[batch, 0 : actual_padding[2], 0 : "
            "actual_padding[0], c] = 0\n"
            "    padded_dilated_input[batch, h + actual_padding[2] : h + "
            "actual_padding[2] + actual_padding[3]\n"
            "                         w + actual_padding[0] : w + "
            "actual_padding[0] + actual_padding[1], c] = 0\n"
            "And here is how to calculate the output shape according to the "
            "attributes:\n\n"
            "(1). if pad_mode == \"`FLOOR`\":\n\n"
            "    output_shape = (in_shape - 1) * stride + dilation * (kernel - "
            "1) + 1 - pad\n"
            "(2). if pad_mode == \"`CEIL`\":\n\n"
            "    output_shape = (in_shape - 1) * stride + dilation * (kernel - "
            "1) + 1 - pad\n"
            "(3). if pad_mode == \"`SAME`\":\n\n"
            "    output_shape = in_shape * stride\n"
            "(4). if pad_mode == \"`VALID`\":\n\n"
            "    output_shape = (in_shape - 1) * stride + kernel\n"
            "For example, to transform a conv2d_transpose or "
            "Conv2DBackpropInput in Tensorflow into XIR:\n"
            "we only need to change the filter in tensorflow into XIR "
            "format.\n\n"
            "(1). flip the filter along the dimension of width and height,\n\n"
            "(2). transpose the filter into `{oc, h, w, ic}`, ic equals the "
            "channel of "
            "input feature maps and oc equals to the channel of output feature "
            "maps.")
        .set_shape_infer(xir::shape_infer_transposed_conv2d);

auto transposed_depthwise_conv2d =
    xir::OpDef("transposed-depthwise-conv2d")
        .inherit_from(ConvOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "2D depth-wise transposed convolution, our equivalent "
            "implementations::\n"
            "Firstly, we dilate the input feature maps by `stride`:\n\n"
            "    dilated_input[batch, h, w, c] =\n"
            "        ((h mod stride[1] == 0) && (w mod stride[0] == 0))\n"
            "        ? input[batch, h / stride[1], h / stride[0], ic]\n"
            "        : 0\n"
            "Secondly, we do 2D-convolution on the feature maps:\n\n"
            "    output[batch, oh, ow, b * c] =\n"
            "        sum_{kw, kh} dilated_input[batch, oh + kh, ow + kw, c] *\n"
            "                     filter[b, kh, kw, c]\n"
            "If pad is set:\n\n"
            "    actual_padding[n] = kernel (h or w) - 1 - pad[n]\n"
            "    padded_dilated_input[batch, h - actual_padding[2], w - "
            "actual_padding[0], c] =\n"
            "        dilated_input[batch, h, w, c]\n"
            "    padded_dilated_input[batch, 0 : actual_padding[2], 0 : "
            "actual_padding[0], c] = 0\n"
            "    padded_dilated_input[batch, h + actual_padding[2] : h + "
            "actual_padding[2] + actual_padding[3]\n"
            "                         w + actual_padding[0] : w + "
            "actual_padding[0] + actual_padding[1], c] = 0\n"
            "And here is how to calculate the output shape according to the "
            "attributes:\n\n"
            "(1). if pad_mode == \"`FLOOR`\":\n\n"
            "    output_shape = (in_shape - 1) * stride + dilation * (kernel - "
            "1) + 1 - pad\n"
            "(2). if pad_mode == \"`CEIL`\":\n\n"
            "    output_shape = (in_shape - 1) * stride + dilation * (kernel - "
            "1) + 1 - pad\n"
            "(3). if pad_mode == \"`SAME`\":\n\n"
            "    output_shape = in_shape * stride\n"
            "(4). if pad_mode == \"`VALID`\":\n\n"
            "    output_shape = (in_shape - 1) * stride + kernel\n")
        .set_shape_infer(xir::shape_infer_transposed_depthwise_conv2d);

XIR_REGISTER_BUILT_IN_OP(conv2d);
XIR_REGISTER_BUILT_IN_OP(depthwiseconv2d);
XIR_REGISTER_BUILT_IN_OP(transposed_conv2d);
XIR_REGISTER_BUILT_IN_OP(transposed_depthwise_conv2d);

std::function<void(xir::OpDef&)> Conv3dOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input =
        xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                      "An input tensor with shape "
                      "`[batch, in_height, in_width, depth, in_channels]`."};
    auto weights = xir::OpArgDef{"weights", OpArgDef::REQUIRED, T,
                                 "A filter tensor with shape "
                                 "`[output_channels, kernel_height, "
                                 "kernel_width, kernel_depth, in_channels]`."};
    auto bias = xir::OpArgDef{"bias", OpArgDef::OPTIONAL, T,
                              "A bias tensor with shape "
                              "`[output_channels]`."};
    auto kernel = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "kernel", AttrDef::REQUIRED, 3,
        "`Datatype`: `vector<int>`\n\n"
        "The kernel sizes of the filter. "
        "The value must be: `{kernel_width, kernel_height, kernel_depth}`.");
    auto stride = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "stride", AttrDef::REQUIRED, 3,
        "`Datatype`: `vector<int>`\n\n"
        "The strides of the filter. "
        "The value must be: `{stride_width, stride_height, stride_depth}`.");
    auto dilation = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "dilation", AttrDef::OPTIONAL, 3,
        "`Datatype`: `vector<int>`\n\n"
        "The dilation of the filter. "
        "The value must be: `{dilation_width, dilation_height, "
        "dilation_depth}`, "
        "The dilation in the batch or depth are 1 in default.",
        {1, 1, 1});
    auto pad_mode = xir::AttrDefBuilder<string>::build(
        "pad_mode", AttrDef::REQUIRED,
        "`Datatype`: `string`\n\n"
        "We support 4 padding mode: `FLOOR, CEIL, SAME, VALID`. "
        "For example, when you parsing models from other frameworks, "
        "`caffe, pytorch->\"FLOOR\", tensorflow->\"SAME\" or \"VALID\"`.");
    auto pad = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "pad", AttrDef::OPTIONAL, 6,
        "`Datatype`: `vector<int>`\n\n"
        "The padding sizes of input feature maps. "
        "The value must be `{left, right, top, bottom, near, far}`.\n\n"
        "For transposed convolutions, the padding here denotes the "
        "`{kernel_size - 1 - actual_padding}`."
        "This is an optional attribute, when the pad_mode is SAME or VALID, "
        "you don't need to specify this attribute.",
        {0, 0, 0, 0, 0, 0});
    op_def.add_input_arg(input)
        .add_input_arg(weights)
        .add_input_arg(bias)
        .add_attr(kernel)
        .add_attr(stride)
        .add_attr(dilation)
        .add_attr(pad_mode)
        .add_attr(pad);
  };
}

auto conv3d = xir::OpDef("conv3d")
                  .inherit_from(Conv3dOpDefGenerator(xir::DataType::FLOAT))
                  .set_annotation("3D convolution.\n\n")
                  .set_shape_infer(xir::shape_infer_conv3d);

auto transposed_conv3d =
    xir::OpDef("transposed-conv3d")
        .inherit_from(Conv3dOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation("Transposed 3D convolution.\n\n")
        .set_shape_infer(xir::shape_infer_transposed_conv3d);

auto depthwise_conv3d =
    xir::OpDef("depthwise-conv3d")
        .inherit_from(Conv3dOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation("Depth-wise 3D convolution.\n\n")
        .set_shape_infer(xir::shape_infer_depthwise_conv3d)
        .add_constraint([](xir::Op* op) {
          auto weights = op->get_input_tensor("weights");
          auto w_shape = weights->get_shape();
          auto in = op->get_input_tensor("input");
          auto in_shape = in->get_shape();
          UNI_LOG_CHECK(w_shape[4] == in_shape[4], XIR_INVALID_ARG_OCCUR)
              << "The channel of weights should be equal to the channel of "
                 "input in "
                 "depthwise conv3d";
        });

auto transposed_depthwise_conv3d =
    xir::OpDef("transposed-depthwise-conv3d")
        .inherit_from(Conv3dOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation("Transposed depth-wise 3D convolution.\n\n")
        .set_shape_infer(xir::shape_infer_transposed_depthwise_conv3d)
        .add_constraint([](xir::Op* op) {
          auto weights = op->get_input_tensor("weights");
          auto w_shape = weights->get_shape();
          auto in = op->get_input_tensor("input");
          auto in_shape = in->get_shape();
          UNI_LOG_CHECK(w_shape[4] == in_shape[4], XIR_INVALID_ARG_OCCUR)
              << "The channel of weights should be equal to the channel of "
                 "input in "
                 "depthwise conv3d";
        });

XIR_REGISTER_BUILT_IN_OP(conv3d);
XIR_REGISTER_BUILT_IN_OP(depthwise_conv3d);
XIR_REGISTER_BUILT_IN_OP(transposed_conv3d);
XIR_REGISTER_BUILT_IN_OP(transposed_depthwise_conv3d);

std::function<void(xir::OpDef&)> FixedConvOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "An input tensor with shape "
                               "`[batch, in_height, in_width, in_channels]`."};
    auto weights = xir::OpArgDef{
        "weights", OpArgDef::REQUIRED, T,
        "A filter tensor with shape "
        "`[output_channels, kernel_height, kernel_width, in_channels]`."};
    auto bias = xir::OpArgDef{"bias", OpArgDef::OPTIONAL, T,
                              "A bias tensor with shape "
                              "`[output_channels]`."};
    auto kernel = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "kernel", AttrDef::REQUIRED, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The kernel sizes of the filter. "
        "The value must be: `{kernel_width, kernel_height}`.");
    auto stride = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "stride", AttrDef::REQUIRED, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The strides of the filter. "
        "The value must be: `{stride_width, stride_height}`.");
    auto dilation = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "dilation", AttrDef::OPTIONAL, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The dilation of the filter. "
        "The value must be: `{dilation_width, dilation_height}`, "
        "The dilation in the batch or depth are 1 in default.",
        {1, 1});
    auto pad = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "pad", AttrDef::OPTIONAL, 4,
        "`Datatype`: `vector<int>`\n\n"
        "The padding sizes of input feature maps. "
        "The value must be `{left, right, top, bottom}`.\n\n"
        "For transposed convolutions, the padding here denotes the "
        "`{kernel_size - 1 - actual_padding}`.",
        {0, 0, 0, 0});
    auto nonlinear = xir::AttrDefBuilder<std::string>::build(
        "nonlinear", AttrDef::OPTIONAL,
        "`Datatype`: `string`\n\n"
        "nonlinear type, \"NONE\", \"RELU\", \"PRELU\", "
        "\"LEAKYRELU\",\"RELU6\",\"HSIGMOID\",\"HSWISH\".",
        "");
    auto hsigmoid_in =
        xir::AttrDefBuilder<int>::build("hsigmoid_in", AttrDef::OPTIONAL,
                                        "`Datatype`: `int`\n\n"
                                        "fix_point of hsigmoid",
                                        -128);
    auto shift_hsigmoid =
        xir::AttrDefBuilder<int>::build("shift_hsigmoid", AttrDef::OPTIONAL,
                                        "`Datatype`: `int`\n\n"
                                        "shift value after hsigmoid",
                                        -128);
    auto shift_hswish =
        xir::AttrDefBuilder<int>::build("shift_hswish", AttrDef::OPTIONAL,
                                        "`Datatype`: `int`\n\n"
                                        "shift value after hswish",
                                        -128);
    op_def.add_input_arg(input)
        .add_input_arg(weights)
        .add_input_arg(bias)
        .add_attr(kernel)
        .add_attr(stride)
        .add_attr(dilation)
        .add_attr(pad)
        .add_attr(nonlinear)
        .add_attr(hsigmoid_in)
        .add_attr(shift_hsigmoid)
        .add_attr(shift_hswish)
        .add_constraint([](xir::Op* op) {
          if (op->has_attr("nonlinear")) {
            if (op->get_attr<std::string>("nonlinear") == "HSIGMOID" ||
                op->get_attr<std::string>("nonlinear") == "HSWISH") {
              UNI_LOG_CHECK(op->get_attr<int>("hsigmoid_in") != -128 &&
                                op->get_attr<int>("shift_hsigmoid") != -128,
                            XIR_INVALID_ARG_OCCUR)
                  << "the activation type is "
                  << op->get_attr<std::string>("nonlinear")
                  << " but you do not set the shift value for this operation.";
              if (op->get_attr<std::string>("nonlinear") == "HSWISH")
                UNI_LOG_CHECK(op->get_attr<int>("shift_hswish") != -128,
                              XIR_INVALID_ARG_OCCUR)
                    << "the activation type is "
                    << op->get_attr<std::string>("nonlinear")
                    << " but you do not set the shift_hswish for this "
                       "operation.";
              UNI_LOG_CHECK(
                  op->has_attr("shift_bias") && op->has_attr("shift_cut"),
                  XIR_INVALID_ARG_OCCUR)
                  << "the activation type is "
                  << op->get_attr<std::string>("nonlinear")
                  << " you need to set shift_bias and shift_cut for it..";
            }
          }
        });
  };
}

auto conv2d_fix =
    xir::OpDef("conv2d-fix")
        .inherit_from(FixedConvOpDefGenerator(xir::DataType::XINT))
        .add_attr(xir::AttrDefBuilder<int32_t>::build(
            "group", AttrDef::OPTIONAL,
            "`Datatype`: `int`\n\n"
            "controls the connections between inputs and outputs."
            "in_channels and out_channels must both be divisible by groups",
            1))
        .set_shape_infer(xir::shape_infer_conv2d_fix);

auto depthwise_conv2d_fix =
    xir::OpDef("depthwise-conv2d-fix")
        .inherit_from(FixedConvOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_depthwise_conv2d_fix);

auto transposed_conv2d_fix =
    xir::OpDef("transposed-conv2d-fix")
        .inherit_from(FixedConvOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_transposed_conv2d_fix);

auto transposed_depthwise_conv2d_fix =
    xir::OpDef("transposed-depthwise-conv2d-fix")
        .inherit_from(FixedConvOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_transposed_depthwise_conv2d_fix);

XIR_REGISTER_BUILT_IN_OP(conv2d_fix);
XIR_REGISTER_BUILT_IN_OP(depthwise_conv2d_fix);
XIR_REGISTER_BUILT_IN_OP(transposed_conv2d_fix);
XIR_REGISTER_BUILT_IN_OP(transposed_depthwise_conv2d_fix);

std::function<void(xir::OpDef&)> FixedConv3dOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input =
        xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                      "An input tensor with shape "
                      "`[batch, in_height, in_width, in_depth, in_channels]`."};
    auto weights = xir::OpArgDef{"weights", OpArgDef::REQUIRED, T,
                                 "A filter tensor with shape "
                                 "`[output_channels, kernel_height, "
                                 "kernel_width, kernel_depth, in_channels]`."};
    auto bias = xir::OpArgDef{"bias", OpArgDef::OPTIONAL, T,
                              "A bias tensor with shape "
                              "`[output_channels]`."};
    auto kernel = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "kernel", AttrDef::REQUIRED, 3,
        "`Datatype`: `vector<int>`\n\n"
        "The kernel sizes of the filter. "
        "The value must be: `{kernel_width, kernel_height}`.");
    auto stride = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "stride", AttrDef::REQUIRED, 3,
        "`Datatype`: `vector<int>`\n\n"
        "The strides of the filter. "
        "The value must be: `{stride_width, stride_height}`.");
    auto dilation = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "dilation", AttrDef::OPTIONAL, 3,
        "`Datatype`: `vector<int>`\n\n"
        "The dilation of the filter. "
        "The value must be: `{dilation_width, dilation_height}`, "
        "The dilation in the batch or depth are 1 in default.",
        {1, 1, 1});
    auto pad = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "pad", AttrDef::OPTIONAL, 6,
        "`Datatype`: `vector<int>`\n\n"
        "The padding sizes of input feature maps. "
        "The value must be `{left, right, top, bottom}`.\n\n"
        "For transposed convolutions, the padding here denotes the "
        "`{kernel_size - 1 - actual_padding}`."
        "This is an optional attribute, when the pad_mode is SAME or VALID, "
        "you don't need to specify this attribute.",
        {0, 0, 0, 0, 0, 0});
    auto nonlinear = xir::AttrDefBuilder<std::string>::build(
        "nonlinear", AttrDef::OPTIONAL,
        "`Datatype`: `string`\n\n"
        "nonlinear type, \"NONE\", \"RELU\", \"PRELU\", "
        "\"LEAKYRELU\",\"RELU6\",\"HSIGMOID\",\"HSWISH\".",
        "");
    auto hsigmoid_in =
        xir::AttrDefBuilder<int>::build("hsigmoid_in", AttrDef::OPTIONAL,
                                        "`Datatype`: `int`\n\n"
                                        "fix_point of hsigmoid",
                                        -128);
    auto shift_hsigmoid =
        xir::AttrDefBuilder<int>::build("shift_hsigmoid", AttrDef::OPTIONAL,
                                        "`Datatype`: `int`\n\n"
                                        "shift value after hsigmoid",
                                        -128);
    auto shift_hswish =
        xir::AttrDefBuilder<int>::build("shift_hswish", AttrDef::OPTIONAL,
                                        "`Datatype`: `int`\n\n"
                                        "shift value after hswish",
                                        -128);
    op_def.add_input_arg(input)
        .add_input_arg(weights)
        .add_input_arg(bias)
        .add_attr(kernel)
        .add_attr(stride)
        .add_attr(dilation)
        .add_attr(pad)
        .add_attr(nonlinear)
        .add_attr(hsigmoid_in)
        .add_attr(shift_hsigmoid)
        .add_attr(shift_hswish)
        .add_constraint([](xir::Op* op) {
          if (op->has_attr("nonlinear")) {
            if (op->get_attr<std::string>("nonlinear") == "HSIGMOID" ||
                op->get_attr<std::string>("nonlinear") == "HSWISH") {
              UNI_LOG_CHECK(op->get_attr<int>("hsigmoid_in") != -128 &&
                                op->get_attr<int>("shift_hsigmoid") != -128,
                            XIR_INVALID_ARG_OCCUR)
                  << "the activation type is "
                  << op->get_attr<std::string>("nonlinear")
                  << " but you do not set the shift value for this operation.";
              if (op->get_attr<std::string>("nonlinear") == "HSWISH")
                UNI_LOG_CHECK(op->get_attr<int>("shift_hswish") != -128,
                              XIR_INVALID_ARG_OCCUR)
                    << "the activation type is "
                    << op->get_attr<std::string>("nonlinear")
                    << " but you do not set the shift_hswish for this "
                       "operation.";
              UNI_LOG_CHECK(
                  op->has_attr("shift_bias") && op->has_attr("shift_cut"),
                  XIR_INVALID_ARG_OCCUR)
                  << "the activation type is "
                  << op->get_attr<std::string>("nonlinear")
                  << " you need to set shift_bias and shift_cut for it..";
            }
          }
        });
  };
}

auto conv3d_fix =
    xir::OpDef("conv3d-fix")
        .inherit_from(FixedConv3dOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_conv3d_fix);

auto depthwise_conv3d_fix =
    xir::OpDef("depthwise-conv3d-fix")
        .inherit_from(FixedConv3dOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_depthwise_conv3d_fix);

auto transposed_conv3d_fix =
    xir::OpDef("transposed-conv3d-fix")
        .inherit_from(FixedConv3dOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_transposed_conv3d_fix);

auto transposed_depthwise_conv3d_fix =
    xir::OpDef("transposed-depthwise-conv3d-fix")
        .inherit_from(FixedConv3dOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_transposed_depthwise_conv3d_fix);

XIR_REGISTER_BUILT_IN_OP(conv3d_fix);
XIR_REGISTER_BUILT_IN_OP(depthwise_conv3d_fix);
XIR_REGISTER_BUILT_IN_OP(transposed_conv3d_fix);
XIR_REGISTER_BUILT_IN_OP(transposed_depthwise_conv3d_fix);

std::function<void(xir::OpDef&)> Pool1dOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "An input tensor with shape "
                               "`[batch, in_length, in_channels]`."};
    auto kernel = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "kernel", AttrDef::REQUIRED, 1,
        "`Datatype`: `vector<int>`\n\n"
        "The kernel sizes of the filter. "
        "The value must be: `{kernel_length}`.");
    auto stride = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "stride", AttrDef::REQUIRED, 1,
        "`Datatype`: `vector<int>`\n\n"
        "The strides of the filter. "
        "The value must be: `{stride_length}`.");
    auto pad_mode = xir::AttrDefBuilder<string>::build(
        "pad_mode", AttrDef::REQUIRED,
        "`Datatype`: `string`\n\n"
        "We support 4 padding mode: FLOOR, CEIL, SAME, VALID. "
        "For example, when you parsing models from other frameworks, "
        "`caffe->\"CEIL\",  tensorflow->\"SAME\" or \"VALID\", "
        "pytorch->\"FLOOR\"(default) or \"CEIL\".`");
    auto pad = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "pad", AttrDef::OPTIONAL, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The padding sizes of input feature maps. "
        "The value must be `{left, right}`.",
        {0, 0});
    auto global = xir::AttrDefBuilder<bool>::build(
        "global", AttrDef::OPTIONAL,
        "`Datatype`: `bool`\n\n"
        "Global pooling, if global is set to be true, "
        "the length of output feature maps would be {1}.",
        0);
    op_def.add_input_arg(input)
        .add_attr(kernel)
        .add_attr(stride)
        .add_attr(pad_mode)
        .add_attr(pad)
        .add_attr(global);
  };
}

auto maxpool1d =
    xir::OpDef("maxpool1d")
        .inherit_from(Pool1dOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "1D max-pooling.\n\n"
            "    output[batch, ol, c] =\n"
            "        max_{kl} input[batch, strides[0] * ol + kl, c] *\n"
            "(1). if pad_mode == \"`FLOOR`\":\n\n"
            "    output_shape = floor((input_shape + pad - kernel) / stride) + "
            "1\n"
            "(2). if pad_mode == \"`CEIL`\":\n\n"
            "    output_shape = ceil((input_shape + pad - kernel) / stride) + "
            "1\n"
            "(3). if pad_mode == \"`SAME`\":\n\n"
            "    output_shape = ceil((input_shape + pad) / stride)\n"
            "(4). if pad_mode == \"`VALID`\":\n\n"
            "    output_shape = ceil((input_shape + pad - kernel) / stride)\n")
        .set_shape_infer(xir::shape_infer_maxpool1d);

XIR_REGISTER_BUILT_IN_OP(maxpool1d);

std::function<void(xir::OpDef&)> Pool2dOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "An input tensor with shape "
                               "`[batch, in_height, in_width, in_channels]`."};
    auto kernel = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "kernel", AttrDef::REQUIRED, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The kernel sizes of the filter. "
        "The value must be: `{kernel_width, kernel_height}`.");
    auto stride = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "stride", AttrDef::REQUIRED, 2,
        "`Datatype`: `vector<int>`\n\n"
        "The strides of the filter. "
        "The value must be: `{stride_width, stride_height}`.");
    auto pad_mode = xir::AttrDefBuilder<string>::build(
        "pad_mode", AttrDef::REQUIRED,
        "`Datatype`: `string`\n\n"
        "We support 4 padding mode: FLOOR, CEIL, SAME, VALID. "
        "For example, when you parsing models from other frameworks, "
        "`caffe->\"CEIL\",  tensorflow->\"SAME\" or \"VALID\", "
        "pytorch->\"FLOOR\"(default) or \"CEIL\".`");
    auto pad = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "pad", AttrDef::OPTIONAL, 4,
        "`Datatype`: `vector<int>`\n\n"
        "The padding sizes of input feature maps. "
        "The value must be `{left, right, top, bottom}`.",
        {0, 0, 0, 0});
    auto global = xir::AttrDefBuilder<bool>::build(
        "global", AttrDef::OPTIONAL,
        "`Datatype`: `bool`\n\n"
        "Global pooling, if global is set to be true, "
        "the width and height of output feature maps would be {1, 1}.",
        0);
    op_def.add_input_arg(input)
        .add_attr(kernel)
        .add_attr(stride)
        .add_attr(pad_mode)
        .add_attr(pad)
        .add_attr(global);
  };
}

auto maxpool2d =
    xir::OpDef("maxpool2d")
        .inherit_from(Pool2dOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "2D max-pooling.\n\n"
            "    output[batch, oh, ow, c] =\n"
            "        max_{kw, kh} input[batch, strides[1] * oh + kh,\n"
            "           strides[0] * ow + kw, c] *\n"
            "(1). if pad_mode == \"`FLOOR`\":\n\n"
            "    output_shape = floor((input_shape + pad - kernel) / stride) + "
            "1\n"
            "(2). if pad_mode == \"`CEIL`\":\n\n"
            "    output_shape = ceil((input_shape + pad - kernel) / stride) + "
            "1\n"
            "(3). if pad_mode == \"`SAME`\":\n\n"
            "    output_shape = ceil((input_shape + pad) / stride)\n"
            "(4). if pad_mode == \"`VALID`\":\n\n"
            "    output_shape = ceil((input_shape + pad - kernel) / stride)\n")
        .set_shape_infer(xir::shape_infer_maxpool2d);

auto avgpool2d =
    xir::OpDef("avgpool2d")
        .inherit_from(Pool2dOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "2D average-pooling.\n\n"
            "    output[batch, oh, ow, c] =\n"
            "        avg_{kw, kh} input[batch, strides[1] * oh + kh,\n"
            "           strides[0] * ow + kw, c] *\n"
            "(1). if pad_mode == \"`FLOOR`\":\n\n"
            "    output_shape = floor((input_shape + pad - kernel) / stride) + "
            "1\n"
            "(2). if pad_mode == \"`CEIL`\":\n\n"
            "    output_shape = ceil((input_shape + pad - kernel) / stride) + "
            "1\n"
            "(3). if pad_mode == \"`SAME`\":\n\n"
            "    output_shape = ceil((input_shape + pad) / stride)\n"
            "(4). if pad_mode == \"`VALID`\":\n\n"
            "    output_shape = ceil((input_shape + pad - kernel) / stride)\n")
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "count_include_pad", xir::AttrDef::OPTIONAL,
            "`Datatype`: `bool`\n\n"
            "if count data in the pad position for avg_pool?"
            "For example, caffe is `true`, tensorflow is `false`,"
            "pytorch uses `true` as default.",
            true))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "count_include_invalid", xir::AttrDef::OPTIONAL,
            "`Datatype`: `bool`\n\n"
            "if count data outside the padded input feature maps?"
            "For example, caffe is `false`, tf is `true`,"
            "pytorch is `true`.",
            true))
        .set_shape_infer(xir::shape_infer_avgpool2d);

auto pool_fix =
    xir::OpDef("pool-fix")
        .add_input_arg(
            xir::OpArgDef{"input", OpArgDef::REQUIRED, xir::DataType::XINT,
                          "An input tensor with shape "
                          "`[batch, in_height, in_width, in_channels]`."})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "kernel", AttrDef::REQUIRED, 2,
            "`Datatype`: `vector<int>`\n\n"
            "The kernel sizes of the filter. "
            "The value must be: `{kernel_width, kernel_height}`."))
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "stride", AttrDef::REQUIRED, 2,
            "`Datatype`: `vector<int>`\n\n"
            "The strides of the filter. "
            "The value must be: `{stride_width, stride_height}`."))
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "pad", AttrDef::OPTIONAL, 4,
            "`Datatype`: `vector<int>`\n\n"
            "The padding sizes of input feature maps. "
            "The value must be `{left, right, top, bottom}`. "
            "This is an optional attribute, when the pad_mode is SAME or "
            "VALID, "
            "you don't need to specify this attribute.",
            {0, 0, 0, 0}))
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "type", AttrDef::REQUIRED,
            "`Datatype`: `string`\n\n"
            "Dpu pooling type, \"MAX\", \"AVG\"."))
        .set_shape_infer(xir::shape_infer_pool_fix);

XIR_REGISTER_BUILT_IN_OP(maxpool2d);
XIR_REGISTER_BUILT_IN_OP(avgpool2d);
XIR_REGISTER_BUILT_IN_OP(pool_fix);

std::function<void(xir::OpDef&)> BroadcastOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED_AND_REPEATED, T,
                               "The feature maps, can be x-dimension."};
    op_def.add_input_arg(input)
        .set_annotation(
            "We support broadcasting operations:\n\n"
            "    \"add\": input[0] + input[1]\n"
            "    \"sub\": input[0] - input[1]\n"
            "    \"mul\": input[0] * input[1]\n"
            "    \"div\": input[0] / input[1]\n"
            "    \"min\": min(input[0], input[1])\n"
            "    \"max\": max(input[0], input[1])\n"
            "What is broadcasting?\n\n"
            "When operating on two arrays, we compare their shapes "
            "element-wise. \n"
            "It starts with the trailing dimensions, and works its way "
            "forward.\n\n"
            "Two dimensions are compatible when:\n\n"
            "1. they are equal, or\n"
            "2. one of them is 1\n"
            "If these conditions are not met, a mismatch would be thrown, \n"
            "indicating that the arrays have incompatible shapes. \n"
            "The size of the resulting array is the maximum size \n"
            "along each dimension of the input arrays.\n"
            "For example,\n\n"
            "(1). bias_add, which is a channel-wise operation:\n\n"
            "    input[0] (4d tensor): 1 x 112 x 112 x 64\n"
            "    input[1] (1d tensor):                 64\n"
            "    result   (4d tensor): 1 x 112 x 112 x 64\n"
            "(2). element-wise add, which is an element-wise operation:\n\n"
            "    input[0] (3d tensor): 32 x 32 x 10\n"
            "    input[1] (3d tensor): 32 x 32 x 10\n"
            "    result   (3d tensor): 32 x 32 x 10\n"
            "(3). more examples:\n\n"
            "    input[0] (4d tensor): 1 x 32 x 32 x 10\n"
            "    input[1] (3d tensor):     32 x  1 x  1\n"
            "    result   (4d tensor): 1 x 32 x 32 x 10\n"
            "(4). mismatched examples:\n\n"
            "    input[0] (4d tensor): 1 x 32 x 32 x 10\n"
            "    input[1] (3d tensor):      1 x 32 x  2\n"
            "    result              :         mismatch\n")
        .add_constraint([](xir::Op* op) {
          UNI_LOG_CHECK(op->get_input_num() > 1, XIR_INVALID_ARG_OCCUR)
              << op->to_string() << " only has " << op->get_input_num()
              << " input arguments, but it requires at least 2 inputs.";
        });
  };
}

auto add = xir::OpDef("add")
               .inherit_from(BroadcastOpDefGenerator(xir::DataType::FLOAT))
               .set_shape_infer(xir::shape_infer_add);

auto sub = xir::OpDef("sub")
               .inherit_from(BroadcastOpDefGenerator(xir::DataType::FLOAT))
               .set_shape_infer(xir::shape_infer_sub);

auto mul = xir::OpDef("mul")
               .inherit_from(BroadcastOpDefGenerator(xir::DataType::FLOAT))
               .set_shape_infer(xir::shape_infer_mul);

auto div = xir::OpDef("div")
               .inherit_from(BroadcastOpDefGenerator(xir::DataType::FLOAT))
               .set_shape_infer(xir::shape_infer_div);

auto min = xir::OpDef("min")
               .inherit_from(BroadcastOpDefGenerator(xir::DataType::FLOAT))
               .set_shape_infer(xir::shape_infer_min);

auto max = xir::OpDef("max")
               .inherit_from(BroadcastOpDefGenerator(xir::DataType::FLOAT))
               .set_shape_infer(xir::shape_infer_max);

XIR_REGISTER_BUILT_IN_OP(add);
XIR_REGISTER_BUILT_IN_OP(sub);
XIR_REGISTER_BUILT_IN_OP(mul);
XIR_REGISTER_BUILT_IN_OP(div);
XIR_REGISTER_BUILT_IN_OP(min);
XIR_REGISTER_BUILT_IN_OP(max);

std::function<void(xir::OpDef&)> ActivationOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "The feature maps, can be x-dimension."};
    op_def.add_input_arg(input);
  };
}

auto relu = xir::OpDef("relu")
                .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                .set_annotation(
                    "Computes the rectified linear element-wise:\n\n"
                    "    f(x) = max(0, x).\n")
                .set_shape_infer(xir::shape_infer_relu);

auto leaky_relu =
    xir::OpDef("leaky-relu")
        .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
        .add_attr(xir::AttrDefBuilder<float>::build(
            "alpha", xir::AttrDef::REQUIRED,
            "`Datatype`: `float`\n\n"
            "Slope of the activation function at x < 0."))
        .set_annotation(
            "Computes the leaky relu function element-wise:\n\n"
            "    f(x) = min(x, 0) + alpha * min(x, 0).\n")
        .set_shape_infer(xir::shape_infer_leaky_relu);

auto prelu = xir::OpDef("prelu")
                 .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                 .add_input_arg(xir::OpArgDef{
                     "weight", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                     "A learnable parameter with shape 1 or `in_channels`."})
                 .set_annotation(
                     "Computes the prelu function element-wise:\n\n"
                     "    f(x) = min(0, x) + weight * min(0, x).\n")
                 .set_shape_infer(xir::shape_infer_prelu);

auto relu6 = xir::OpDef("relu6")
                 .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                 .set_annotation(
                     "Computes the relu6 function element-wise:\n\n"
                     "    f(x) = min(max(x, 0), 6).\n")
                 .set_shape_infer(xir::shape_infer_relu6);

auto elu = xir::OpDef("elu")
               .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
               .set_annotation(
                   "Computes the elu function element-wise:\n\n"
                   "    f(x) = x if x > 0.\n"
                   "    f(x) = alpha * (exp(x) - 1) if x <= 0.\n")
               .add_attr(xir::AttrDefBuilder<float>::build(
                   "alpha", xir::AttrDef::OPTIONAL,
                   "`Datatype`: `float`\n\n"
                   "Slope of the activation function at x <= 0.",
                   1))
               .set_shape_infer(xir::shape_infer_elu);

auto celu =
    xir::OpDef("celu")
        .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Computes the celu function element-wise:\n\n"
            "    f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1)).\n")
        .add_attr(xir::AttrDefBuilder<float>::build(
            "alpha", xir::AttrDef::OPTIONAL,
            "`Datatype`: `float`\n\n"
            "Slope of the activation function.",
            1))
        .set_shape_infer(xir::shape_infer_celu);

auto gelu = xir::OpDef("gelu")
                .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                .set_annotation(
                    "Computes the gelu function element-wise:\n\n"
                    "    f(x) = x * 1 / 2 * (1 + erf(x / sqrt(2))).\n")
                .set_shape_infer(xir::shape_infer_gelu);

auto mish = xir::OpDef("mish")
                .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                .set_annotation(
                    "Computes the mish function element-wise:\n\n"
                    "    f(x) = x ∗ Tanh(Softplus(x)) .\n")
                .set_shape_infer(xir::shape_infer_mish);

auto selu =
    xir::OpDef("selu")
        .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Computes the selu function element-wise:\n\n"
            "    f(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1))).\n"
            "alpha and scale are constant value.\n")
        .set_shape_infer(xir::shape_infer_selu);

auto sigmoid = xir::OpDef("sigmoid")
                   .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                   .set_annotation(
                       "Computes the sigmoid function element-wise:\n\n"
                       "    f(x) = 1 / (1 + exp(-x)).\n")
                   .set_shape_infer(xir::shape_infer_sigmoid);

auto swish = xir::OpDef("swish")
                 .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                 .set_annotation(
                     "Computes the swish function element-wise:\n\n"
                     "    f(x) = x * sigmoid(x).\n")
                 .set_shape_infer(xir::shape_infer_swish);

auto tanh = xir::OpDef("tanh")
                .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
                .set_annotation(
                    "Computes the tanh function element-wise:\n\n"
                    "    f(x) = tanh(x).\n")
                .set_shape_infer(xir::shape_infer_tanh);

auto hard_sigmoid =
    xir::OpDef("hard-sigmoid")
        .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Computes the hard sigmoid function element-wise:\n\n"
            "    f(x) = relu6(x + 3) / 6.\n")
        .set_shape_infer(xir::shape_infer_hard_sigmoid);

auto hard_sigmoid_fix =
    xir::OpDef("hard-sigmoid-fix")
        .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Computes the hard sigmoid function element-wise:\n\n"
            "    f(x) = relu6(x + 3) * 2731 / 2 ^ 14.\n")
        .set_shape_infer(xir::shape_infer_hard_sigmoid);

auto hard_swish =
    xir::OpDef("hard-swish")
        .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Computes the hard swish function element-wise:\n\n"
            "    f(x) = x * relu6(x + 3) / 6.\n")
        .set_shape_infer(xir::shape_infer_hard_swish);

auto hard_tanh =
    xir::OpDef("hard-tanh")
        .inherit_from(ActivationOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Computes the hard tanh function element-wise:\n\n"
            "    f(x) = clip(x, -1, 1).\n")
        .set_shape_infer(xir::shape_infer_hard_tanh);

XIR_REGISTER_BUILT_IN_OP(relu);
XIR_REGISTER_BUILT_IN_OP(leaky_relu);
XIR_REGISTER_BUILT_IN_OP(prelu);
XIR_REGISTER_BUILT_IN_OP(relu6);
XIR_REGISTER_BUILT_IN_OP(elu);
XIR_REGISTER_BUILT_IN_OP(celu);
XIR_REGISTER_BUILT_IN_OP(gelu);
XIR_REGISTER_BUILT_IN_OP(mish);
XIR_REGISTER_BUILT_IN_OP(selu);
XIR_REGISTER_BUILT_IN_OP(sigmoid);
XIR_REGISTER_BUILT_IN_OP(swish);
XIR_REGISTER_BUILT_IN_OP(tanh);
XIR_REGISTER_BUILT_IN_OP(hard_sigmoid);
XIR_REGISTER_BUILT_IN_OP(hard_sigmoid_fix);
XIR_REGISTER_BUILT_IN_OP(hard_swish);
XIR_REGISTER_BUILT_IN_OP(hard_tanh);

std::function<void(xir::OpDef&)> FixOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "The feature maps, can be x-dimension."};
    auto fix_point = xir::AttrDefBuilder<std::int32_t>::build(
        "fix_point", AttrDef::REQUIRED,
        "`Datatype`: `int`\n\n"
        "The fixed position of the output feature maps.");
    auto bit_width = xir::AttrDefBuilder<std::int32_t>::build(
        "bit_width", AttrDef::REQUIRED,
        "`Datatype`: `int`\n\n"
        "The bit width of the output feature maps.");
    auto if_signed = xir::AttrDefBuilder<bool>::build(
        "if_signed", AttrDef::REQUIRED,
        "`Datatype`: `bool`\n\n"
        "If the output feature maps is signed, this attr is set to be true.");
    auto round_mode = xir::AttrDefBuilder<std::string>::build(
        "round_mode", AttrDef::REQUIRED,
        "`Datatype`: `string`\n\n"
        "The round mode function for transforming the float data."
        "The round mode is one of `{STD_ROUND, DPU_ROUND, PY3_ROUND}`\n\n"
        "(1). If the round_mode = `STD_ROUND`:\n\n"
        "    f(x) = std::round(x)\n"
        "For example, f(2.3) = 2, f(2.5) = 3, f(-2.5) = -3, f(-2.6) = -3.\n\n"
        "(2). If the round_mode = `DPU_ROUND`:\n\n"
        "    f(x) = ((x < 0) && (x - floor(x) == 0.5))\n"
        "           ? std::ceil(x)\n"
        "           : std::round(x)\n"
        "For example, f(2.3) = 2, f(2.5) = 3, f(-2.5) = -2, f(-2.6) = -3.\n\n"
        "(3). If the round_mode = `PY3_ROUND`:\n\n"
        "Round to even."
        "For example, f(2.3) = 2, f(2.5) = 2, f(-2.5) = -2, f(-2.6) = -3.\n\n");
    op_def.add_input_arg(input)
        .add_attr(fix_point)
        .add_attr(bit_width)
        .add_attr(round_mode)
        .add_attr(if_signed);
  };
}

auto fix =
    xir::OpDef("fix")
        .inherit_from(FixOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "fix operator transforms float-point value into "
            "fixed-point value into float-point format.\n\n"
            "(1). Firstly, we transform the float input feature map x into "
            "fixed value:\n\n"
            "    fixed_value = round(x * pow(2, fix_point))\n"
            "and then\n\n"
            "(2) transform the fixed value into float-point format:\n\n"
            "-> if_signed == true:\n\n"
            "    output = max(-pow(2, bit_width - 1),\n"
            "                 min(fixed_value, pow(2, bit_width - 1) - 1)))\n"
            "               * pow(2, -fix_point)\n"
            "-> if_signed == false:\n\n"
            "    output = max(0, min(fixed_value, pow(2, bit_width) - 1)))\n"
            "               * pow(2, -fix_point)\n")
        .set_shape_infer(xir::shape_infer_fix);

auto fix2float =
    xir::OpDef("fix2float")
        .inherit_from(FixOpDefGenerator(xir::DataType::XINT))
        .set_annotation(
            "Transform the fixed value x into float output:\n\n"
            "(1). if_signed == true:\n\n"
            "    output = max(-pow(2, bit_width - 1),\n"
            "                 min(x, pow(2, bit_width - 1) - 1)))\n"
            "               * pow(2, -fix_point)\n"
            "(2). if_signed == false:\n\n"
            "    output = max(0, min(x, pow(2, bit_width) - 1)))\n"
            "               * pow(2, -fix_point)\n")
        .set_shape_infer(xir::shape_infer_fix2float);

auto float2fix =
    xir::OpDef("float2fix")
        .inherit_from(FixOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Transform the float value x into fixed value:\n\n"
            "    f(x) = round(x * pow(2, fix_point))\n"
            "The round function is determined by the round_mode.\n\n"
            "(1). if_signed == true:\n\n"
            "    output = max(-pow(2, bit_width - 1),\n"
            "                 min(f(x), pow(2, bit_width - 1) - 1)))\n"
            "(2). if_signed == false:\n\n"
            "    output = max(0, min(f(x), pow(2, bit_width) - 1)))\n")
        .set_shape_infer(xir::shape_infer_float2fix);

auto threshold =
    xir::OpDef("threshold")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::XINT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"threshold", OpArgDef::REQUIRED,
                                     xir::DataType::XINT,
                                     "1-dimension, 24-bit XINT"})
        .set_annotation(
            "Threshold operator is used to transform fixed-point values \n"
            "to fixed-point values of different bit width.\n\n"
            "    24 bit threshold = 13-bit base + 10-bit delta + 1-bit "
            "signal.\n"
            "base is a channel-wise parameter, an int_13 number.\n"
            "11 bit interger and 2 bit decimal.\n\n"
            "delta is a channel-wise parameter, an uint_10 number.\n"
            "8 bit interger and 2 bit decimal.\n\n"
            "The output can be calculated by this function:\n\n"
            "    base + out * delta <= in < base + (out + 1) * delta\n"
            "In addition, signal indicates whether actual step is a positive "
            "number.\n"
            "0 indicates positive, 1 is negative.\n")
        .set_shape_infer(xir::shape_infer_threshold);

XIR_REGISTER_BUILT_IN_OP(fix);
XIR_REGISTER_BUILT_IN_OP(fix2float);
XIR_REGISTER_BUILT_IN_OP(float2fix);
XIR_REGISTER_BUILT_IN_OP(threshold);

std::function<void(xir::OpDef&)> ReductionOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "The feature maps, can be x-dimension."};
    auto axis = xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
        "axis", AttrDef::REQUIRED, 0,
        "`Datatype`: `vector<int>`\n\n"
        "The dimensions to reduce.");
    auto keep_dims = xir::AttrDefBuilder<bool>::build(
        "keep_dims", AttrDef::REQUIRED,
        "`Datatype`: `bool`\n\n"
        "specify whether the reduced dimension is kept or not.");
    op_def.add_input_arg(input).add_attr(axis).add_attr(keep_dims);
  };
}

auto reduction_mean =
    xir::OpDef("reduction_mean")
        .inherit_from(ReductionOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation("Implement the mean along each of the axis dimensions.")
        .set_shape_infer(xir::shape_infer_reduction_mean);

auto reduction_product =
    xir::OpDef("reduction_product")
        .inherit_from(ReductionOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Implement the product along each of the axis dimensions.")
        .set_shape_infer(xir::shape_infer_reduction_product);

auto reduction_sum =
    xir::OpDef("reduction_sum")
        .inherit_from(ReductionOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation("Implement the sum along each of the axis dimensions.")
        .set_shape_infer(xir::shape_infer_reduction_sum);

auto reduction_max =
    xir::OpDef("reduction_max")
        .inherit_from(ReductionOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Find the maximum value along each of the axis dimensions.")
        .set_shape_infer(xir::shape_infer_reduction_max);

auto reduction_max_fix =
    xir::OpDef("reduction_max-fix")
        .inherit_from(ReductionOpDefGenerator(xir::DataType::XINT))
        .set_annotation(
            "Find the maximum value along each of the axis dimensions.")
        .set_shape_infer(xir::shape_infer_reduction_max_fix);

XIR_REGISTER_BUILT_IN_OP(reduction_mean);
XIR_REGISTER_BUILT_IN_OP(reduction_product);
XIR_REGISTER_BUILT_IN_OP(reduction_sum);
XIR_REGISTER_BUILT_IN_OP(reduction_max);
XIR_REGISTER_BUILT_IN_OP(reduction_max_fix);

auto l2_normalize =
    xir::OpDef("l2_normalize")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "axis", AttrDef::OPTIONAL, 0,
            "`Datatype`: `vector<int>`\n\n"
            "Dimension along which to normalize.",
            {}))
        .add_attr(xir::AttrDefBuilder<double>::build(
            "epsilon", AttrDef::OPTIONAL,
            "`Datatype`: `double`\n\n"
            "A lower bound value for the norm.",
            0.000000000001))
        .set_annotation(
            "For a 1-D tensor with `axis = 0`, computes\n\n"
            "    output = x / sqrt(max(sum(x ^ 2), epsilon))\n"
            "For x with more dimensions,\n"
            "independently normalizes each 1-D slice along dimension axis.\n")
        .set_shape_infer(xir::shape_infer_l2_normalize);

XIR_REGISTER_BUILT_IN_OP(l2_normalize);

auto argmax = xir::OpDef("argmax")
                  .add_input_arg(xir::OpArgDef{
                      "input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                      "The feature maps, can be x-dimension."})
                  .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
                      "axis", AttrDef::REQUIRED,
                      "`Datatype`: `int`\n\n"
                      "axis"))
                  .set_annotation(
                      "Computes the index of the maximum of values along "
                      "dimension axis.\n\n")
                  .set_shape_infer(xir::shape_infer_argmax);

auto argmax_fix =
    xir::OpDef("argmax-fix")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::XINT, "x-dimension"})
        .add_attr(
            xir::AttrDefBuilder<std::int32_t>::build("axis", AttrDef::REQUIRED,
                                                     "`Datatype`: `int`\n\n"
                                                     "axis"))
        .set_annotation(
            "Computes the index of the maximum of values along dimension "
            "axis.\n\n")
        .set_shape_infer(xir::shape_infer_argmax);

XIR_REGISTER_BUILT_IN_OP(argmax);
XIR_REGISTER_BUILT_IN_OP(argmax_fix);

std::function<void(xir::OpDef&)> InterfaceOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    auto input = xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                               "The feature maps, can be x-dimension."};
    op_def.add_input_arg(input);
  };
}

auto identity =
    xir::OpDef("identity")
        .inherit_from(InterfaceOpDefGenerator(xir::DataType::XINT))
        .set_annotation(
            "An interface operator that holds the data. Do nothing here.")
        .set_shape_infer(xir::shape_infer_identity);

auto upload = xir::OpDef("upload")
                  .inherit_from(InterfaceOpDefGenerator(xir::DataType::FLOAT))
                  .set_annotation(
                      "An interface operator that holds the data achieved by a "
                      "CPU-runner, "
                      "and would be sent to a DPU-runner later.")
                  .set_shape_infer(xir::shape_infer_upload);

auto download =
    xir::OpDef("download")
        .inherit_from(InterfaceOpDefGenerator(xir::DataType::XINT))
        .set_annotation(
            "An interface operator that holds the data achieved by a "
            "DPU-runner, "
            "and would be sent to a CPU-runner later.")
        .set_shape_infer(xir::shape_infer_download);

auto placeholder =
    xir::OpDef("placeholder")
        .inherit_from(InterfaceOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "An interface operator that holds the data. Do nothing here.")
        .set_shape_infer(xir::shape_infer_placeholder);

XIR_REGISTER_BUILT_IN_OP(identity);
XIR_REGISTER_BUILT_IN_OP(upload);
XIR_REGISTER_BUILT_IN_OP(download);
XIR_REGISTER_BUILT_IN_OP(placeholder);

auto shape = xir::OpDef("shape")
                 .add_input_arg(xir::OpArgDef{
                     "input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                     "The feature maps, can be x-dimension."})
                 .set_annotation("Return the shape of the input feature maps.")
                 .set_shape_infer(xir::shape_infer_shape);

XIR_REGISTER_BUILT_IN_OP(shape);

auto reshape =
    xir::OpDef("reshape")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{
            "shape", OpArgDef::OPTIONAL, xir::DataType::INT,
            "Constant values that define the shape of the output."})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "shape", AttrDef::OPTIONAL, 0,
            "`Datatype`: `vector<int>`\n\n"
            "Constant values that define the shape of the output.",
            {0}))
        .set_annotation(
            "Reshape the feature maps or constant data into new shape without "
            "changing "
            "the layout of data in memory.")
        .set_shape_infer(xir::shape_infer_reshape);

XIR_REGISTER_BUILT_IN_OP(reshape);

auto reshape_fix =
    xir::OpDef("reshape-fix")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::XINT, "x-dimension"})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "shape", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "shape"))
        .set_shape_infer(xir::shape_infer_reshape_fix);

XIR_REGISTER_BUILT_IN_OP(reshape_fix);

auto squeeze =
    xir::OpDef("squeeze")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "axis", AttrDef::OPTIONAL, 0,
            "`Datatype`: `vector<int>`\n\n"
            "The dimensions to be squeezed.\n"
            "If axis is not specified, all dimensions equal to 1 would be "
            "squeezed.",
            {}))
        .set_annotation(
            "For example:\n\n"
            "    input.shape  = [32, 2, 1, 1]\n"
            "    axis = {2, 3}\n"
            "    output.shape = [32, 2]")
        .set_shape_infer(xir::shape_infer_squeeze);

XIR_REGISTER_BUILT_IN_OP(squeeze);

auto transpose =
    xir::OpDef("transpose")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "order", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "The order to be transposed."))
        .set_annotation(
            "For example:\n\n"
            "    input.shape  = [32, 2, 64, 4]\n"
            "    order = {0, 3, 2, 1}\n"
            "    output = input.transpose([0, 3, 2, 1]\n"
            "    output.shape = [32, 4, 64, 2]")
        .set_shape_infer(xir::shape_infer_transpose);

XIR_REGISTER_BUILT_IN_OP(transpose);

auto flatten = xir::OpDef("flatten")
                   .add_input_arg(xir::OpArgDef{
                       "input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                       "The feature maps, can be x-dimension."})
                   .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
                       "start_axis", AttrDef::REQUIRED,
                       "`Datatype`: `int`\n\n"
                       "start axis to be flattened"))
                   .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
                       "end_axis", AttrDef::REQUIRED,
                       "`Datatype`: `int`\n\n"
                       "end axis to be flattened"))
                   .set_annotation(
                       "For example:\n\n"
                       "    input.shape  = [32, 5, 5, 2, 4]\n"
                       "    start_axis = 1\n"
                       "    end_axis = -1\n"
                       "    output.shape = [32, 200]")
                   .set_shape_infer(xir::shape_infer_flatten);

XIR_REGISTER_BUILT_IN_OP(flatten);

std::function<void(xir::OpDef&)> ResizeOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{
            "size", OpArgDef::OPTIONAL, xir::DataType::INT,
            "Constant values denotes the shape of the output feature maps."})
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "mode", AttrDef::REQUIRED,
            "`Datatype`: `string`\n\n"
            "NEAREST, BILINEAR or TRILINEAR"))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "align_corners", AttrDef::OPTIONAL,
            "`Datatype`: `bool`\n\n"
            "If true, preserving the values at the corner pixels. "
            "Defaults to false.",
            false))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "half_pixel_centers", AttrDef::OPTIONAL,
            "`Datatype`: `bool`\n\n"
            "If true, use half-pixel as centers.",
            false));
  };
}

auto resize =
    xir::OpDef("resize")
        .inherit_from(ResizeOpDefGenerator(xir::DataType::FLOAT))
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "scale", AttrDef::OPTIONAL, 0,
            "`Datatype`: `vector<float>`\n\n"
            "Constant values denotes the scale to resize the input. "
            "scale = out / in",
            {}))
        .set_annotation(
            "Operator resize the feature maps. For example, if the input "
            "is an image, and the shape of this image is [h, w, c], after "
            "2d resize, "
            "the shape of the output image is [oh, ow, c].\n\n"
            "    scale = (align_corners && out > 1)\n"
            "            ? (in - 1) / (out - 1)\n"
            "            : in / out\n"
            "When given the index of output, how to find the corresponding "
            "input pixels:\n\n"
            "    scaler = half_pixel_centers\n"
            "             ? (out + 0.5) * scale - 0.5\n"
            "             : out * scale\n"
            "(1). for NEAREST resize:\n\n"
            "    w_idx[ow] = min((w - 1),\n"
            "                    align_corners ? round(scaler(ow))\n"
            "                                  : floor(scaler(ow)))\n"
            "    h_idx[oh] = min((h - 1),\n"
            "                    align_corners ? round(scaler(oh))\n"
            "                                  : floor(scaler(oh)))\n"
            "    resize[oh, ow, c] = image[h_idx[oh], w_idx[ow], c]\n"
            "(2). for BILINEAR resize:\n\n"
            "    top = floor(scaler(oh))\n"
            "    bottom = min(h - 1, top + 1)\n"
            "    left = floor(scaler(ow))\n"
            "    right = min(w - 1, left + 1)\n"
            "    x_lerp = scaler(ow) - left\n"
            "    y_lerp = scaler(oh) - top\n"
            "    reisze[oh, ow, c] = (image[top, left, c] * (1 - x_lerp) +\n"
            "                         image[top, right, c] * x_lerp) * (1 - "
            "y_lerp)\n"
            "                        (image[bottom, left, c] * (1 - x_lerp) +\n"
            "                         image[bottom, right, c] * x_lerp) * "
            "y_lerp\n")
        .add_constraint([](xir::Op* op) {
          auto mode = op->get_attr<std::string>("mode");
          auto in = op->get_input_tensor("input");
          auto in_shape = in->get_shape();
          if (mode == "NEAREST" || mode == "BILINEAR") {
            UNI_LOG_CHECK(in_shape.size() == 4, XIR_INVALID_ARG_OCCUR)
                << "We only support NEAREST and BILINEAR resize for 4-D "
                   "feature maps.";
          }
          if (mode == "TRILINEAR") {
            UNI_LOG_CHECK(in_shape.size() == 5, XIR_INVALID_ARG_OCCUR)
                << "We only support TRILINEAR resize for 5-D feature maps.";
          }
        })
        .set_shape_infer(xir::shape_infer_resize);

auto upsample_fix =
    xir::OpDef("upsample-fix")
        .inherit_from(ResizeOpDefGenerator(xir::DataType::XINT))
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "scale",                              //
            AttrDef::REQUIRED,                    //
            2,                                    //
            "`DataType` : `std::vector<float>` "  //
            "{scale_w, scale_h}"                  //
            ))
        .set_shape_infer(xir::shape_infer_upsample_fix);

auto downsample_fix =
    xir::OpDef("downsample-fix")
        .inherit_from(ResizeOpDefGenerator(xir::DataType::XINT))
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "scale",                              //
            AttrDef::REQUIRED,                    //
            2,                                    //
            "`DataType` : `std::vector<float>` "  //
            "{scale_w, scale_h}"                  //
            ))
        .set_shape_infer(xir::shape_infer_downsample_fix);

XIR_REGISTER_BUILT_IN_OP(resize);
XIR_REGISTER_BUILT_IN_OP(upsample_fix);
XIR_REGISTER_BUILT_IN_OP(downsample_fix);

auto inner_product =
    xir::OpDef("inner-product")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"weights", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"bias", OpArgDef::OPTIONAL,
                                     xir::DataType::FLOAT, "1-dimension, OC"})
        .add_attr(
            xir::AttrDefBuilder<std::int32_t>::build("axis", AttrDef::REQUIRED,
                                                     "`Datatype`: `int`\n\n"
                                                     "[axis:-1] for flatten"))
        .set_annotation(
            "Do inner-product for the input feature maps.\n\n"
            "For example, the shape of the input is [n, a, b, c], axis = 1, "
            "Firstly, flatten the input feature maps starting from the `axis` "
            "dimension "
            "to the end. the input would be reshaped to [n, a * b * c].\n\n"
            "Secondly, the weights would be reshaped to [k, a * b * c], \n\n"
            "Thirdly, the inner-product would be implemented:\n\n"
            "    output[n, k] = sum_{i} input(n, i) * weights(k, i)\n"
            "The number of bias equals to k.")
        .set_shape_infer(xir::shape_infer_inner_product);

XIR_REGISTER_BUILT_IN_OP(inner_product);

std::function<void(xir::OpDef&)> ConcatOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED_AND_REPEATED,
                                     T,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "axis", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "Dimension along which to concatenate."));
  };
}

auto concat =
    xir::OpDef("concat")
        .inherit_from(ConcatOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "Concatenates different feature maps along the dimension `axis`.\n"
            "All dimensions except axis must be equal.")
        .set_shape_infer(xir::shape_infer_concat);

auto concat_fix = xir::OpDef("concat-fix")
                      .inherit_from(ConcatOpDefGenerator(xir::DataType::XINT))
                      .set_shape_infer(xir::shape_infer_concat_fix);

XIR_REGISTER_BUILT_IN_OP(concat);
XIR_REGISTER_BUILT_IN_OP(concat_fix);

std::function<void(xir::OpDef&)> ReorgOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(
            xir::OpArgDef{"input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                          "`[batch, in_height, in_width, in_channels]`."})
        .add_attr(
            xir::AttrDefBuilder<std::int32_t>::build("scale", AttrDef::REQUIRED,
                                                     "`Datatype`: `int`\n\n"
                                                     "scale for reorg"))
        .add_attr(xir::AttrDefBuilder<bool>::build("reverse", AttrDef::REQUIRED,
                                                   "`Datatype`: `bool`\n\n"
                                                   "reorg or reversed reorg"));
  };
}

auto reorg = xir::OpDef("reorg")
                 .inherit_from(ReorgOpDefGenerator(xir::DataType::FLOAT))
                 .set_annotation(
                     "Reorg Operator in YOLO."
                     "The implementations can be seen in "
                     "https://github.com/intel/caffe/blob/master/include/caffe/"
                     "layers/reorg_layer.hpp.")
                 .set_shape_infer(xir::shape_infer_reorg);

auto reorg_fix = xir::OpDef("reorg-fix")
                     .inherit_from(ReorgOpDefGenerator(xir::DataType::XINT))
                     .set_shape_infer(xir::shape_infer_reorg_fix);

XIR_REGISTER_BUILT_IN_OP(reorg);
XIR_REGISTER_BUILT_IN_OP(reorg_fix);

std::function<void(xir::OpDef&)> PixelShuffleOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(
            xir::OpArgDef{"input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                          "`[batch, in_height, in_width, in_channels]`."})
        .add_attr(
            xir::AttrDefBuilder<std::int32_t>::build("scale", AttrDef::REQUIRED,
                                                     "`Datatype`: `int`\n\n"
                                                     "scale for PixelShuffle"))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "upscale", AttrDef::REQUIRED,
            "`Datatype`: `bool`\n\n"
            "upscale or downscale PixelShuffle."));
  };
}

auto pixel_shuffle =
    xir::OpDef("pixel-shuffle")
        .inherit_from(PixelShuffleOpDefGenerator(xir::DataType::FLOAT))
        .set_annotation(
            "https://pytorch.org/docs/stable/generated/"
            "torch.nn.PixelShuffle.html")
        .set_shape_infer(xir::shape_infer_pixel_shuffle);

auto pixel_shuffle_fix =
    xir::OpDef("pixel-shuffle-fix")
        .inherit_from(ReorgOpDefGenerator(xir::DataType::XINT))
        .set_shape_infer(xir::shape_infer_pixel_shuffle_fix);

XIR_REGISTER_BUILT_IN_OP(pixel_shuffle);
XIR_REGISTER_BUILT_IN_OP(pixel_shuffle_fix);

auto softmax =
    xir::OpDef("softmax")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "axis", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "the dimension softmax would be performed on."))
        .set_annotation(
            "Softmax Operator performs softmax along the dimension of axis.\n\n"
            "    f(o) = exp(o) / sum_{i}(exp(i))")
        .set_shape_infer(xir::shape_infer_softmax);

XIR_REGISTER_BUILT_IN_OP(softmax);

auto hard_softmax = xir::OpDef("hard-softmax")
                        .add_input_arg(xir::OpArgDef{
                            "input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                            "The feature maps, can be x-dimension."})
                        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
                            "axis", AttrDef::REQUIRED,
                            "`Datatype`: `int`\n\n"
                            "the dimension softmax would be performed on."))
                        .add_attr(xir::AttrDefBuilder<std::string>::build(
                            "type", AttrDef::REQUIRED,
                            "`Datatype`: `string`\n\n"
                            "the type hard-softmax would be performed on."))
                        .set_annotation(
                            "hard-softmax Operator performs hard-softmax along "
                            "the dimension of axis.\n\n"
                            "    f(o) = exp(o) / sum_{i}(exp(i))")
                        .set_shape_infer(xir::shape_infer_softmax);

XIR_REGISTER_BUILT_IN_OP(hard_softmax);

auto cast = xir::OpDef("cast")
                .add_input_arg(xir::OpArgDef{
                    "input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                    "The feature maps, can be x-dimension."})
                .add_attr(xir::AttrDefBuilder<std::string>::build(
                    "data_type", AttrDef::REQUIRED,
                    "`Datatype`: `string`\n\n"
                    "the data type would cast."))
                .set_annotation("Cast Operator cast input tensor to dtype.\n\n")
                .set_shape_infer(xir::shape_infer_cast);

XIR_REGISTER_BUILT_IN_OP(cast);

std::function<void(xir::OpDef&)> PadOpDefGenerator(xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED, T,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "paddings", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "pad along different dimensions, the number of value in paddings "
            "should be 2 times the number of dimensions of input feature maps."
            "The n-th dimension of the output feature maps equals to:\n\n"
            "    (n-th dim) out =\n"
            "        paddings[2n] + (n-th dim) in + paddings[2n + 1]"))
        .add_attr(xir::AttrDefBuilder<string>::build(
            "mode", AttrDef::REQUIRED,
            "`Datatype`: `string`\n\n"
            "`CONSTANT`,`REFLECT` or `SYMMETRIC`"))
        .set_annotation(
            "For example,\n\n"
            "if the mode = \"CONSTANT\"\n\n"
            "    input = [[1, 2],\n"
            "             [3, 4]]\n"
            "    paddings = [0, 1, 1, 0]\n"
            "    output = [[0, 1, 2],\n"
            "              [0, 3, 4],\n"
            "              [0, 0, 0]]\n"
            "if the mode = \"REFLECT\"\n\n"
            "    input = [[1, 2],\n"
            "             [3, 4]]\n"
            "    paddings = [0, 1, 1, 0]\n"
            "    output = [[2, 1, 2],\n"
            "              [4, 3, 4],\n"
            "              [2, 1, 2]]\n"
            "if the mode = \"SYMMETRIC\"\n\n"
            "    input = [[1, 2],\n"
            "             [3, 4]]\n"
            "    paddings = [0, 1, 1, 0]\n"
            "    output = [[1, 1, 2],\n"
            "              [3, 3, 4],\n"
            "              [3, 3, 4]]\n")
        .add_constraint([](xir::Op* op) {
          auto in = op->get_input_tensor("input");
          auto in_shape = in->get_shape();
          auto pad_shape = op->get_attr<std::vector<int>>("paddings");
          UNI_LOG_CHECK(in_shape.size() * 2 == pad_shape.size(),
                        XIR_INVALID_ARG_OCCUR)
              << "the number of attr \"paddings\" should be equal to 2 * "
                 "the number of input dimensions.";
        });
  };
}

auto pad = xir::OpDef("pad")
               .inherit_from(PadOpDefGenerator(xir::DataType::FLOAT))
               .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
                   "constant_values", AttrDef::OPTIONAL, 0,
                   "`Datatype`: `vector<float>`\n\n"
                   "the value set into the padded locations, 2 * len(paddings)",
                   {}))
               .set_shape_infer(xir::shape_infer_pad);

auto pad_fix =
    xir::OpDef("pad-fix")
        .inherit_from(PadOpDefGenerator(xir::DataType::XINT))
        .add_attr(xir::AttrDefBuilder<std::vector<char>>::build(
            "constant_values", AttrDef::OPTIONAL, 0,
            "`Datatype`: `vector<char>`\n\n"
            "the value set into the padded locations, 2 * len(paddings)",
            {}))
        .set_shape_infer(xir::shape_infer_pad_fix);

XIR_REGISTER_BUILT_IN_OP(pad);
XIR_REGISTER_BUILT_IN_OP(pad_fix);

auto batchnorm =
    xir::OpDef("batchnorm")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"gamma", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "`[in.shape[axis]]`"})
        .add_input_arg(xir::OpArgDef{"beta", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "`[in.shape[axis]]`"})
        .add_input_arg(xir::OpArgDef{"moving_mean", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "`[in.shape[axis]]`"})
        .add_input_arg(xir::OpArgDef{"moving_var", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "`[in.shape[axis]]`"})
        .add_attr(xir::AttrDefBuilder<int>::build(
            "axis", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "the axis of the input to implement batchnorm"))
        .add_attr(xir::AttrDefBuilder<float>::build(
            "epsilon", AttrDef::REQUIRED,
            "`Datatype`: `float`\n\n"
            "a value added to the denominator for numerical stability"))
        .set_annotation(
            "implements batchnorm along the last dimension of input feature "
            "maps.\n\n"
            "    output = (input - moving_mean) /\n"
            "             sqrt(moving_var + epsilon) * gamma + beta")
        .set_shape_infer(xir::shape_infer_batchnorm);

auto instancenorm =
    xir::OpDef("instancenorm")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"weight", OpArgDef::OPTIONAL,
                                     xir::DataType::FLOAT,
                                     "gamma: channel-wise."})
        .add_input_arg(xir::OpArgDef{"bias", OpArgDef::OPTIONAL,
                                     xir::DataType::FLOAT,
                                     "beta: channel-wise."})
        .add_attr(xir::AttrDefBuilder<float>::build(
            "eps", AttrDef::REQUIRED,
            "`Datatype`: `float`\n\n"
            "a value added to the denominator for numerical stability"))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "affine", AttrDef::REQUIRED,
            "`Datatype`: `bool`\n\n"
            "a boolean value that when set to ``True``, this module has "
            "learnable "
            "affine parameters, initialized the same way as done for batch "
            "normalization."))
        .set_annotation(
            "implements instancenorm along the last dimension of input feature "
            "maps.\n\n"
            "    output = (input - moving_mean) /\n"
            "             sqrt(moving_var + epsilon) * gamma + beta")
        .set_shape_infer(xir::shape_infer_instancenorm);

auto groupnorm =
    xir::OpDef("groupnorm")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"weight", OpArgDef::OPTIONAL,
                                     xir::DataType::FLOAT,
                                     "gamma: channel-wise."})
        .add_input_arg(xir::OpArgDef{"bias", OpArgDef::OPTIONAL,
                                     xir::DataType::FLOAT,
                                     "beta: channel-wise."})
        .add_attr(xir::AttrDefBuilder<int>::build(
            "num_groups", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "number of groups to separate the channels into"))
        .add_attr(xir::AttrDefBuilder<int>::build(
            "num_channels", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "number of channels expected in input"))
        .add_attr(xir::AttrDefBuilder<float>::build(
            "eps", AttrDef::REQUIRED,
            "`Datatype`: `float`\n\n"
            "a value added to the denominator for numerical stability"))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "affine", AttrDef::REQUIRED,
            "`Datatype`: `bool`\n\n"
            "a boolean value that when set to ``True``, this module has "
            "learnable "
            "affine parameters, initialized the same way as done for batch "
            "normalization."))
        .set_annotation(
            "Applies Group Normalization over a mini-batch of inputs as "
            "described in `VAI-SRC-CONF-MAP-15`.\n\n"
            "..math::"
            "y = frac{x - mathrm{E}[x]}{ sqrt{mathrm{Var}[x] + epsilon}} * "
            "gamma + beta")
        .set_shape_infer(xir::shape_infer_groupnorm);

XIR_REGISTER_BUILT_IN_OP(batchnorm);
XIR_REGISTER_BUILT_IN_OP(instancenorm);
XIR_REGISTER_BUILT_IN_OP(groupnorm);

std::function<void(xir::OpDef&)> StridedSliceOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "begin", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "start location of slicing"))
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "end", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "end location of slicing"))
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "strides", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "strides of slicing"))
        .set_annotation(
            "This operator is NumPy-style slicing syntax,\n\n"
            "    output = input[begin:end:strides]\n")
        .set_shape_infer(xir::shape_infer_strided_slice);
  };
}
auto strided_slice =
    xir::OpDef("strided_slice")
        .inherit_from(StridedSliceOpDefGenerator(xir::DataType::FLOAT));
auto strided_slice_fix =
    xir::OpDef("strided_slice-fix")
        .inherit_from(StridedSliceOpDefGenerator(xir::DataType::XINT));

XIR_REGISTER_BUILT_IN_OP(strided_slice);
XIR_REGISTER_BUILT_IN_OP(strided_slice_fix);

auto priorbox =
    xir::OpDef("priorbox")
        .add_input_arg(xir::OpArgDef{
            "input", OpArgDef::REQUIRED_AND_REPEATED, xir::DataType::FLOAT,
            "`[batch, in_height, in_width, in_channels]`."})
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "min_sizes", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<float>`\n\n"
            "minimum box size (in pixels)"))
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "max_sizes", AttrDef::OPTIONAL, 0,
            "`Datatype`: `vector<float>`\n\n"
            "maximum box size (in pixels)",
            {}))
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "aspect_ratio", AttrDef::OPTIONAL, 0,
            "`Datatype`: `vector<float>`\n\n"
            "various of aspect ratios",
            {}))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "flip", AttrDef::REQUIRED,
            "`Datatype`: `bool`\n\n"
            "if true, will flip each aspect ratio, default True"))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "clip", AttrDef::REQUIRED,
            "`Datatype`: `bool`\n\n"
            "if true, will clip the prior so that it is within [0, "
            "1]."))
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "variance", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<float>`\n\n"
            "variance for adjusting the prior bboxes"))
        .add_attr(xir::AttrDefBuilder<std::vector<float>>::build(
            "step", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<float>`\n\n"
            "step size"))
        .add_attr(xir::AttrDefBuilder<float>::build(
            "offset", AttrDef::REQUIRED,
            "`Datatype`: `vector<float>`\n\n"
            "offset to the top left corner of each cell."))
        .set_shape_infer(xir::shape_infer_priorbox);
XIR_REGISTER_BUILT_IN_OP(priorbox);

auto stack =
    xir::OpDef("stack")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED_AND_REPEATED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<int>::build(
            "axis", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "Dimension along which to pack. Negative values wrap around, "
            "so the valid range is [-(R+1), R+1)"))
        .set_annotation(
            "Stacks a list of `rank-R` tensors into one `rank-(R+1)` "
            "tensor.\n\n"
            "For example, given a list of length N of tensors of shape `(A, B, "
            "C)`;\n\n"
            "if axis == 0 then the output tensor will have the shape `(N, A, "
            "B, C)`.\n\n"
            "if axis == 1 then the output tensor will have the shape `(A, N, "
            "B, C)`.")
        .set_shape_infer(xir::shape_infer_stack);
XIR_REGISTER_BUILT_IN_OP(stack);

auto matmul =
    xir::OpDef("matmul")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED_AND_REPEATED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"bias", OpArgDef::OPTIONAL,
                                     xir::DataType::FLOAT, "1-dimension"})
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "transpose_a", AttrDef::REQUIRED,
            "`Datatype`: `bool`\n\n"
            "If true, input[0] is transposed before multiplication.\n\n"
            "transpose(input[0]):\n\n"
            "    [..., a, b] -> [..., b, a]\n"))
        .add_attr(xir::AttrDefBuilder<bool>::build(
            "transpose_b", AttrDef::REQUIRED,
            "`Datatype`: `bool`\n\n"
            "If true, input[1] is transposed before multiplication.\n\n"
            "transpose(input[1]):\n\n"
            "    [..., b, c] -> [..., c, b]\n"))
        .set_annotation(
            "This operator is batched matmul.\n\n"
            "    input[0] : [..., a, b]\n"
            "    input[1] : [..., b, c]\n"
            "    output   : [..., a, c]\n"
            "    output[..., a, c] = sum_{i}\n"
            "                      input[0][..., a, i] * input[1][..., i, b]\n"
            "In this operator, ... denotes non-matrix dimensions,\n"
            "and non-matrix dimensions are broadcasted.\n"
            "For example,  if input[0].shape is `(1, j, m, n)`, "
            "and the other is `(k, 1, n, p)`, the out.shape would be `(k, j, "
            "m, p)`.")
        .set_shape_infer(xir::shape_infer_matmul);
XIR_REGISTER_BUILT_IN_OP(matmul);

auto gstiling =
    xir::OpDef("gstiling")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_attr(xir::AttrDefBuilder<bool>::build("reverse", AttrDef::REQUIRED,
                                                   "`Datatype`: `bool`\n\n"
                                                   "if reverse"))
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "stride", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "stride for feature maps"))
        .set_shape_infer(xir::shape_infer_gstiling);
XIR_REGISTER_BUILT_IN_OP(gstiling);

auto tile_fix =
    xir::OpDef("tile-fix")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::XINT,
                                     "4-dimension, N H W C"})
        .add_attr(xir::AttrDefBuilder<bool>::build("reverse", AttrDef::REQUIRED,
                                                   "`Datatype`: `bool`\n\n"
                                                   "if reverse"))
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "stride", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "stride for feature maps"))
        .set_shape_infer(xir::shape_infer_tile_fix);
XIR_REGISTER_BUILT_IN_OP(tile_fix);

auto eltwise_fix =
    xir::OpDef("eltwise-fix")
        .add_input_arg(xir::OpArgDef{
            "input", OpArgDef::REQUIRED_AND_REPEATED, xir::DataType::XINT,
            "The feature maps, can be x-dimension. "
            "eltwise-fix operator implements element-wise operations."})
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "nonlinear", AttrDef::OPTIONAL,
            "`Datatype`: `string`\n\n"
            "nonlinear type, \"NONE\", \"RELU\", \"PRELU\", "
            "\"LEAKYRELU\",\"RELU6\". Default is \"NONE\"",
            "NONE"))
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "type", AttrDef::OPTIONAL,
            "`Datatype`: `string`\n\n"
            "eltwise type, \"ADD\", \"MUL\". Default is \"ADD\"",
            "ADD"))
        .set_shape_infer(xir::shape_infer_eltwise_fix);

XIR_REGISTER_BUILT_IN_OP(eltwise_fix);

auto depthwise_fix =
    xir::OpDef("depthwise-fix")
        .add_input_arg(xir::OpArgDef{
            "input", OpArgDef::REQUIRED_AND_REPEATED, xir::DataType::XINT,
            "The feature maps, can be x-dimension. "
            "depthwise-fix operator implements channel-wise operations."})
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "type", AttrDef::OPTIONAL,
            "`Datatype`: `string`\n\n"
            "depthwise type, \"ADD\", \"MUL\". Default is \"ADD\"",
            "MUL"))
        .set_shape_infer(xir::shape_infer_depthwise_fix);

XIR_REGISTER_BUILT_IN_OP(depthwise_fix);

auto exp = xir::OpDef("exp")
               .add_input_arg(xir::OpArgDef{
                   "input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                   "The feature maps, can be x-dimension."})
               .set_annotation(
                   "This function computes the exponential of the input tensor "
                   "element-wise.\n\n"
                   "    f(x) = exp(x)\n")
               .set_shape_infer(xir::shape_infer_exp);
XIR_REGISTER_BUILT_IN_OP(exp);

auto neg = xir::OpDef("neg")
               .add_input_arg(xir::OpArgDef{
                   "input", OpArgDef::REQUIRED, xir::DataType::FLOAT,
                   "The feature maps, can be x-dimension."})
               .set_annotation(
                   "This function computes the numerical negative value "
                   "element-wise.\n\n"
                   "    f(x) = -x\n")
               .set_shape_infer(xir::shape_infer_neg);
XIR_REGISTER_BUILT_IN_OP(neg);

auto scale =
    xir::OpDef("scale")
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "The feature maps, can be x-dimension."})
        .add_input_arg(xir::OpArgDef{"scale", OpArgDef::REQUIRED,
                                     xir::DataType::FLOAT,
                                     "1-dimension, channel-wise."})
        .add_input_arg(xir::OpArgDef{"bias", OpArgDef::OPTIONAL,
                                     xir::DataType::FLOAT,
                                     "1-dimension, channel-wise."})
        .add_attr(xir::AttrDefBuilder<int>::build(
            "axis", AttrDef::OPTIONAL,
            "`Datatype`: `int`\n\n"
            "the axis of the input to implement scale",
            -1))
        .set_annotation(
            "This function computes the channel-wise dot product and adds the "
            "bias. For example, axis = -1:\n\n"
            "    output[b, h, w, c] = input[b, h, w, c] * scale[c] + bias[c]\n")
        .set_shape_infer(xir::shape_infer_scale);
XIR_REGISTER_BUILT_IN_OP(scale);

std::function<void(xir::OpDef&)> Correlation2dOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REPEATED,
                                     xir::DataType::FLOAT, ""})
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "pad_size", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "pad_size for feature maps"))
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "shape", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "The shape of the output tensor"))
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "data_type", AttrDef::REQUIRED,
            "`Datatype`: `string`\n\n"
            "The data type of the data of output feature maps, "
            "we use FLOAT32 as the default."))
        .set_annotation("This operator is not defined by XIR.")
        .set_shape_infer(xir::shape_infer_correlation2d_elemwise);
  };
}

auto correlation2d_elemwise =
    xir::OpDef("correlation2d_elemwise")
        .inherit_from(Correlation2dOpDefGenerator(xir::DataType::FLOAT));

auto correlation2d_elemwise_fix =
    xir::OpDef("correlation2d_elemwise-fix")
        .inherit_from(Correlation2dOpDefGenerator(xir::DataType::XINT));
XIR_REGISTER_BUILT_IN_OP(correlation2d_elemwise);
XIR_REGISTER_BUILT_IN_OP(correlation2d_elemwise_fix);

std::function<void(xir::OpDef&)> Correlation1dOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REPEATED,
                                     xir::DataType::FLOAT, ""})
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "pad_size", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "pad_size for feature maps"))
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "shape", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "The shape of the output tensor"))
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "data_type", AttrDef::REQUIRED,
            "`Datatype`: `string`\n\n"
            "The data type of the data of output feature maps, "
            "we use FLOAT32 as the default."))
        .set_annotation("This operator is not defined by XIR.")
        .set_shape_infer(xir::shape_infer_correlation1d_elemwise);
  };
}

auto correlation1d_elemwise =
    xir::OpDef("correlation1d_elemwise")
        .inherit_from(Correlation1dOpDefGenerator(xir::DataType::FLOAT));

auto correlation1d_elemwise_fix =
    xir::OpDef("correlation1d_elemwise-fix")
        .inherit_from(Correlation1dOpDefGenerator(xir::DataType::XINT));
XIR_REGISTER_BUILT_IN_OP(correlation1d_elemwise);
XIR_REGISTER_BUILT_IN_OP(correlation1d_elemwise_fix);

std::function<void(xir::OpDef&)> CostVolumeOpDefGenerator(
    xir::DataType::Type T) {
  return [=](xir::OpDef& op_def) {
    op_def
        .add_input_arg(xir::OpArgDef{"input", OpArgDef::REPEATED,
                                     xir::DataType::FLOAT, ""})
        .add_attr(xir::AttrDefBuilder<std::int32_t>::build(
            "maxdisp", AttrDef::REQUIRED,
            "`Datatype`: `int`\n\n"
            "maxdisp for feature maps"))
        .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
            "shape", AttrDef::REQUIRED, 0,
            "`Datatype`: `vector<int>`\n\n"
            "The shape of the output tensor"))
        .add_attr(xir::AttrDefBuilder<std::string>::build(
            "data_type", AttrDef::REQUIRED,
            "`Datatype`: `string`\n\n"
            "The data type of the data of output feature maps, "
            "we use FLOAT32 as the default."))
        .set_annotation("This operator is not defined by XIR.")
        .set_shape_infer(xir::shape_infer_cost_volume);
  };
}

auto cost_volume =
    xir::OpDef("cost_volume")
        .inherit_from(CostVolumeOpDefGenerator(xir::DataType::FLOAT));
auto cost_volume_fix =
    xir::OpDef("cost_volume-fix")
        .inherit_from(CostVolumeOpDefGenerator(xir::DataType::XINT));
XIR_REGISTER_BUILT_IN_OP(cost_volume);
XIR_REGISTER_BUILT_IN_OP(cost_volume_fix);
}  // namespace xir
