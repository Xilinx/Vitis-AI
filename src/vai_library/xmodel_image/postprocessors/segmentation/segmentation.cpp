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
#include <glog/logging.h>

#include <cstddef>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>

#include "vart/runner_helper.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "xir/graph/graph.hpp"
DEF_ENV_PARAM(DEBUG_XMODLE_IMAGE_SEGMEMTATION, "0");

namespace {

struct Segmentation {
 public:
  static xir::OpDef get_op_def() {
    return xir::OpDef("segmentation")  //
        .add_input_arg(xir::OpArgDef{
            "input", xir::OpArgDef::REQUIRED, xir::DataType::Type::FLOAT,
            "segmentaion "
            "`[batch, in_height, in_width, in_channels]`."})
        .set_annotation("postprocessor for segmetation");
  }

  explicit Segmentation(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    auto input_shape = args.graph_input_tensor->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    height_ = input_shape[1];
    width_ = input_shape[2];
  }
  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& input);

 private:
  int width_;
  int height_;
};

vitis::ai::proto::DpuModelResult Segmentation::process(
    const vart::simple_tensor_buffer_t<float>& input) {
  //LOG(INFO) << "segmentation";
  auto input_shape = input.tensor->get_shape();
  //LOG(INFO) << "bit width" << input.tensor->get_data_type().bit_width;
  CHECK_EQ(input_shape.size(), 4u);
  size_t height = input_shape[1];
  size_t width = input_shape[2];
  // how to get_tensor_format and get channel num??
  size_t channel = input_shape[3];

  size_t col_ind = 0;
  size_t row_ind = 0;
  if (ENV_PARAM(DEBUG_XMODLE_IMAGE_SEGMEMTATION)) {
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    CHECK(
        std::ofstream("xmodel_image_before_post_process.bin", mode)
            .write((char*)input.data, sizeof(float) * height * width * channel)
            .good())
        << " faild to dump";
  }

  auto ret = vitis::ai::proto::DpuModelResult();
  auto segmenation_result = ret.mutable_segmentation_result();
  for (size_t i = 0; i < height * width * channel; i = i + channel) {
    auto max_ind = std::max_element(input.data + i, input.data + i + channel);
    uint8_t posit = std::distance(input.data + i, max_ind);
    *(segmenation_result->mutable_data()->Add()) = posit;
    col_ind++;
    if (col_ind > width - 1) {
      row_ind++;
      col_ind = 0;
    }
  }

  if (ENV_PARAM(DEBUG_XMODLE_IMAGE_SEGMEMTATION)) {
    col_ind = 0;
    row_ind = 0;
    cv::Mat segMat(height, width, CV_8UC1);
    for (size_t i = 0; i < height * width; i++) {
      segMat.at<uchar>(row_ind, col_ind) =
          10 * (uchar)(segmenation_result->data(i));
      col_ind++;
      if (col_ind > width - 1) {
        row_ind++;
        col_ind = 0;
      }
    }
    cv::imwrite("xmodel_image_8UC1.jpg",
                segMat);  // Save the result as an image;
  }

  return ret;
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<Segmentation>>();
}
