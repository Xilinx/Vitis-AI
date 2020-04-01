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

#include "vitis/ai/nnpp/segmentation.hpp"
#include <queue>
#include <vector>

namespace vitis {
namespace ai {

static void convert_color(std::string& src, std::vector<uint8_t>& dest) {
  size_t pos = 0;
  while ((pos = src.find_first_of("0123456789", pos)) != std::string::npos) {
    dest.push_back(std::stoi(std::string(src, pos)));
    pos = src.find_first_of(" ", pos);
  }
}

static void load_color(const vitis::ai::proto::DpuModelParam& config,
                       std::vector<uint8_t>& color_c1,
                       std::vector<uint8_t>& color_c2,
                       std::vector<uint8_t>& color_c3) {
  std::string scolor1{config.segmentation_param().color1()};
  std::string scolor2{config.segmentation_param().color2()};
  std::string scolor3{config.segmentation_param().color3()};
  convert_color(scolor1, color_c1);
  convert_color(scolor2, color_c2);
  convert_color(scolor3, color_c3);
}

SegmentationResult segmentation_post_process_8UC1(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    size_t batch_idx) {
  unsigned int col_ind = 0;
  unsigned int row_ind = 0;

  auto output_layer = output_tensors[0];
  cv::Mat segMat(output_layer.height, output_layer.width, CV_8UC1);
  for (size_t i = 0; i < output_layer.height * output_layer.width * output_layer.channel; i = i + output_layer.channel) {
    auto max_ind = std::max_element(
        ((int8_t*)output_layer.get_data(batch_idx)) + i,
        ((int8_t*)output_layer.get_data(batch_idx)) + i + output_layer.channel);
    uint8_t posit = std::distance(((int8_t*)output_layer.get_data(batch_idx)) + i, max_ind);
    segMat.at<uchar>(row_ind, col_ind) = posit;
    col_ind++;
    if (col_ind > output_layer.width - 1) {
      row_ind++;
      col_ind = 0;
    }
  }
  return SegmentationResult{(int)input_tensors[0].width,
                            (int)input_tensors[0].height, segMat};
}

std::vector<SegmentationResult> segmentation_post_process_8UC1(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors) {
  auto batch_size = input_tensors[0].batch;
  auto ret = std::vector<SegmentationResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(
        segmentation_post_process_8UC1(input_tensors, output_tensors, i));
  }
  return ret;
}

SegmentationResult segmentation_post_process_8UC3(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config,
    size_t batch_idx) {
  std::vector<uint8_t> color_c1;
  std::vector<uint8_t> color_c2;
  std::vector<uint8_t> color_c3;
  load_color(config, color_c1, color_c2, color_c3);
  unsigned int col_ind = 0;
  unsigned int row_ind = 0;
  auto output_layer = output_tensors[0];
  cv::Mat segMat(output_layer.height, output_layer.width, CV_8UC3);
  for (size_t i = 0; i < output_layer.height * output_layer.width * output_layer.channel; i = i + output_layer.channel) {
    auto max_ind = std::max_element(
        ((int8_t*)output_layer.get_data(batch_idx)) + i,
        ((int8_t*)output_layer.get_data(batch_idx)) + i + output_layer.channel);
    uint8_t posit = std::distance(((int8_t*)output_layer.get_data(batch_idx)) + i, max_ind);
    segMat.at<cv::Vec3b>(row_ind, col_ind) =
        cv::Vec3b(color_c1[posit], color_c2[posit], color_c3[posit]);
    col_ind++;
    if (col_ind > output_layer.width - 1) {
      row_ind++;
      col_ind = 0;
    }
  }
  return SegmentationResult{(int)input_tensors[0].width,
                            (int)input_tensors[0].height, segMat};
}

std::vector<SegmentationResult> segmentation_post_process_8UC3(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  auto batch_size = input_tensors[0].batch;
  auto ret = std::vector<SegmentationResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(
        segmentation_post_process_8UC3(input_tensors, output_tensors, config, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
