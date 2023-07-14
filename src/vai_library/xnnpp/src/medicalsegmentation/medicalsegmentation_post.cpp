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
#include "medicalsegmentation_post.hpp"

#include <sys/stat.h>

#include <vitis/ai/image_util.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

MedicalSegmentationPost::~MedicalSegmentationPost() {}

MedicalSegmentationPost::MedicalSegmentationPost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config,
    int& real_batch_sizex)
    : input_tensors_(input_tensors), output_tensors_(output_tensors), real_batch_size(real_batch_sizex) {}

std::vector<vitis::ai::MedicalSegmentationResult>
MedicalSegmentationPost::medicalsegmentation_post_process() {
  auto ret = std::vector<vitis::ai::MedicalSegmentationResult>{};
  ret.reserve(real_batch_size);

  for (auto i = 0; i < real_batch_size; ++i) {
    ret.emplace_back(medicalsegmentation_post_process(i));
  }
  return ret;
}

vitis::ai::MedicalSegmentationResult
MedicalSegmentationPost::medicalsegmentation_post_process(
    unsigned int batch_idx) {
  std::vector<cv::Mat> segMatV(5);

  for (int j = 0; j < 5; ++j) {
    unsigned int col_ind = 0;
    unsigned int row_ind = 0;
    auto output_layer = output_tensors_[j];
    cv::Mat segMat(output_layer.height, output_layer.width, CV_8UC1);

    for (size_t i = 0;
         i < output_layer.height * output_layer.width * output_layer.channel;
         i = i + output_layer.channel) {
      auto max_ind =
          std::max_element(((int8_t*)output_layer.get_data(batch_idx)) + i,
                           ((int8_t*)output_layer.get_data(batch_idx)) + i +
                               output_layer.channel);
      uint8_t posit = std::distance(
          ((int8_t*)output_layer.get_data(batch_idx)) + i, max_ind);
      segMat.at<uchar>(row_ind, col_ind) = posit;
      col_ind++;
      if (col_ind > output_layer.width - 1) {
        row_ind++;
        col_ind = 0;
      }
    }
    segMatV[j] = segMat;
  }
  return MedicalSegmentationResult{(int)input_tensors_[0].width,
                                   (int)input_tensors_[0].height, segMatV};
}

}  // namespace ai
}  // namespace vitis
