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
#ifndef OPENPOSE_UTIL_HPP
#define OPENPOSE_UTIL_HPP

#include "vitis/ai/nnpp/rcan.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

namespace vitis {
namespace ai {

static uint8_t op_move_left(int8_t i) {
  int16_t ret = i << 1;
  if (ret < 0) {
    ret = 0;
  }
  return uint8_t(ret);
}

RcanResult rcan_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    size_t batch_idx) {
  int sWidth = input_tensors[0][0].width;
  int sHeight = input_tensors[0][0].height;

  int8_t* data = (int8_t*)output_tensors[0][0].get_data(batch_idx);
  // float scale = vitis::ai::library::tensor_scale(output_tensors[0][0]);
  int channels = output_tensors[0][0].channel;
  int width = output_tensors[0][0].width;
  int height = output_tensors[0][0].height;
  Mat result_img = Mat(height, width, CV_8UC3);
  transform(data, data + height * width * channels, result_img.data,
            op_move_left);
  RcanResult result{sWidth, sHeight, result_img};
  return result;
}

std::vector<RcanResult> rcan_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors) {
  auto batch = input_tensors[0][0].batch;
  auto ret = std::vector<RcanResult>{};
  ret.reserve(batch);
  for (auto i = 0u; i < batch; i++) {
    ret.emplace_back(rcan_post_process(input_tensors, output_tensors, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
#endif
