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
#ifndef OPENPOSE_UTIL_HPP
#define OPENPOSE_UTIL_HPP

#include "vitis/ai/nnpp/reid.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

namespace vitis {
namespace ai {

float bn(float input, float weight, float mean, float var) {
  return ((input - mean) / sqrt(var + 1e-5)) * weight;
}

ReidResult reid_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx) {
  int sWidth = input_tensors[0][0].width;
  int sHeight = input_tensors[0][0].height;

//# DPUV1 needs input data as float
#ifdef ENABLE_DPUCADX8G_RUNNER
  float* data = (float*)output_tensors[0][0].get_data(batch_idx);
#else
  int8_t* data = (int8_t*)output_tensors[0][0].get_data(batch_idx);
#endif
  float scale = vitis::ai::library::tensor_scale(output_tensors[0][0]);
  int channels = output_tensors[0][0].channel;
  int width = output_tensors[0][0].width;
  int height = output_tensors[0][0].height;
  auto bn_means = vector<float>(config.reid_param().bn_means().begin(),
                                config.reid_param().bn_means().end());
  auto bn_weights = vector<float>(config.reid_param().bn_weights().begin(),
                                  config.reid_param().bn_weights().end());
  auto bn_vars = vector<float>(config.reid_param().bn_vars().begin(),
                               config.reid_param().bn_vars().end());
  // std::cout<<bn_means.size()<<" "<<bn_weights.size()<<" "<<bn_vars.size()<<"
  // "<<width<<" "<<height<<" "<<channels<<std::endl;
  float a[channels];
  float sum;
  for (int c = 0; c < channels; c++) {
    sum = 0;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        sum += data[h * width * channels + w * channels + c] * scale;
      }
    }
    a[c] = sum / (height * width);
    a[c] = bn(a[c], bn_weights[c], bn_means[c], bn_vars[c]);
  }
  Mat x = Mat(1, channels, CV_32F, a);
  Mat feat;
  normalize(x, feat);
  ReidResult result{sWidth, sHeight, feat};
  return result;
}

std::vector<ReidResult> reid_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  auto batch = input_tensors[0][0].batch;
  auto ret = std::vector<ReidResult>{};
  ret.reserve(batch);
  for (auto i = 0u; i < batch; i++) {
    ret.emplace_back(
        reid_post_process(input_tensors, output_tensors, config, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
#endif
