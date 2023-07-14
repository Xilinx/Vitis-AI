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
#ifndef HOURGLASS_HPP
#define HOURGLASS_HPP

#include "vitis/ai/nnpp/hourglass.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>

using namespace std;

namespace vitis {
namespace ai {

int find_point(cv::Mat ori_img, cv::Point2f& point) {
  int ret = 0;
  float score = 0;
  for (int x = 1; x < ori_img.cols - 1; ++x)
    for (int y = 1; y < ori_img.rows - 1; ++y) {
      {
        if (ori_img.at<float>(y, x) <= 0.001) continue;
        if (ori_img.at<float>(y, x) > score) {
          ret = 1;
          score = ori_img.at<float>(y, x);
          point = cv::Point(x, y);
        }
      }
    }
  return ret;
}

HourglassResult hourglass_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const int ori_w,
    const int ori_h, size_t batch_idx) {
  int sWidth = input_tensors[0].width;
  int sHeight = input_tensors[0].height;
  float scale_x = float(ori_w) / float(sWidth);
  float scale_y = float(ori_h) / float(sHeight);
  std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  /* Get channel count of the output Tensor for FC Task  */
  output_tensors_.emplace_back(output_tensors[0]);

  int8_t* data = (int8_t*)output_tensors_[0].get_data(batch_idx);
  float outscale = vitis::ai::library::tensor_scale(output_tensors_[0]);
  int channel = output_tensors_[0].channel;
  int w = output_tensors_[0].width;
  int h = output_tensors_[0].height;
  int size = output_tensors_[0].size;

  vector<float> chwdata;
  chwdata.reserve(size);
  for (int ih = 0; ih < h; ++ih)
    for (int iw = 0; iw < w; ++iw)
      for (int ic = 0; ic < channel; ++ic) {
        int offset = ic * w * h + ih * w + iw;
        chwdata[offset] = data[ih * w * channel + iw * channel + ic] * outscale;
      }
  HourglassResult::PosePoint posePoint;
  vector<HourglassResult::PosePoint> pose(16, posePoint);
  for (int i = 0; i < channel; ++i) {
    cv::Mat um(h, w, CV_32F, chwdata.data() + i * w * h);
    resize(um, um, cv::Size(0, 0), 4, 4, CV_INTER_CUBIC);
    cv::Point2f point;
    if (find_point(um, point) < 1) continue;
    point.x *= scale_x;
    point.y *= scale_y;
    pose[i].type = 1;
    pose[i].point = point;
  }
  HourglassResult result{sWidth, sHeight, pose};
  return result;
}

std::vector<HourglassResult> hourglass_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int>& ws,
    const std::vector<int>& hs) {
  auto batch = input_tensors[0].batch;
  auto ret = std::vector<HourglassResult>{};
  ret.reserve(batch);
  for (auto i = 0u; i < batch; i++) {
    ret.emplace_back(hourglass_post_process(input_tensors, output_tensors,
                                            config, ws[i], hs[i], i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
#endif
