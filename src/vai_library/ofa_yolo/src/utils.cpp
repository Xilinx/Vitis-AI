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

#include "utils.hpp"

using cv::Mat;
using cv::Size;

namespace vitis {
namespace ai {
namespace ofa_yolo {

cv::Mat letterbox(const cv::Mat& im, int w, int h) {
  float scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);

  int new_w = round(im.cols * scale);
  int new_h = round(im.rows * scale);

  Mat img_res;
  if (im.size() != Size(new_w, new_h)) {
    cv::resize(im, img_res, Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  } else {
    img_res = im;
  }
  auto dw = float(w - new_w) / 2.0f;
  auto dh = float(h - new_h) / 2.0f;

  Mat new_img(Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
  copyMakeBorder(img_res, new_img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                 int(round(dw - 0.1)), int(round(dw + 0.1)),
                 cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  return new_img;
}

}  // namespace ofa_yolo
}  // namespace ai
}  // namespace vitis
