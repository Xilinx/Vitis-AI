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
#pragma once

#include "vitis/ai/yolov6.hpp"

namespace vitis {
namespace ai {

class YOLOv6Imp : public YOLOv6 {
 public:
  YOLOv6Imp(const std::string& model_name, bool need_preprocess = true);
  YOLOv6Imp(const std::string& model_name, xir::Attrs* attrs,
            bool need_preprocess = true);
  virtual ~YOLOv6Imp();

 private:
  virtual YOLOv6Result run(const cv::Mat& image) override;
  virtual std::vector<YOLOv6Result> run(
      const std::vector<cv::Mat>& image) override;
  void letterbox(const cv::Mat& im, int w, int h, cv::Mat& om, float& scale);
  void letterbox(const cv::Mat& im, int w, int h, int load_size, cv::Mat& om,
                 float& scale, int& left, int& top);
};

}  // namespace ai
}  // namespace vitis
