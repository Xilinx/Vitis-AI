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
/*
 * Filename: facedetect.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <utility>

namespace vitis {
namespace ai {

struct HFnetResult {
  std::vector<std::vector<float>> descriptor;
  std::vector<std::pair<float, float>> keypoints;
  std::vector<float> vlad;
  float scale_w;
  float scale_h;
};

class HFnet {
 public:
  static std::unique_ptr<HFnet> create(const std::string& model_name);

 protected:
  explicit HFnet(const std::string& model_name);
  HFnet(const HFnet&) = delete;
  HFnet& operator=(const HFnet&) = delete;

 public:
  virtual ~HFnet();
  virtual std::vector<HFnetResult> run(const std::vector<cv::Mat>& imgs) = 0;
  //virtual std::vector<cv::Mat> get_result() = 0;
  virtual size_t get_input_batch() = 0;
  virtual int getInputWidth() const = 0;
  virtual int getInputHeight() const = 0;
};

}  // namespace ai
}  // namespace vitis


