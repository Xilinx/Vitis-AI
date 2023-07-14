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

#include <memory>
#include <opencv2/core.hpp>

namespace vitis {
namespace ai {

class FadNet {
 public:
  static std::unique_ptr<FadNet> create();

 protected:
  explicit FadNet();
  FadNet(const FadNet&) = delete;
  FadNet& operator=(const FadNet&) = delete;

 public:
  virtual ~FadNet();
  virtual std::vector<cv::Mat> run(
      const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) = 0;
  virtual size_t get_input_batch() = 0;
  virtual int get_input_width() const = 0;
  virtual int get_input_height() const = 0;
};

}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:

