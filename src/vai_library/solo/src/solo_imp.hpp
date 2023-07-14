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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/solo.hpp>
#include "input.hpp"

using namespace cv;
using namespace std;

namespace vitis {
namespace ai {
class SoloImp : public Solo {
 public:
  SoloImp(const std::string& model_name, bool need_preprocess = true);
  virtual ~SoloImp();

 private:
  virtual SoloResult run(const cv::Mat& image) override;
  virtual std::vector<SoloResult> run(
      const std::vector<cv::Mat>& image) override;
  std::vector<int> sub_x_in_; 
  std::vector<int> sub_y_in_; 
  std::vector<int> sub_x_out_; 
  std::vector<int> sub_y_out_; 
  std::vector<vector<int8_t>> const_input_data_;
  std::vector<std::vector<vector<int8_t>>> const_input_data_batch_;
  std::vector<vector<float>> float_input_data_;
};

}  // namespace ai
}  // namespace vitis
