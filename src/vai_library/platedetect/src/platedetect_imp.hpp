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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/platedetect.hpp>

namespace vitis {
namespace ai {

class PlateDetectImp : public vitis::ai::TConfigurableDpuTask<PlateDetect> {
 public:
  explicit PlateDetectImp(const std::string &model_name, bool need_preprocess);
  explicit PlateDetectImp(const std::string &model_name, xir::Attrs *attrs, bool need_preprocess);
  virtual ~PlateDetectImp();

 private:
  virtual PlateDetectResult run(const cv::Mat &image) override;
  virtual std::vector<PlateDetectResult> run(
      const std::vector<cv::Mat> &img) override;
};
}  // namespace ai
}  // namespace vitis
