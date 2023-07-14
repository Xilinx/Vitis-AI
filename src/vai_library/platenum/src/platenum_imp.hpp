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
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/platenum.hpp>

namespace vitis {
namespace ai {

class PlateNumImp : public vitis::ai::TConfigurableDpuTask<PlateNum> {
 public:
  explicit PlateNumImp(const std::string& model_name,
                       bool need_preprocess = true);
  explicit PlateNumImp(const std::string& model_name,
                       xir::Attrs *attrs,
                       bool need_preprocess = true);
  virtual ~PlateNumImp();

 private:
  virtual PlateNumResult run(const cv::Mat& image) override;
  virtual std::vector<PlateNumResult> run(
      const std::vector<cv::Mat>& imgs) override;
  std::vector<int> sub_x_;
  std::vector<int> sub_y_;
};
}  // namespace ai
}  // namespace vitis
