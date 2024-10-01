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
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/lanedetect.hpp>

namespace vitis {
namespace ai {
class RoadLineImp : public vitis::ai::TConfigurableDpuTask<RoadLine> {
 public:
  RoadLineImp(const std::string& model_name, bool need_preprocess = true);
  virtual ~RoadLineImp();

 private:
  virtual RoadLineResult run(const cv::Mat& image) override;
  virtual std::vector<RoadLineResult> run(
      const std::vector<cv::Mat>& image) override;
  std::string model_name_;
  std::unique_ptr<RoadLinePostProcess> processor_;
};

}  // namespace ai
}  // namespace vitis
