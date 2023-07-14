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

#include "vitis/ai/arflow.hpp"

namespace vitis {
namespace ai {

class ARFlowImp : public ARFlow {
 public:
  ARFlowImp(const std::string& model_name, bool need_preprocess = true);
  virtual ~ARFlowImp();

 private:
  virtual std::vector<vitis::ai::library::OutputTensor> run(
      const cv::Mat& image_1, const cv::Mat& image_2) override;
  virtual std::vector<vitis::ai::library::OutputTensor> run(
      const std::vector<cv::Mat>& image_1,
      const std::vector<cv::Mat>& image_2) override;
};

}  // namespace ai
}  // namespace vitis
