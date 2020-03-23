/*
 * Copyright 2019 Xilinx Inc.
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
#include "vitis/ai/classification.hpp"

namespace vitis {
namespace ai {

class ClassificationImp : public TConfigurableDpuTask<Classification> {
 public:
  ClassificationImp(const std::string& model_name, bool need_preprocess = true);
  virtual ~ClassificationImp();

 private:
  virtual ClassificationResult run(const cv::Mat& image) override;
  virtual std::vector<ClassificationResult> run(
      const std::vector<cv::Mat>& images) override;

 private:
  int preprocess_type;
  const int TOP_K;
  const bool test_accuracy;
};

}  // namespace ai
}  // namespace vitis
