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
#include <vitis/ai/multitask.hpp>

namespace vitis {
namespace ai {

class MultiTaskImp : public vitis::ai::TConfigurableDpuTask<MultiTask> {
 public:
  MultiTaskImp(const std::string& model_name, bool need_preprocess = true);
  virtual ~MultiTaskImp();

 private:
  virtual MultiTaskResult run_8UC1(const cv::Mat& image) override;
  virtual std::vector<MultiTaskResult> run_8UC1(
      const std::vector<cv::Mat>& images) override;
  virtual MultiTaskResult run_8UC3(const cv::Mat& image) override;
  virtual std::vector<MultiTaskResult> run_8UC3(
      const std::vector<cv::Mat>& images) override;

 private:
  void run_it(const cv::Mat& image);
  void run_it(const std::vector<cv::Mat>& images);

 private:
  std::unique_ptr<MultiTaskPostProcess> processor_;
};

}  // namespace ai
}  // namespace vitis
