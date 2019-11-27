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
#include <xilinx/ai/configurable_dpu_task.hpp>
#include <xilinx/ai/multitask.hpp>

namespace xilinx {
namespace ai {
class MultiTaskImp : public xilinx::ai::TConfigurableDpuTask<MultiTask> {
public:
  MultiTaskImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~MultiTaskImp();

private:
  virtual MultiTaskResult run_8UC1(const cv::Mat &image) override;
  virtual MultiTaskResult run_8UC3(const cv::Mat &image) override;

private:
  void run_it(const cv::Mat &image);

private:
  std::unique_ptr<MultiTaskPostProcess> processor_;
};
} // namespace ai
} // namespace xilinx
