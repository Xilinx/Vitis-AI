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
#include "xilinx/ai/configurable_dpu_task.hpp"
#include "xilinx/ai/dpu_task.hpp"

namespace xilinx {
namespace ai {

class ConfigurableDpuTaskImp : public ConfigurableDpuTask {
 public:
  ConfigurableDpuTaskImp(const std::string& model_name,
                         bool need_preprocess = true);
  virtual ~ConfigurableDpuTaskImp();

 private:
  virtual void setInputImageBGR(const cv::Mat& image) override;
  virtual void setInputImageRGB(const cv::Mat& image) override;
  virtual void run(int task_index) override;
  virtual const xilinx::ai::proto::DpuModelParam& getConfig() const override;
  virtual std::vector<std::vector<xilinx::ai::InputTensor>> getInputTensor()
      const override;
  virtual std::vector<std::vector<xilinx::ai::OutputTensor>> getOutputTensor()
      const override;

  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual const vitis::ai::DpuMeta& get_dpu_meta_info() const override;

 private:
  std::unique_ptr<DpuTask> tasks_;
  xilinx::ai::proto::DpuModelParam model_;
};
}  // namespace ai
}  // namespace xilinx
