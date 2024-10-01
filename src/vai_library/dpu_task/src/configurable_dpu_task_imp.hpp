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
#include "vitis/ai/configurable_dpu_task.hpp"
#include "vitis/ai/dpu_task.hpp"

namespace vitis {
namespace ai {

class ConfigurableDpuTaskImp : public ConfigurableDpuTask {
 public:
  ConfigurableDpuTaskImp(const std::string& model_name,
                         bool need_preprocess = true);
  ConfigurableDpuTaskImp(const std::string& model_name, xir::Attrs* attrs,
                         bool need_preprocess = true);
  virtual ~ConfigurableDpuTaskImp();

 private:
  virtual void setInputImageBGR(const cv::Mat& image) override;
  virtual void setInputImageRGB(const cv::Mat& image, size_t ind) override;
  virtual void setInputImageBGR(const std::vector<cv::Mat>& image) override;
  virtual void setInputImageRGB(const std::vector<cv::Mat>& image, size_t ind) override;
  virtual void setInputDataArray(const std::vector<int8_t>& array, size_t ind) override;
  virtual void setInputDataArray(
      const std::vector<std::vector<int8_t>>& array, size_t ind) override;
  virtual void run(int task_index) override;
  virtual void run_with_xrt_bo(
      const std::vector<vart::xrt_bo_t>& input_bos) override;
  virtual const vitis::ai::proto::DpuModelParam& getConfig() const override;
  virtual std::vector<std::vector<vitis::ai::library::InputTensor>>
  getInputTensor() const override;
  virtual std::vector<std::vector<vitis::ai::library::OutputTensor>>
  getOutputTensor() const override;

  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;

  virtual size_t get_input_batch() const override;
  virtual const xir::Graph* get_graph() const override;

  virtual int get_input_buffer_size() const override;
  virtual size_t get_input_offset() const override;
  virtual int get_input_fix_point() const override;

 private:
  std::unique_ptr<xir::Attrs> default_attrs_;
  vitis::ai::proto::DpuModelParam model_;
  std::unique_ptr<DpuTask> tasks_;
};

}  // namespace ai
}  // namespace vitis
