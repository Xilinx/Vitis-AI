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
#include <memory>
#include <opencv2/core.hpp>
#include "./dpu_task.hpp"
#include "./vitis/ai/proto/dpu_model_param.pb.h"
namespace vitis {
namespace ai {
class ConfigurableDpuTask {
 public:
  static std::unique_ptr<ConfigurableDpuTask> create(
      const std::string& model_name, bool need_preprocess = true);

 protected:
  explicit ConfigurableDpuTask();

 public:
  ConfigurableDpuTask(const ConfigurableDpuTask&) = delete;
  virtual ~ConfigurableDpuTask();

 public:
  virtual void setInputImageBGR(const cv::Mat& image) = 0;
  virtual void setInputImageRGB(const cv::Mat& image) = 0;
  virtual void setInputImageBGR(const std::vector<cv::Mat>& images) = 0;
  virtual void setInputImageRGB(const std::vector<cv::Mat>& images) = 0;
  virtual void run(int task_index) = 0;
  virtual const vitis::ai::proto::DpuModelParam& getConfig() const = 0;
  virtual std::vector<std::vector<vitis::ai::library::InputTensor>>
  getInputTensor() const = 0;
  virtual std::vector<std::vector<vitis::ai::library::OutputTensor>>
  getOutputTensor() const = 0;
  /**
   * @brief Function to get InputWidth of the neural network (input image cols).
   *
   * @return InputWidth of the facedetect network
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeigth of the  neural network (input image
   *rows).
   *
   *@return InputHeight of the neural network.
   */
  virtual int getInputHeight() const = 0;
  /**
   * @brief get meta info a model
   * */
  virtual const vitis::ai::DpuMeta& get_dpu_meta_info() const = 0;
  virtual size_t get_input_batch() const = 0;
};

template <typename Interface>
class TConfigurableDpuTask : public Interface {
 public:
  explicit TConfigurableDpuTask(const std::string& model_name,
                                bool need_preprocess = true)
      : configurable_dpu_task_{
            ConfigurableDpuTask::create(model_name, need_preprocess)} {};
  TConfigurableDpuTask(const TConfigurableDpuTask&) = delete;
  virtual ~TConfigurableDpuTask(){};
  virtual int getInputWidth() const override {
    return configurable_dpu_task_->getInputWidth();
  }
  virtual int getInputHeight() const override {
    return configurable_dpu_task_->getInputHeight();
  }
  virtual size_t get_input_batch() const override {
    return configurable_dpu_task_->get_input_batch();
  }

 protected:
  std::unique_ptr<ConfigurableDpuTask> configurable_dpu_task_;
};
}  // namespace ai
}  // namespace vitis
