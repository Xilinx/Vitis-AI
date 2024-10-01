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
#include <memory>
#include <opencv2/core.hpp>
namespace xir {
class Graph;
class Attrs;
};  // namespace xir
#include "./dpu_task.hpp"
#include "./vitis/ai/proto/dpu_model_param.pb.h"
namespace vitis {
namespace ai {
class ConfigurableDpuTask {
 public:
  static std::unique_ptr<ConfigurableDpuTask> create(
      const std::string& model_name, bool need_preprocess = true);

  static std::unique_ptr<ConfigurableDpuTask> create(
      const std::string& model_name, xir::Attrs* attrs,
      bool need_preprocess = true);

 protected:
  explicit ConfigurableDpuTask();

 public:
  ConfigurableDpuTask(const ConfigurableDpuTask&) = delete;
  virtual ~ConfigurableDpuTask();

 public:
  virtual void setInputImageBGR(const cv::Mat& image) = 0;
  virtual void setInputImageRGB(const cv::Mat& image, size_t ind = 0) = 0;
  virtual void setInputDataArray(const std::vector<int8_t>& array, size_t ind = 0) = 0;
  virtual void setInputImageBGR(const std::vector<cv::Mat>& images) = 0;
  virtual void setInputImageRGB(const std::vector<cv::Mat>& images, size_t ind = 0) = 0;
  virtual void setInputDataArray(
      const std::vector<std::vector<int8_t>>& array, size_t ind = 0) = 0;
  virtual void run(int task_index) = 0;
  virtual void run_with_xrt_bo(
      const std::vector<vart::xrt_bo_t>& input_bos) = 0;
  virtual const vitis::ai::proto::DpuModelParam& getConfig() const = 0;
  virtual std::vector<std::vector<vitis::ai::library::InputTensor>>
  getInputTensor() const = 0;
  virtual std::vector<std::vector<vitis::ai::library::OutputTensor>>
  getOutputTensor() const = 0;
  /**
   * @brief Function to get InputWidth of the neural network (input image cols).
   *
   * @return InputWidth of the network
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeigth of the  neural network (input image
   *rows).
   *
   *@return InputHeight of the neural network.
   */
  virtual int getInputHeight() const = 0;

  virtual size_t get_input_batch() const = 0;
  virtual const xir::Graph* get_graph() const = 0;
  virtual int get_input_buffer_size() const = 0;
  virtual size_t get_input_offset() const = 0;
  virtual int get_input_fix_point() const = 0;
};

template <typename Interface>
class TConfigurableDpuTask : public Interface {
 public:
  explicit TConfigurableDpuTask(const std::string& model_name,
                                bool need_preprocess = true)
      : configurable_dpu_task_{
            ConfigurableDpuTask::create(model_name, need_preprocess)} {};
  explicit TConfigurableDpuTask(const std::string& model_name,
                                xir::Attrs* attrs, bool need_preprocess = true)
      : configurable_dpu_task_{
            ConfigurableDpuTask::create(model_name, attrs, need_preprocess)} {};
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

class ConfigurableDpuTaskBase {
 public:
  explicit ConfigurableDpuTaskBase(const std::string& model_name,
                                   bool need_preprocess = true);
  explicit ConfigurableDpuTaskBase(const std::string& model_name,
                                   xir::Attrs* attrs,
                                   bool need_preprocess = true);
  ConfigurableDpuTaskBase(const ConfigurableDpuTaskBase&) = delete;
  virtual ~ConfigurableDpuTaskBase();
  /**
   * @brief Function to get InputWidth of the network (input image
   * columns).
   *
   * @return InputWidth of the network
   */
  int getInputWidth() const {
    return get_input_width();  //
  }
  /**
   *@brief Function to get InputHeight of the network (input image
   *rows).
   *
   *@return InputHeight of the network.
   */
  int getInputHeight() const { return get_input_height(); }

  /**
   * @brief Function to get InputWidth of the network (input image
   * columns).
   *
   * @return InputWidth of the network.
   */
  int get_input_width() const;
  /**
   *@brief Function to get InputHeight of the network (input image
   *rows).
   *
   *@return InputHeight of the network.
   */
  int get_input_height() const;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  size_t get_input_batch() const;

  /**
   * @brief The total size of the buffer including input and padding.
   *
   * For DPU model with one input, input data should be organized as one buffer
   * with padding
   *
   * | padding input |
   *
   * For input, there is an offset which take the start address of padding as
   * the base.
   *
   * @return The total size of the input buffer including input and padding,
   * return -1 if zero copy is not supported.
   */
  int get_input_buffer_size() const;

  /**
   * @brief Function to get input offset for zero copy.
   *
   * @return Input offset
   */
  size_t get_input_offset() const;

  /**
   * @brief Function to get input tensor's fix point of the network.
   *
   * @return Input fix point.
   */
  int get_input_fix_point() const;

 protected:
  std::unique_ptr<ConfigurableDpuTask> configurable_dpu_task_;
};

}  // namespace ai
}  // namespace vitis
